"""
Consumer-patched data preparation for autoresearch experiments.
Supports TinyStories (default) and ClimbMix datasets with configurable parameters.

Based on upstream prepare.py with consumer-friendly modifications:
- CLI args for dataset selection, shard count, sequence length, vocab size
- TinyStories download from HuggingFace (karpathy/tinystories-gpt4-clean)
- Configurable MAX_SEQ_LEN, EVAL_TOKENS, VOCAB_SIZE via tier config or CLI
- Separate cache directory (~/.cache/tuneforge/)

Usage:
    python prepare_consumer.py                                  # TinyStories, default settings
    python prepare_consumer.py --dataset climbmix --num-shards 64
    python prepare_consumer.py --tier-config configs/tier1-8gb.json
    python prepare_consumer.py --dataset tinystories --max-seq-len 512 --vocab-size 4096
"""

import os
import sys
import time
import math
import json
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Defaults (overridden by tier config or CLI args)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024       # context length (smaller default for consumer GPUs)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_TOKENS = 524288      # number of tokens for val eval (consumer-friendly: ~500K)
VOCAB_SIZE = 8192         # BPE vocabulary size

# ---------------------------------------------------------------------------
# Cache directory (separate from upstream to avoid conflicts)
# ---------------------------------------------------------------------------

PRIMARY_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tuneforge")
LEGACY_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-consumer")
CACHE_DIR = PRIMARY_CACHE_DIR if os.path.isdir(PRIMARY_CACHE_DIR) or not os.path.isdir(LEGACY_CACHE_DIR) else LEGACY_CACHE_DIR
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# ---------------------------------------------------------------------------
# ClimbMix dataset config (original upstream dataset)
# ---------------------------------------------------------------------------

CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_SHARD = 6542
CLIMBMIX_VAL_SHARD = 6542  # pinned validation shard

# ---------------------------------------------------------------------------
# TinyStories dataset config (consumer-friendly, small download)
# ---------------------------------------------------------------------------

TINYSTORIES_REPO = "roneneldan/TinyStories"
TINYSTORIES_BASE_URL = "https://huggingface.co/api/datasets/roneneldan/TinyStories/parquet/default"

# ---------------------------------------------------------------------------
# BPE tokenizer config (same as upstream)
# ---------------------------------------------------------------------------

# NOTE: pickle is used here intentionally — it is required by the upstream
# tiktoken library for serializing the trained BPE encoding object.
# This matches the upstream prepare.py behavior exactly.

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Active dataset state (set by configure_dataset)
# ---------------------------------------------------------------------------

DATASET = "tinystories"
VAL_FILENAME = None  # set by configure_dataset


def configure_dataset(dataset, num_shards_override=None):
    """Configure global dataset parameters based on dataset choice."""
    global DATASET, DATA_DIR, VAL_FILENAME

    DATASET = dataset
    DATA_DIR = os.path.join(CACHE_DIR, "data", dataset)

    if dataset == "tinystories":
        VAL_FILENAME = "val_shard.parquet"
    elif dataset == "climbmix":
        VAL_FILENAME = f"shard_{CLIMBMIX_VAL_SHARD:05d}.parquet"


# ---------------------------------------------------------------------------
# TinyStories download
# ---------------------------------------------------------------------------

def _discover_tinystories_shards():
    """Discover available parquet files in the TinyStories dataset.
    roneneldan/TinyStories has: train/0-3.parquet, validation/0.parquet
    We use the HF parquet API endpoint which serves files directly."""
    train_shards = [f"train/{i}.parquet" for i in range(4)]
    val_shards = ["validation/0.parquet"]
    return train_shards, val_shards


def _download_one_tinystories(shard_name):
    """Download a single TinyStories shard. Top-level for pickle compatibility."""
    # shard_name is like "train/0.parquet" — flatten to "train-0.parquet" for local storage
    local_name = shard_name.replace("/", "-")
    local_path = os.path.join(DATA_DIR, local_name)
    if os.path.exists(local_path):
        return True
    url = f"{TINYSTORIES_BASE_URL}/{shard_name}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = local_path + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, local_path)
            print(f"    Downloaded {shard_name}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"    Attempt {attempt}/{max_attempts} failed for {shard_name}: {e}")
            for path in [local_path + ".tmp", local_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_tinystories(num_shards, download_workers=4):
    """Download TinyStories parquet shards from HuggingFace."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_shards, val_shards = _discover_tinystories_shards()
    num_train = min(num_shards, len(train_shards))
    shards_to_download = train_shards[:num_train] + val_shards

    # Check what's already downloaded
    existing = 0
    for shard in shards_to_download:
        local_name = shard.replace("/", "-")
        local_path = os.path.join(DATA_DIR, local_name)
        if os.path.exists(local_path):
            existing += 1

    if existing == len(shards_to_download):
        print(f"  TinyStories: all {len(shards_to_download)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(shards_to_download) - existing
    print(f"  TinyStories: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(_download_one_tinystories, shards_to_download)

    ok = sum(1 for r in results if r)
    print(f"  TinyStories: {ok}/{len(shards_to_download)} shards ready at {DATA_DIR}")

    # Create a copy for the val shard with our standard name
    val_local = val_shards[0].replace("/", "-")
    val_src = os.path.join(DATA_DIR, val_local)
    val_dst = os.path.join(DATA_DIR, VAL_FILENAME)
    if os.path.exists(val_src) and not os.path.exists(val_dst):
        import shutil
        shutil.copy2(val_src, val_dst)
        print(f"    Created validation shard: {VAL_FILENAME}")


# ---------------------------------------------------------------------------
# ClimbMix download (same as upstream)
# ---------------------------------------------------------------------------

def download_single_climbmix_shard(index):
    """Download one ClimbMix parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{CLIMBMIX_BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"    Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"    Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_climbmix(num_shards, download_workers=8):
    """Download ClimbMix training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, CLIMBMIX_MAX_SHARD)
    ids = list(range(num_train))
    if CLIMBMIX_VAL_SHARD not in ids:
        ids.append(CLIMBMIX_VAL_SHARD)

    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"  ClimbMix: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"  ClimbMix: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_climbmix_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"  ClimbMix: {ok}/{len(ids)} shards ready at {DATA_DIR}")


# ---------------------------------------------------------------------------
# Unified download dispatcher
# ---------------------------------------------------------------------------

def download_data(dataset, num_shards, download_workers=8):
    """Download data shards for the selected dataset."""
    print(f"[1/2] Downloading {dataset} dataset...")
    if dataset == "tinystories":
        download_tinystories(num_shards, download_workers=min(download_workers, 4))
    elif dataset == "climbmix":
        download_climbmix(num_shards, download_workers=download_workers)
    else:
        print(f"  ERROR: Unknown dataset '{dataset}'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Tokenizer training (adapted from upstream, uses configurable VOCAB_SIZE)
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    parquet_paths = [p for p in list_parquet_files() if os.path.abspath(p) != os.path.abspath(val_path)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"  Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("  Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print(f"  Tokenizer: training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer (pickle required by tiktoken Encoding objects)
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"  Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("  Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"  Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"  Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train_consumer.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        if tokenizer_dir is None:
            tokenizer_dir = TOKENIZER_DIR
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare_consumer.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if os.path.abspath(p) != os.path.abspath(val_path)]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).

    Identical logic to upstream prepare.py.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric, same as upstream)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Tier config loading
# ---------------------------------------------------------------------------

def load_tier_config(config_path):
    """Load tier configuration from JSON file. Returns dict or None."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data and tokenizer for autoresearch (consumer edition)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["climbmix", "tinystories"],
        default="tinystories",
        help="Dataset to download (default: tinystories)"
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Number of training shards to download. Default: 8 for tinystories, 64 for climbmix. Use -1 for all."
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Maximum sequence length / context window (overrides tier config default)"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=None,
        help="BPE vocabulary size (overrides tier config default)"
    )
    parser.add_argument(
        "--eval-tokens", type=int, default=None,
        help="Number of tokens for validation evaluation (default: 524288)"
    )
    parser.add_argument(
        "--tier-config", type=str, default=None,
        help="Path to tier JSON config file (auto-sets defaults for max-seq-len, vocab-size, etc.)"
    )
    parser.add_argument(
        "--download-workers", type=int, default=8,
        help="Number of parallel download workers (default: 8)"
    )
    args = parser.parse_args()

    # --- Load tier config if provided ---
    tier = load_tier_config(args.tier_config)
    if tier:
        print(f"Loaded tier config: {args.tier_config}")
        print(f"  Tier: {tier.get('name', 'unknown')}")

    # --- Resolve num_shards ---
    if args.num_shards is not None:
        num_shards = args.num_shards
    elif tier and "num_shards" in tier:
        num_shards = tier["num_shards"]
    else:
        num_shards = 8 if args.dataset == "tinystories" else 64

    if num_shards == -1:
        if args.dataset == "climbmix":
            num_shards = CLIMBMIX_MAX_SHARD
        else:
            num_shards = 4  # TinyStories has 4 train shards

    # --- Resolve MAX_SEQ_LEN ---
    if args.max_seq_len is not None:
        MAX_SEQ_LEN = args.max_seq_len
    elif tier and "max_seq_len" in tier:
        MAX_SEQ_LEN = tier["max_seq_len"]
    # else: keep default (1024)

    # --- Resolve VOCAB_SIZE ---
    if args.vocab_size is not None:
        VOCAB_SIZE = args.vocab_size
    elif tier and "vocab_size" in tier:
        VOCAB_SIZE = tier["vocab_size"]
    # else: keep default (8192)

    # --- Resolve EVAL_TOKENS ---
    if args.eval_tokens is not None:
        EVAL_TOKENS = args.eval_tokens
    elif tier and "eval_tokens" in tier:
        EVAL_TOKENS = tier["eval_tokens"]
    # else: keep default (524288)

    # --- Configure dataset paths ---
    configure_dataset(args.dataset)

    # --- Print configuration summary ---
    print()
    print("=" * 60)
    print("  autoresearch consumer — data preparation")
    print("=" * 60)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Num shards:    {num_shards}")
    print(f"  Max seq len:   {MAX_SEQ_LEN}")
    print(f"  Vocab size:    {VOCAB_SIZE}")
    print(f"  Eval tokens:   {EVAL_TOKENS:,}")
    print(f"  Cache dir:     {CACHE_DIR}")
    print(f"  Data dir:      {DATA_DIR}")
    print(f"  Tokenizer dir: {TOKENIZER_DIR}")
    print("=" * 60)
    print()

    # Step 1: Download data
    download_data(args.dataset, num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    print("[2/2] Training tokenizer...")
    train_tokenizer()
    print()

    print("Done! Ready to train with train_consumer.py")
