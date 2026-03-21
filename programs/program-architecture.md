# Architecture Research Program
> Based on the autoresearch framework by Andrej Karpathy: https://github.com/karpathy/autoresearch
> Modified variant: focused exclusively on architecture experiments.

This program systematically explores model architecture variations to find the best structure for a given compute and VRAM budget. Hyperparameters are held constant while architecture is the variable.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5-arch`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation Philosophy

This program focuses purely on architecture. Do NOT change hyperparameters (learning rate, batch size, optimizer settings) unless strictly necessary to accommodate a new architecture (e.g., adjusting LR for a much larger/smaller model).

### Experiment Categories

Work through these categories in order. Within each category, try the most promising variant first based on recent literature.

#### 1. Attention Mechanisms
- **MHA** (Multi-Head Attention): The standard. This is the baseline.
- **GQA** (Grouped Query Attention): Reduce n_kv_head to n_head/2 or n_head/4. Saves memory, often neutral on quality.
- **MQA** (Multi-Query Attention): n_kv_head=1. Maximum memory savings. Test if quality holds.
- **Sliding window attention**: Restrict attention to a local window (e.g., 128 or 256 tokens). Combine with one global layer.
- **Differential attention**: Two attention heads with subtracted outputs. Test if it helps focus.

#### 2. Normalization Variants
- **RMSNorm** vs **LayerNorm**: RMSNorm is simpler (no mean subtraction). Often equivalent or better.
- **Pre-norm** vs **post-norm**: Pre-norm (norm before attention/FFN) is standard in modern transformers. Try post-norm to see if it matters at this scale.
- **No normalization**: Remove all norms. Does the model still train at this scale?
- **QK-norm**: Apply normalization to query and key vectors before attention. Can stabilize training.

#### 3. Activation Functions
- **SiLU** (Swish): Current default in many modern architectures.
- **GELU**: The GPT-2/BERT standard. Compare directly.
- **ReLU**: The simplest option. Sometimes competitive at small scale.
- **SwiGLU**: Gated variant. Adds parameters but often improves quality. Adjust hidden dim to match parameter count.
- **GeGLU**: GELU-gated linear unit. Another gated variant.

#### 4. Positional Encoding
- **RoPE** (Rotary Position Embeddings): Current standard. This is likely the baseline.
- **ALiBi** (Attention with Linear Biases): No learned parameters. Good extrapolation.
- **Learned absolute embeddings**: The original transformer approach. Test at this scale.
- **NoPE** (No Position Encoding): Remove positional encoding entirely. Causal mask provides implicit position info. Does it work at this scale?
- **Fire embeddings**: Functional interpolation of rotary encodings.

#### 5. FFN Architecture
- **Standard FFN**: Two linear layers with activation in between.
- **SwiGLU FFN**: Three matrices (gate, up, down) with SiLU gating.
- **FFN ratio changes**: Try 3x, 4x, 5x, 8/3x expansion ratio. Default is usually 4x.
- **MoE** (Mixture of Experts): If feasible within VRAM, try 2 or 4 experts with top-1 routing.
- **No FFN**: Remove the FFN entirely. Attention-only transformer. How much does it lose?

#### 6. Skip Connections and Residual Structure
- **Standard residual**: x + sublayer(x). The baseline.
- **Pre-residual scaling**: Scale the residual by 1/sqrt(depth) or a learned scalar.
- **ReZero**: Initialize residual scaling to zero. Can help training dynamics.
- **Parallel attention+FFN**: Compute attention and FFN in parallel instead of sequential. Used by PaLM.
- **Deep narrow vs shallow wide**: Same parameter count, different depth/width ratio.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key metric: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

## Logging results

Log every experiment to `results.tsv` (tab-separated). The TSV has 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. peak memory in GB (peak_vram_mb / 1024, round to .1f) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description, always starting with the category (e.g., "attn: switch to GQA n_kv_head=2")

## The experiment loop

LOOP FOREVER:

1. Review results.tsv. Identify which categories have been explored and which haven't.
2. Pick the next architecture experiment from the category list above. Work through categories in order, but skip to the next category if the current one seems exhausted.
3. Implement the change in `train.py`. Keep the change focused on one architectural element.
4. git commit with a descriptive message (e.g., "arch: replace MHA with GQA n_kv_head=2").
5. Run: `uv run train.py > run.log 2>&1`
6. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If crashed: `tail -n 50 run.log`. Fix trivial issues, otherwise skip.
8. Record in results.tsv.
9. If val_bpb improved: keep. If equal or worse: revert.
10. After going through all categories once, start combining the best findings from each category into a single "best architecture" run.

**Combination phase**: After individual experiments, combine winners. For example, if GQA and SwiGLU both helped individually, try them together. If the combination helps, keep. If it regresses, try other pairings.

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Architecture changes often cause shape mismatches or CUDA errors. Fix obvious bugs (dimension mismatches, wrong tensor shapes). If the architecture is fundamentally incompatible with the training loop, skip it.

**NEVER STOP**: Continue indefinitely until manually interrupted. After exhausting the category list and combinations, try more exotic architectures: linear attention, state-space models (if implementable within constraints), mixture of depths, token merging, etc.
