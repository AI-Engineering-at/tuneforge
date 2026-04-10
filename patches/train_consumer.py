"""
Consumer-patched autoresearch pretraining script. Single-GPU, single-file.
Based on upstream train.py with consumer GPU support.

Changes from upstream:
1. Flash Attention 3 fallback to PyTorch SDPA for non-Hopper/non-Blackwell GPUs
2. Tier config loading from JSON (env: AUTORESEARCH_TIER_CONFIG)
3. Dynamic GPU peak FLOPS detection (RTX 2060 through 4090)
4. VRAM check before training
5. BF16/TF32 auto-detection based on GPU compute capability
6. Configurable model architecture via tier config

Usage:
    AUTORESEARCH_TIER_CONFIG=configs/tier1-8gb.json uv run train_consumer.py
    uv run train_consumer.py  # uses defaults
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Flash Attention 3 — graceful fallback to PyTorch SDPA
# ---------------------------------------------------------------------------

USE_FA3 = False
fa3 = None

try:
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    USE_FA3 = True
    print(f"Flash Attention 3: loaded from {repo}")
except Exception as e:
    USE_FA3 = False
    print(f"Flash Attention 3: not available ({e})")
    print("  Falling back to PyTorch scaled_dot_product_attention (works on all NVIDIA GPUs)")

# ---------------------------------------------------------------------------
# Tier config loading
# ---------------------------------------------------------------------------

TIER_CONFIG_PATH = os.environ.get("AUTORESEARCH_TIER_CONFIG", "configs/tier1-8gb.json")
tier = None
if os.path.exists(TIER_CONFIG_PATH):
    with open(TIER_CONFIG_PATH) as f:
        tier = json.load(f)
    print(f"Tier config: loaded {TIER_CONFIG_PATH} ({tier.get('name', 'unknown')})")
else:
    print(f"Tier config: {TIER_CONFIG_PATH} not found, using defaults")

# ---------------------------------------------------------------------------
# Import from consumer prepare (uses separate cache dir)
# ---------------------------------------------------------------------------

# entrypoint.sh copies prepare_consumer.py → prepare.py
try:
    from prepare_consumer import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb, configure_dataset
except ModuleNotFoundError:
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb, configure_dataset

# Configure dataset from tier config
if tier:
    configure_dataset(tier.get("dataset", "tinystories"), tier.get("max_shards"))

# ---------------------------------------------------------------------------
# GPU capability detection
# ---------------------------------------------------------------------------

cap = torch.cuda.get_device_capability()
USE_BF16 = cap[0] >= 8   # Ampere (SM 80) and newer support BF16
USE_TF32 = cap[0] >= 8   # Ampere (SM 80) and newer support TF32
# Turing GPUs (SM 7.x): Muon optimizer hardcodes bfloat16 for polar express
# orthogonalization (line ~484). FP16 training + BF16 optimizer = precision loss
# and NaN gradients. Use FP32 with autocast disabled on pre-Ampere GPUs.
USE_FP32_FALLBACK = not USE_BF16  # Turing and older: skip FP16, use FP32

if USE_BF16:
    print(f"GPU compute capability: {cap[0]}.{cap[1]} — using BF16 + TF32")
elif USE_FP32_FALLBACK:
    print(f"GPU compute capability: {cap[0]}.{cap[1]} — using FP32 (Turing: Muon needs BF16, FP16 unstable)")

# ---------------------------------------------------------------------------
# Approximate peak BF16 FLOPS for common consumer GPUs
# Used for MFU (Model FLOPS Utilization) calculation only — not critical
# ---------------------------------------------------------------------------

GPU_PEAK_FLOPS = {
    # Turing (20-series) — FP16 tensor core FLOPS (no native BF16)
    "RTX 2060": 13e12,
    "RTX 2060 SUPER": 14e12,
    "RTX 2070": 15e12,
    "RTX 2070 SUPER": 18e12,
    "RTX 2080": 20e12,
    "RTX 2080 SUPER": 22e12,
    "RTX 2080 Ti": 27e12,
    "TITAN RTX": 32e12,
    # Ampere (30-series) — BF16 tensor core FLOPS
    "RTX 3060": 25e12,
    "RTX 3060 Ti": 32e12,
    "RTX 3070": 40e12,
    "RTX 3070 Ti": 43e12,
    "RTX 3080": 60e12,
    "RTX 3080 Ti": 68e12,
    "RTX 3090": 71e12,
    "RTX 3090 Ti": 80e12,
    # Ada Lovelace (40-series) — BF16 tensor core FLOPS
    "RTX 4060": 30e12,
    "RTX 4060 Ti": 44e12,
    "RTX 4070": 45e12,
    "RTX 4070 SUPER": 56e12,
    "RTX 4070 Ti": 56e12,
    "RTX 4070 Ti SUPER": 66e12,
    "RTX 4080": 97e12,
    "RTX 4080 SUPER": 104e12,
    "RTX 4090": 165e12,
    # Laptop variants (approximate — lower clocks)
    "RTX 4070 Laptop GPU": 35e12,
    "RTX 4080 Laptop GPU": 58e12,
    "RTX 4090 Laptop GPU": 86e12,
    "RTX 3070 Laptop GPU": 28e12,
    "RTX 3080 Laptop GPU": 42e12,
    # Data center / workstation (for reference)
    "A100": 312e12,
    "A100-SXM4-80GB": 312e12,
    "H100": 989.5e12,
    "H100-SXM5-80GB": 989.5e12,
    "L40S": 366e12,
}


def get_gpu_peak_flops():
    """Detect GPU and return approximate peak BF16/FP16 FLOPS."""
    gpu_name = torch.cuda.get_device_name()
    # Try exact match first
    for key, flops in GPU_PEAK_FLOPS.items():
        if key in gpu_name:
            return flops, key
    # Fallback: estimate from compute capability
    if cap[0] >= 9:
        return 200e12, f"unknown Hopper+ ({gpu_name})"
    elif cap[0] >= 8 and cap[1] >= 9:
        return 100e12, f"unknown Ada ({gpu_name})"
    elif cap[0] >= 8:
        return 50e12, f"unknown Ampere ({gpu_name})"
    elif cap[0] >= 7:
        return 20e12, f"unknown Turing/Volta ({gpu_name})"
    else:
        return 10e12, f"unknown older GPU ({gpu_name})"

# ---------------------------------------------------------------------------
# GPT Model (identical architecture to upstream)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if USE_FA3:
            # Flash Attention 3 path (Hopper/Blackwell GPUs with FA3 support)
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.contiguous().view(B, T, -1)
        else:
            # Standard PyTorch SDPA fallback (works on all NVIDIA GPUs with CUDA)
            # SDPA expects (B, n_heads, T, head_dim) layout
            # GQA: expand kv heads to match query heads
            n_rep = self.n_head // self.n_kv_head
            if n_rep > 1:
                k = k.unsqueeze(3).expand(B, T, self.n_kv_head, n_rep, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
                v = v.unsqueeze(3).expand(B, T, self.n_kv_head, n_rep, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
            # Transpose to (B, n_heads, T, head_dim) for SDPA
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            # Note: SDPA doesn't support sliding window natively,
            # so for non-FA3 path, all attention is full causal (window_size ignored).
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to training dtype (skip on Turing — stay FP32)
        if USE_BF16:
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)
        # FP32_FALLBACK: keep embeddings in FP32 for Turing stability

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        if USE_BF16:
            cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        # FP32_FALLBACK: keep rotary embeddings in FP32
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR proportional to 1/sqrt(dmodel) (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only — identical to upstream)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Without torch.compile, 0-D CPU tensors (float) don't auto-cast to bf16.
    # Extract Python scalars with .item() for compatibility with any dtype.
    step = step_t.item() if hasattr(step_t, 'item') else float(step_t)
    lr = lr_t.item() if hasattr(lr_t, 'item') else float(lr_t)
    beta1 = beta1_t.item() if hasattr(beta1_t, 'item') else float(beta1_t)
    beta2 = beta2_t.item() if hasattr(beta2_t, 'item') else float(beta2_t)
    eps = eps_t.item() if hasattr(eps_t, 'item') else float(eps_t)
    wd = wd_t.item() if hasattr(wd_t, 'item') else float(wd_t)
    grad = grad.to(p.dtype)
    p.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1_corr = 1 - beta1 ** step
    bias2_corr = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2_corr).sqrt() + eps
    step_size = lr / bias1_corr
    p.add_(exp_avg / denom, alpha=-step_size)

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum_val = momentum_t.item() if hasattr(momentum_t, 'item') else float(momentum_t)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum_val)
    g = stacked_grads.lerp_(momentum_buffer, momentum_val)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2_val = beta2_t.item() if hasattr(beta2_t, 'item') else float(beta2_t)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2_val)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr_val = lr_t.item() if hasattr(lr_t, 'item') else float(lr_t)
    wd_val = wd_t.item() if hasattr(wd_t, 'item') else float(wd_t)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr_val * g + lr_val * wd_val * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters — defaults, overridden by tier config
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Override hyperparameters from tier config
# ---------------------------------------------------------------------------

if tier:
    DEPTH = tier.get("depth", DEPTH)
    DEVICE_BATCH_SIZE = tier.get("device_batch_size", DEVICE_BATCH_SIZE)
    TOTAL_BATCH_SIZE = tier.get("total_batch_size", TOTAL_BATCH_SIZE)
    # Override MAX_SEQ_LEN from tier config (imported value is module-level default)
    if "max_seq_len" in tier:
        MAX_SEQ_LEN = tier["max_seq_len"]
    ASPECT_RATIO = tier.get("aspect_ratio", ASPECT_RATIO)
    HEAD_DIM = tier.get("head_dim", HEAD_DIM)
    WINDOW_PATTERN = tier.get("window_pattern", WINDOW_PATTERN)
    EMBEDDING_LR = tier.get("embedding_lr", EMBEDDING_LR)
    UNEMBEDDING_LR = tier.get("unembedding_lr", UNEMBEDDING_LR)
    MATRIX_LR = tier.get("matrix_lr", MATRIX_LR)
    SCALAR_LR = tier.get("scalar_lr", SCALAR_LR)
    WEIGHT_DECAY = tier.get("weight_decay", WEIGHT_DECAY)
    WARMUP_RATIO = tier.get("warmup_ratio", WARMUP_RATIO)
    WARMDOWN_RATIO = tier.get("warmdown_ratio", WARMDOWN_RATIO)
    FINAL_LR_FRAC = tier.get("final_lr_frac", FINAL_LR_FRAC)

# ---------------------------------------------------------------------------
# Setup: VRAM check, tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# TF32 matmul precision (Ampere+ only)
if USE_TF32:
    torch.set_float32_matmul_precision("high")

device = torch.device("cuda")
if USE_BF16:
    train_dtype = torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=train_dtype)
elif USE_FP32_FALLBACK:
    train_dtype = torch.float32
    # No autocast on Turing — FP32 everywhere for stability with Muon optimizer
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)

# --- VRAM check ---
free_mem, total_mem = torch.cuda.mem_get_info()
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}, VRAM: {total_mem // 1024 // 1024}MB ({free_mem // 1024 // 1024}MB free)")

# --- Dynamic peak FLOPS ---
gpu_peak_flops, gpu_match = get_gpu_peak_flops()
print(f"GPU peak FLOPS (approx): {gpu_peak_flops:.1e} (matched: {gpu_match})")

# --- Tokenizer ---
tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# --- Model config ---
def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    # Allow tier config to override n_head, n_kv_head, n_embd directly
    if tier:
        model_dim = tier.get("n_embd", model_dim)
        num_heads = tier.get("n_head", num_heads)
        n_kv_head = tier.get("n_kv_head", num_heads)
    else:
        n_kv_head = num_heads
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=n_kv_head, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

# --- Estimate VRAM requirement before building model ---
estimated_params = (config.n_layer * config.n_embd * config.n_embd * 8  # rough: 4 linear layers per block
                    + config.vocab_size * config.n_embd * 2)  # embeddings
estimated_vram_mb = estimated_params * 2 / 1024 / 1024  # bf16 = 2 bytes
# Training needs ~4x model size (model + gradients + optimizer states)
estimated_training_vram_mb = estimated_vram_mb * 4
if estimated_training_vram_mb > free_mem / 1024 / 1024:
    print(f"WARNING: Estimated training VRAM ({estimated_training_vram_mb:.0f}MB) may exceed free VRAM ({free_mem // 1024 // 1024}MB)")
    print("  Consider reducing DEPTH, DEVICE_BATCH_SIZE, or MAX_SEQ_LEN")

# --- Build model ---
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, (
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by "
    f"DEVICE_BATCH_SIZE * MAX_SEQ_LEN ({tokens_per_fwdbwd})"
)
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile can OOM on consumer GPUs (subprocess workers get killed)
# Disable via env var or tier config
USE_COMPILE = os.environ.get("AUTORESEARCH_COMPILE", "").lower() not in ("0", "false", "no")
if tier and tier.get("disable_compile", False):
    USE_COMPILE = False
if USE_COMPILE:
    try:
        model = torch.compile(model, dynamic=False)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: failed ({e}), continuing without compilation")
else:
    print("torch.compile: disabled (consumer GPU mode)")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Training dtype: {train_dtype}")
print(f"Flash Attention 3: {'enabled' if USE_FA3 else 'disabled (using PyTorch SDPA)'}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop (identical logic to upstream, with dynamic FLOPS)
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging (uses dynamic GPU peak FLOPS instead of hardcoded H100 value)
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / gpu_peak_flops
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final validation
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary (uses dynamic GPU peak FLOPS)
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / gpu_peak_flops if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"gpu:              {gpu_name}")
print(f"dtype:            {train_dtype}")
print(f"flash_attn_3:     {USE_FA3}")
