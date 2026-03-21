# Hyperparameter Optimization Program
> Based on the autoresearch framework by Andrej Karpathy: https://github.com/karpathy/autoresearch
> Modified variant: focused exclusively on hyperparameter tuning and optimization settings.

This program systematically tunes hyperparameters while keeping the model architecture fixed. The architecture stays constant while learning rate, batch size, optimizer, and training dynamics are the variables.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5-hparams`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Record baseline hyperparameters**: Before making any changes, document the current values of ALL hyperparameters in train.py (learning rate, batch size, weight decay, warmup steps, betas, grad clip, etc.).
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation Philosophy

Architecture stays FIXED. Only change hyperparameters, optimizer settings, and training dynamics. The model structure (depth, width, attention type, etc.) must remain unchanged throughout.

### Experiment Categories

Work through these categories in order. Each category has a systematic grid to explore.

#### 1. Learning Rate
The single most impactful hyperparameter. Explore systematically.

- **Baseline LR**: Record current value.
- **Coarse sweep**: Try 2x, 4x, 0.5x, 0.25x of baseline LR.
- **Fine sweep**: Once you find the best coarse value, try values +/- 20% around it.
- **Peak LR with warmup**: Try different peak LR values with the same warmup schedule.

#### 2. Learning Rate Schedule
- **Cosine decay**: Standard approach. Decay from peak LR to some fraction (0.1x, 0.01x, 0x).
- **Linear decay**: Simple linear decrease. Often competitive.
- **Cosine with restarts**: Reset the schedule partway through training. Try 1 and 2 restarts.
- **Warmup duration**: Try 0%, 1%, 2%, 5%, 10% of total steps as warmup.
- **Cooldown**: Add a final cooldown phase (last 10% of training at very low LR).
- **Constant LR**: No schedule at all. How much does scheduling actually help?
- **WSD** (Warmup-Stable-Decay): Warmup, then constant, then decay. Try different stable/decay ratios.

#### 3. Batch Size
- **Baseline batch size**: Record current total_batch_size and device_batch_size.
- **Double batch size**: 2x total_batch_size (adjust gradient accumulation).
- **Half batch size**: 0.5x total_batch_size.
- **Large batch + linear LR scaling**: When doubling batch, also scale LR proportionally.
- **Large batch + sqrt LR scaling**: Scale LR by sqrt(batch_ratio) instead of linear.
- **Gradient accumulation steps**: Try different accumulation counts that achieve the same effective batch size.

#### 4. Optimizer Settings
- **Muon vs AdamW**: If the code uses one, try the other. This is a big lever.
- **AdamW betas**: Try (0.9, 0.95), (0.9, 0.99), (0.9, 0.999), (0.95, 0.999).
- **Weight decay**: Try 0, 0.01, 0.05, 0.1, 0.2. Weight decay interacts with LR.
- **Epsilon**: Usually 1e-8. Try 1e-6 for potential stability gains.
- **Decoupled weight decay**: If using coupled, try decoupled (and vice versa).
- **Muon momentum**: If using Muon, try different momentum values.
- **Separate LR groups**: Different LR for embeddings vs transformer layers vs head.

#### 5. Gradient Clipping
- **Max norm clipping**: Try 1.0, 0.5, 0.1, 2.0. Record which prevents spikes.
- **No clipping**: Remove gradient clipping entirely. Sometimes the model trains fine without it.
- **Adaptive clipping**: Clip based on the running average of gradient norms.
- **Value clipping**: Clip individual gradient values instead of the norm.

#### 6. Model Scaling (Fixed Architecture)
Keep the architecture type the same but change the size allocation.

- **Wider model**: Increase n_embd, decrease depth. Same parameter count.
- **Deeper model**: Increase depth, decrease n_embd. Same parameter count.
- **Wider FFN**: Increase FFN hidden dimension ratio (e.g., 4x to 6x) while reducing another dimension.
- **More heads**: Increase n_head while keeping n_embd constant (smaller head dim).
- **Fewer heads**: Decrease n_head (larger head dim). Sometimes better at small scale.

#### 7. Training Dynamics
- **Label smoothing**: Try 0.05, 0.1, 0.2 label smoothing on the cross-entropy loss.
- **Dropout**: Try 0.0, 0.05, 0.1 dropout. At small scale, dropout often hurts.
- **Stochastic depth**: Randomly skip layers during training with some probability.
- **Gradient noise**: Add small Gaussian noise to gradients. Can help escape local minima.
- **Loss scaling**: If using mixed precision, experiment with different loss scale values.
- **Z-loss**: Add a small penalty on the log-partition function to stabilize softmax.

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
5. short text description, always starting with the category (e.g., "lr: increase to 0.002", "optim: switch to AdamW")

## The experiment loop

LOOP FOREVER:

1. Review results.tsv. Track which hyperparameter ranges have been tried and which gave the best results.
2. Pick the next experiment from the category list. Start with learning rate (category 1) since it has the highest impact.
3. Make exactly ONE hyperparameter change in `train.py`.
4. git commit with a descriptive message (e.g., "hparam: LR 0.001 -> 0.002").
5. Run: `uv run train.py > run.log 2>&1`
6. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If crashed: `tail -n 50 run.log`. Fix trivial issues (NaN from too high LR, OOM from large batch).
8. Record in results.tsv.
9. Decision:
   - val_bpb improved: **keep**, advance branch
   - val_bpb equal or worse: **discard**, git reset
10. After finding the best value in a category, move to the next category. Each subsequent category builds on the best settings found so far.

**Interaction effects**: After completing all categories individually, run a "re-sweep" phase:
- Re-test the best LR with the new batch size and optimizer settings.
- Re-test the best schedule with the new LR.
- Hyperparameters interact, so the optimal LR from step 1 may not be optimal after changing the optimizer in step 4.

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Hyperparameter experiments commonly crash from:
- NaN/Inf from too high LR: reduce LR and retry.
- OOM from large batch size: reduce batch size.
- Numerical instability: try adding epsilon, clipping, or reducing LR.

**NEVER STOP**: Continue indefinitely until manually interrupted. After exhausting all categories and re-sweeps, try exotic combinations: cyclic learning rates, per-layer learning rates, progressive resizing, curriculum learning (if possible within constraints), or other training tricks from recent papers.
