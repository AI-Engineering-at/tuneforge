# Conservative Research Program
> Based on the autoresearch framework by Andrej Karpathy: https://github.com/karpathy/autoresearch
> Modified variant: conservative approach with smaller steps and aggressive reverting.

This program takes a methodical, low-risk approach to improving val_bpb. One change at a time, always with a clear baseline, and aggressive reverting when improvements are marginal.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5-conservative`). The branch `autoresearch/<tag>` must not already exist.
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

This is the conservative variant. The core principles are:

1. **One change at a time.** Never combine two untested ideas in a single experiment. If you want to try a new learning rate AND a new activation function, do them in separate runs.
2. **Always establish a baseline first.** The very first run is always the unmodified train.py.
3. **Hyperparameters before architecture.** Exhaust hyperparameter tuning opportunities before making structural changes to the model. The order of exploration should be:
   - Learning rate (try 0.5x, 0.75x, 1.5x, 2x of current — 0.5x was best in testing)
   - Gradient clipping thresholds
   - Only then: architecture changes

   **Tested and proven ineffective for 5-minute runs (skip these):**
   - Batch size changes (both halving and doubling hurt significantly)
   - Weight decay changes (0.2 is optimal, both 0.1 and 0.4 are worse)
   - Warmup (any warmup wastes precious training time in short runs)
   - Depth changes (depth 3 is optimal for 5-min experiments on Tier 1)
4. **Aggressive reverting.** Only keep a change if the improvement is > 0.005 val_bpb. Marginal gains (< 0.005) that add any complexity are discarded. The threshold is strict because small gains often vanish with further changes.
5. **Simplification passes.** After every 5 experiments, do a "simplification pass": look at the current code and try removing things. If removing something gives equal or better results, that is a strong keep.
6. **No speculative complexity.** Do not add mechanisms "just in case" they help. Every line of code must earn its place through measured improvement.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness.
- Change more than one thing per experiment.

**The goal is simple: get the lowest val_bpb** while keeping the code as clean and simple as possible.

**VRAM** is a hard constraint in this program. Do not increase peak VRAM by more than 10% over baseline. If an experiment exceeds this, revert immediately.

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
5. short text description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Review the current results.tsv to understand what has been tried and what worked.
2. Pick the SINGLE most promising next change, following the priority order:
   - Hyperparameter tuning (LR, batch size, weight decay, warmup, grad clip)
   - Optimizer changes (Muon vs AdamW, beta values)
   - Model scaling (wider vs deeper, within VRAM budget)
   - Architecture changes (only after hyperparams are exhausted)
3. Make exactly ONE change to `train.py`.
4. git commit with a clear description of the single change.
5. Run the experiment: `uv run train.py > run.log 2>&1`
6. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If crashed: `tail -n 50 run.log`, attempt one fix. If still broken, give up on this idea.
8. Record in results.tsv.
9. Decision:
   - If val_bpb improved by > 0.005: **keep** (advance branch)
   - If val_bpb improved by 0.001-0.005 AND no added complexity: **keep**
   - If val_bpb improved by < 0.001 OR added complexity with small gain: **discard** (git reset)
   - If val_bpb is worse: **discard** (git reset)
   - If VRAM increased by > 10% over baseline: **discard** regardless of val_bpb
10. Every 5 experiments: run a simplification pass (try removing things).

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Fix trivial bugs (typos, imports). If the idea is broken, skip it.

**NEVER STOP**: Continue indefinitely until manually interrupted. If you run out of hyperparameter ideas, move to architecture. If you run out of architecture ideas, try combinations of previously successful changes, or revisit near-misses with slight variations.
