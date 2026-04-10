#!/usr/bin/env python3
"""
visualize.py — Dashboard charts + terminal summary for autoresearch experiments.

Reads results.tsv (the experiment log from the autoresearch loop) and generates
PNG charts plus a concise terminal summary.

Based on Andrej Karpathy's autoresearch framework.
See: https://github.com/karpathy/autoresearch
Original concept and training loop by Andrej Karpathy (MIT License).

Usage:
    python visualize.py                          # uses ./results or /app/results
    python visualize.py --results-dir /path/to   # custom results dir
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/Docker use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHART_DIR_NAME = "charts"
TSV_FILENAME = "results.tsv"

# Expected TSV columns (tab-separated)
# commit | val_bpb | memory_gb | status | description
COL_COMMIT = "commit"
COL_VAL_BPB = "val_bpb"
COL_MEMORY = "memory_gb"
COL_STATUS = "status"
COL_DESC = "description"

COLORS = {
    "kept": "#2ecc71",
    "discarded": "#e74c3c",
    "crashed": "#95a5a6",
    "line": "#3498db",
    "best": "#f39c12",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_results_dir(override: str | None = None) -> Path:
    """Resolve the results directory. Priority: CLI arg > ./results > /app/results."""
    if override:
        return Path(override)
    local = Path("./results")
    if local.is_dir():
        return local
    docker = Path("/app/results")
    if docker.is_dir():
        return docker
    return local  # fallback — will error later if missing


def load_results(results_dir: Path) -> list[dict]:
    """
    Read results.tsv and return a list of experiment dicts.

    Handles:
    - Missing file (returns empty list)
    - Partial/malformed rows (skips with warning)
    - Numeric parsing for val_bpb and memory_gb
    """
    tsv_path = results_dir / TSV_FILENAME
    if not tsv_path.exists():
        print(f"[WARN] {tsv_path} not found. Nothing to visualize.")
        return []

    experiments = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Validate that we have the expected columns
        if reader.fieldnames is None:
            print(f"[WARN] {tsv_path} appears empty.")
            return []

        for i, row in enumerate(reader, start=2):  # line 2+ (1 is header)
            try:
                exp = {
                    "num": i - 1,  # 1-indexed experiment number
                    "commit": row.get(COL_COMMIT, "").strip(),
                    "val_bpb": _parse_float(row.get(COL_VAL_BPB, "")),
                    "memory_gb": _parse_float(row.get(COL_MEMORY, "")),
                    "status": row.get(COL_STATUS, "").strip().lower(),
                    "description": row.get(COL_DESC, "").strip(),
                }
                experiments.append(exp)
            except Exception as e:
                print(f"[WARN] Skipping line {i}: {e}")

    return experiments


def _parse_float(value: str) -> float | None:
    """Parse a float, returning None for empty/invalid values."""
    value = value.strip() if value else ""
    if not value or value == "N/A" or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def make_charts(experiments: list[dict], chart_dir: Path) -> None:
    """Generate all PNG charts and save to chart_dir."""
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_val_bpb(experiments, chart_dir)
    chart_memory(experiments, chart_dir)
    chart_keep_ratio(experiments, chart_dir)
    chart_best_progression(experiments, chart_dir)

    print(f"[OK] Charts saved to {chart_dir}/")


def chart_val_bpb(experiments: list[dict], chart_dir: Path) -> None:
    """Line chart: val_bpb over experiments, color-coded by kept/discarded."""
    valid = [e for e in experiments if e["val_bpb"] is not None]
    if not valid:
        print("[WARN] No valid val_bpb data — skipping val_bpb chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot all points as a line
    nums = [e["num"] for e in valid]
    bpbs = [e["val_bpb"] for e in valid]
    ax.plot(nums, bpbs, color=COLORS["line"], alpha=0.3, linewidth=1, zorder=1)

    # Scatter: color by status
    for e in valid:
        color = COLORS.get(e["status"], COLORS["crashed"])
        ax.scatter(e["num"], e["val_bpb"], color=color, s=40, zorder=2,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("val_bpb (lower is better)")
    ax.set_title("Validation BPB Across Experiments")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["kept"],
               markersize=8, label="Kept"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["discarded"],
               markersize=8, label="Discarded"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(chart_dir / "val_bpb_over_experiments.png", dpi=150)
    plt.close(fig)


def chart_memory(experiments: list[dict], chart_dir: Path) -> None:
    """Bar chart: memory usage (GB) per experiment."""
    valid = [e for e in experiments if e["memory_gb"] is not None]
    if not valid:
        print("[WARN] No memory data — skipping memory chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    nums = [e["num"] for e in valid]
    mem = [e["memory_gb"] for e in valid]
    colors = [COLORS.get(e["status"], COLORS["crashed"]) for e in valid]

    ax.bar(nums, mem, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("GPU Memory Usage per Experiment")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(chart_dir / "memory_usage.png", dpi=150)
    plt.close(fig)


def chart_keep_ratio(experiments: list[dict], chart_dir: Path) -> None:
    """Pie chart: kept vs discarded vs crashed."""
    if not experiments:
        return

    counts = {}
    for e in experiments:
        status = e["status"] if e["status"] in ("kept", "discarded") else "crashed/other"
        counts[status] = counts.get(status, 0) + 1

    if not counts:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    labels = list(counts.keys())
    sizes = list(counts.values())
    color_map = {
        "kept": COLORS["kept"],
        "discarded": COLORS["discarded"],
        "crashed/other": COLORS["crashed"],
    }
    colors = [color_map.get(line, "#bdc3c7") for line in labels]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Experiment Outcomes")

    fig.tight_layout()
    fig.savefig(chart_dir / "keep_discard_ratio.png", dpi=150)
    plt.close(fig)


def chart_best_progression(experiments: list[dict], chart_dir: Path) -> None:
    """Line chart: best val_bpb over time (only kept experiments, cumulative min)."""
    kept = [e for e in experiments if e["status"] == "kept" and e["val_bpb"] is not None]
    if not kept:
        print("[WARN] No kept experiments — skipping best progression chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Build cumulative best
    nums = []
    best_vals = []
    current_best = float("inf")
    for e in kept:
        if e["val_bpb"] < current_best:
            current_best = e["val_bpb"]
        nums.append(e["num"])
        best_vals.append(current_best)

    ax.plot(nums, best_vals, color=COLORS["best"], linewidth=2, marker="o",
            markersize=6, markerfacecolor=COLORS["best"], markeredgecolor="white")
    ax.fill_between(nums, best_vals, alpha=0.1, color=COLORS["best"])

    ax.set_xlabel("Experiment # (kept only)")
    ax.set_ylabel("Best val_bpb (cumulative min)")
    ax.set_title("Best val_bpb Progression")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(chart_dir / "best_progression.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_summary(experiments: list[dict]) -> None:
    """Print a concise terminal summary of the experiment run."""
    if not experiments:
        print("\n[SUMMARY] No experiments found.\n")
        return

    total = len(experiments)
    kept = [e for e in experiments if e["status"] == "kept"]
    discarded = [e for e in experiments if e["status"] == "discarded"]
    crashed = [e for e in experiments if e["status"] not in ("kept", "discarded")]

    # Best val_bpb overall
    valid_bpb = [e for e in experiments if e["val_bpb"] is not None and e["val_bpb"] > 0]
    best_bpb = min(valid_bpb, key=lambda e: e["val_bpb"]) if valid_bpb else None

    # Improvement calculation: compare first kept to last best kept
    kept_with_bpb = [e for e in kept if e["val_bpb"] is not None and e["val_bpb"] > 0]
    improvement = None
    avg_improvement = None
    if len(kept_with_bpb) >= 2:
        first_bpb = kept_with_bpb[0]["val_bpb"]
        last_bpb = min(e["val_bpb"] for e in kept_with_bpb)
        improvement = first_bpb - last_bpb
        avg_improvement = improvement / (len(kept_with_bpb) - 1)

    # Print
    print("\n" + "=" * 60)
    print("  AUTORESEARCH — EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Total experiments:   {total}")
    print(f"  Kept:                {len(kept)} ({_pct(len(kept), total)})")
    print(f"  Discarded:           {len(discarded)} ({_pct(len(discarded), total)})")
    if crashed:
        print(f"  Crashed/Other:       {len(crashed)} ({_pct(len(crashed), total)})")
    print("-" * 60)

    if best_bpb:
        print(f"  Best val_bpb:        {best_bpb['val_bpb']:.6f}")
        print(f"    -> Experiment #{best_bpb['num']}: {best_bpb['description'][:60]}")

    if improvement is not None:
        print(f"  Total improvement:   {improvement:.6f} (from {first_bpb:.6f})")
        print(f"  Avg per kept step:   {avg_improvement:.6f}")

    # Memory stats
    mem_vals = [e["memory_gb"] for e in experiments if e["memory_gb"] is not None]
    if mem_vals:
        print(f"  Memory range:        {min(mem_vals):.1f} — {max(mem_vals):.1f} GB")

    print("=" * 60 + "\n")


def _pct(part: int, total: int) -> str:
    """Format a percentage string."""
    if total == 0:
        return "0%"
    return f"{100 * part / total:.0f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize autoresearch experiment results."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to results directory containing results.tsv (default: ./results or /app/results)"
    )
    args = parser.parse_args()

    results_dir = find_results_dir(args.results_dir)
    print(f"[INFO] Results dir: {results_dir.resolve()}")

    experiments = load_results(results_dir)
    if not experiments:
        sys.exit(0)

    print(f"[INFO] Loaded {len(experiments)} experiments.")

    # Generate charts
    chart_dir = results_dir / CHART_DIR_NAME
    make_charts(experiments, chart_dir)

    # Terminal summary
    print_summary(experiments)


if __name__ == "__main__":
    main()
