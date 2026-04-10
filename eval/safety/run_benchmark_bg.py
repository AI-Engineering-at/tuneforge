"""
Background benchmark runner - runs full benchmark in background.
Use: python eval/safety/run_benchmark_bg.py
Then check: eval/safety/results/benchmark_gemma4_baseline_2026-04-10.json
"""
import subprocess
import sys
from pathlib import Path

# Ensure results dir exists
Path("eval/safety/results").mkdir(parents=True, exist_ok=True)

# Run benchmark
result = subprocess.run(
    [
        sys.executable,
        "eval/safety/benchmark.py",
        "--model", "gemma4:26b-a4b-it-q4_K_M",
        "--ollama-url", "http://10.40.10.90:11434",
        "--condition", "baseline",
        "--output", "eval/safety/results/benchmark_gemma4_baseline_2026-04-10.json"
    ],
    capture_output=True,
    text=True,
)

# Write log
log_path = Path("eval/safety/results/benchmark.log")
log_path.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nReturn code: {result.returncode}")

print(f"Benchmark finished with code {result.returncode}")
print(f"Log: {log_path}")
