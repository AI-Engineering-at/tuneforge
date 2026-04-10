"""
Quick Safety Benchmark (10+10 prompts) for environments with timeout constraints.
Full benchmark: eval/safety/benchmark.py (50+50 prompts, ~20 min)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.safety.benchmark import SafetyBenchmark, SafetyScore
from eval.safety.prompts_harmful import HARMFUL_PROMPTS
from eval.safety.prompts_benign import BENIGN_PROMPTS


class QuickSafetyBenchmark(SafetyBenchmark):
    """Quick benchmark using only first 10 prompts from each category."""
    
    def evaluate(self) -> SafetyScore:
        """Run quick safety benchmark with subset of prompts."""
        print(f"\n[QUICK BENCHMARK] {self.model_name} ({self.condition})")
        print("=" * 60)
        print("NOTE: Using 10 harmful + 10 benign prompts (full: 50+50)")
        print("=" * 60)
        
        # Use only first 10 prompts
        harmful_prompts = HARMFUL_PROMPTS[:10]
        benign_prompts = BENIGN_PROMPTS[:10]
        
        # Evaluate harmful prompts
        print(f"\nTesting {len(harmful_prompts)} harmful prompts...")
        harmful_total, harmful_refused = self.evaluate_prompts(harmful_prompts)
        harmful_rate = harmful_refused / harmful_total if harmful_total > 0 else 0.0
        
        # Evaluate benign prompts
        print(f"\nTesting {len(benign_prompts)} benign prompts...")
        benign_total, benign_refused = self.evaluate_prompts(benign_prompts)
        benign_rate = benign_refused / benign_total if benign_total > 0 else 0.0
        
        score = SafetyScore(
            harmful_refusal_rate=harmful_rate,
            benign_refusal_rate=benign_rate,
            harmful_total=harmful_total,
            harmful_refused=harmful_refused,
            benign_total=benign_total,
            benign_refused=benign_refused,
            model_name=self.model_name,
            condition=self.condition + "_quick",
        )
        
        print("\n" + "=" * 60)
        print("QUICK RESULTS (10+10 prompts):")
        print(f"  Harmful refusal rate: {harmful_rate:.2%} ({harmful_refused}/{harmful_total})")
        print(f"  Benign refusal rate:  {benign_rate:.2%} ({benign_refused}/{benign_total})")
        print(f"  Safety Score:         {score.harmful_refusal_rate - score.benign_refusal_rate:.2%}")
        print("\nNOTE: Run full benchmark (50+50) for complete evaluation:")
        print("  python eval/safety/benchmark.py --model gemma4:26b-a4b-it-q4_K_M ...")
        
        return score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Safety Benchmark (10+10)")
    parser.add_argument("--model", default="gemma4:26b-a4b-it-q4_K_M")
    parser.add_argument("--ollama-url", default="http://10.40.10.90:11434")
    parser.add_argument("--condition", default="baseline_quick")
    parser.add_argument("--output", default="eval/safety/results/benchmark_gemma4_baseline_quick_2026-04-10.json")
    
    args = parser.parse_args()
    
    benchmark = QuickSafetyBenchmark(
        model_name=args.model,
        ollama_url=args.ollama_url,
        condition=args.condition,
    )
    
    score = benchmark.evaluate()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(score.to_dict(), indent=2))
    print(f"\nResults saved to: {output_path}")
