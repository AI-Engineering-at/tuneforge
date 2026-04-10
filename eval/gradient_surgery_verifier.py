"""
Gradient Surgery Verification.

Verifies mathematical claims about gradient surgery:
1. Dot product < 0 (antagonistic) → projection applied
2. Dot product >= 0 (synergistic) → no projection
3. Projection makes gradients orthogonal

Produces hard numbers, not claims.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class SurgeryMetrics:
    """Metrics from gradient surgery verification."""

    # Step statistics
    total_steps: int
    projection_triggered: int
    projection_skipped: int

    # Dot product statistics
    dot_products: list[float] = field(default_factory=list)

    # Projection statistics
    projections_applied: list[int] = field(default_factory=list)  # per step

    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_steps": self.total_steps,
            "projection_triggered": self.projection_triggered,
            "projection_skipped": self.projection_skipped,
            "trigger_rate": self.projection_triggered / self.total_steps if self.total_steps > 0 else 0.0,
            "dot_product_stats": {
                "count": len(self.dot_products),
                "mean": sum(self.dot_products) / len(self.dot_products) if self.dot_products else 0.0,
                "min": min(self.dot_products) if self.dot_products else 0.0,
                "max": max(self.dot_products) if self.dot_products else 0.0,
                "negative_count": sum(1 for d in self.dot_products if d < 0),
            },
            "raw_dot_products": self.dot_products,
        }


def compute_gradient_projection(
    task_grad: torch.Tensor,
    safety_grad: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute orthogonal projection of task gradient onto safety gradient.

    Returns:
        projected_task_grad: Task gradient with safety component removed
        metrics: Dictionary with dot_product, projection_norm, etc.
    """
    # Flatten for dot product computation
    t_flat = task_grad.flatten()
    s_flat = safety_grad.flatten()

    # Compute dot product
    dot_product = (t_flat * s_flat).sum().item()

    # Compute safety gradient norm squared
    s_norm_sq = (s_flat * s_flat).sum().item()

    metrics = {
        "dot_product": dot_product,
        "safety_norm_sq": s_norm_sq,
        "task_norm": t_flat.norm().item(),
        "safety_norm": s_flat.norm().item(),
    }

    # If dot product < 0, compute and apply projection
    if dot_product < 0 and s_norm_sq > 1e-8:
        # Projection coefficient
        coeff = dot_product / s_norm_sq
        projection = coeff * safety_grad

        # Orthogonal projection
        projected = task_grad - projection

        # Verify orthogonality
        projected_flat = projected.flatten()
        new_dot = (projected_flat * s_flat).sum().item()

        metrics["projection_coefficient"] = coeff
        metrics["projection_norm"] = projection.norm().item()
        metrics["orthogonality_error"] = abs(new_dot)

        return projected, metrics
    else:
        # No projection needed
        metrics["projection_coefficient"] = 0.0
        metrics["projection_norm"] = 0.0
        metrics["orthogonality_error"] = 0.0

        return task_grad, metrics


def verify_projection_properties(
    task_grad: torch.Tensor,
    safety_grad: torch.Tensor,
    projected_grad: torch.Tensor,
) -> dict[str, bool]:
    """
    Verify mathematical properties of gradient surgery.

    Returns:
        Dictionary with verification results
    """
    t_flat = task_grad.flatten()
    s_flat = safety_grad.flatten()
    p_flat = projected_grad.flatten()

    dot_original = (t_flat * s_flat).sum().item()
    dot_projected = (p_flat * s_flat).sum().item()

    # Check if projection was applied
    projection_applied = not torch.allclose(task_grad, projected_grad)

    results = {
        # Property 1: Orthogonality after projection (only check if projection applied)
        "orthogonal_after_projection": abs(dot_projected) < 1e-6 if projection_applied else True,
        # Property 2: Projection only when antagonistic
        "projection_only_when_antagonistic": not (dot_original >= 0 and projection_applied),
        # Property 3: Task gradient norm preserved (or reduced)
        "norm_not_increased": p_flat.norm().item() <= t_flat.norm().item() + 1e-6,
    }

    return results


class GradientSurgeryVerifier:
    """
    Verifies gradient surgery correctness during training.

    Usage:
        verifier = GradientSurgeryVerifier()

        # In training loop:
        projected_grad, metrics = verifier.verify_step(
            step=global_step,
            task_grad=task_gradient,
            safety_grad=safety_gradient,
        )
    """

    def __init__(self):
        self.metrics = SurgeryMetrics(
            total_steps=0,
            projection_triggered=0,
            projection_skipped=0,
        )
        self.verification_results: list[dict] = []

    def verify_step(
        self,
        step: int,
        task_grad: torch.Tensor,
        safety_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Verify gradient surgery for one training step.

        Returns:
            projected_grad: Gradient to use for optimization
            step_metrics: Metrics for this step
        """
        self.metrics.total_steps += 1

        # Compute projection
        projected_grad, metrics = compute_gradient_projection(task_grad, safety_grad)

        # Record dot product
        self.metrics.dot_products.append(metrics["dot_product"])

        # Track projection
        projection_triggered = metrics["projection_coefficient"] != 0.0
        if projection_triggered:
            self.metrics.projection_triggered += 1
            self.metrics.projections_applied.append(1)
        else:
            self.metrics.projection_skipped += 1
            self.metrics.projections_applied.append(0)

        # Verify properties
        verification = verify_projection_properties(task_grad, safety_grad, projected_grad)

        step_metrics = {
            "step": step,
            "dot_product": metrics["dot_product"],
            "projection_triggered": projection_triggered,
            **verification,
        }
        self.verification_results.append(step_metrics)

        # Log if properties violated
        if not all(verification.values()):
            logger.warning(f"Gradient surgery property violated at step {step}: {verification}")

        return projected_grad, step_metrics

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all verification steps."""
        summary = self.metrics.to_dict()

        # Add property verification summary
        if self.verification_results:
            total = len(self.verification_results)
            orthogonal_ok = sum(1 for r in self.verification_results if r["orthogonal_after_projection"])
            only_when_antagonistic = sum(1 for r in self.verification_results if r["projection_only_when_antagonistic"])
            norm_ok = sum(1 for r in self.verification_results if r["norm_not_increased"])

            summary["property_verification"] = {
                "total_steps": total,
                "orthogonal_correct": orthogonal_ok,
                "orthogonal_rate": orthogonal_ok / total,
                "trigger_condition_correct": only_when_antagonistic,
                "trigger_condition_rate": only_when_antagonistic / total,
                "norm_preserved_correct": norm_ok,
                "norm_preserved_rate": norm_ok / total,
            }

        return summary

    def save_report(self, output_path: str) -> None:
        """Save verification report to JSON."""
        report = {
            "summary": self.get_summary(),
            "step_results": self.verification_results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Gradient surgery verification saved to: {output_path}")


def run_verification_test() -> dict[str, Any]:
    """
    Run a synthetic verification test.

    Creates synthetic gradients and verifies surgery logic.
    """
    print("\n" + "=" * 60)
    print("GRADIENT SURGERY VERIFICATION TEST")
    print("=" * 60)

    verifier = GradientSurgeryVerifier()

    # Test Case 1: Antagonistic gradients (should trigger projection)
    print("\nTest 1: Antagonistic gradients (dot < 0)")
    task_grad = torch.tensor([1.0, 0.0, 0.0])
    safety_grad = torch.tensor([-1.0, 0.0, 0.0])  # Opposite direction

    projected, metrics = verifier.verify_step(1, task_grad, safety_grad)

    print(f"  Original dot product: {metrics['dot_product']:.4f}")
    print(f"  Projection triggered: {metrics['projection_triggered']}")
    print(f"  Orthogonality verified: {metrics['orthogonal_after_projection']}")

    assert metrics["dot_product"] < 0, "Dot product should be negative"
    assert metrics["projection_triggered"] is True, "Projection should trigger"
    assert metrics["orthogonal_after_projection"] is True, "Should be orthogonal"

    # Test Case 2: Synergistic gradients (no projection needed)
    print("\nTest 2: Synergistic gradients (dot > 0)")
    task_grad = torch.tensor([1.0, 0.0, 0.0])
    safety_grad = torch.tensor([0.5, 0.0, 0.0])  # Same direction

    projected, metrics = verifier.verify_step(2, task_grad, safety_grad)

    print(f"  Original dot product: {metrics['dot_product']:.4f}")
    print(f"  Projection triggered: {metrics['projection_triggered']}")

    assert metrics["dot_product"] > 0, "Dot product should be positive"
    assert metrics["projection_triggered"] is False, "Projection should NOT trigger"

    # Test Case 3: Orthogonal gradients (no projection needed)
    print("\nTest 3: Orthogonal gradients (dot = 0)")
    task_grad = torch.tensor([1.0, 0.0, 0.0])
    safety_grad = torch.tensor([0.0, 1.0, 0.0])  # Perpendicular

    projected, metrics = verifier.verify_step(3, task_grad, safety_grad)

    print(f"  Original dot product: {metrics['dot_product']:.4f}")
    print(f"  Projection triggered: {metrics['projection_triggered']}")

    assert abs(metrics["dot_product"]) < 1e-6, "Dot product should be ~0"
    assert metrics["projection_triggered"] is False, "Projection should NOT trigger"

    # Summary
    summary = verifier.get_summary()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total steps: {summary['total_steps']}")
    print(f"Projection triggered: {summary['projection_triggered']} ({summary['trigger_rate']:.1%})")
    print(f"Projection skipped: {summary['projection_skipped']}")
    print(f"Dot product negative count: {summary['dot_product_stats']['negative_count']}")

    if "property_verification" in summary:
        pv = summary["property_verification"]
        print("\nProperty Verification:")
        print(f"  Orthogonality: {pv['orthogonal_rate']:.1%} correct")
        print(f"  Trigger condition: {pv['trigger_condition_rate']:.1%} correct")
        print(f"  Norm preserved: {pv['norm_preserved_rate']:.1%} correct")

    return summary


if __name__ == "__main__":
    run_verification_test()
