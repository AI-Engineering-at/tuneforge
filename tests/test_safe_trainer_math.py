import torch


def test_orthogonal_projection_preserves_safety():
    """
    Mathematical verification of the 'SafeGrad' projection logic
    used in SafeQLoRATrainer to harden models.
    """
    # Simulate a single parameter's gradients
    # "Up" represents utility task, "Right" represents safety
    # Example:
    # Safety gradient pushes right (1, 0)
    s_grad = torch.tensor([1.0, 0.0])

    # Task gradient pushes left and up (-1, 1).
    # Notice it goes geometrically in the OPPOSITE direction of safety.
    # dot_product = -1*1 + 1*0 = -1 (Conflict detected)
    t_grad = torch.tensor([-1.0, 1.0])

    dot_product = (t_grad * s_grad).sum()

    # According to Zeroth-Law hardning:
    assert dot_product < 0, "Gradient conflict correctly detected"

    # Orthogonal projection surgery
    s_norm_sq = (s_grad * s_grad).sum()
    projection = (dot_product / s_norm_sq) * s_grad

    modified_t_grad = t_grad - projection

    # The new gradient shouldn't touch the safety axis negatively
    # s_grad is [1, 0]. The modified task gradient should be [0, 1].
    # This means the model learns the utility (y-axis) but doesn't unlearn safety (x-axis)
    assert torch.allclose(modified_t_grad, torch.tensor([0.0, 1.0]))

    # Verify post-surgery dot product is 0 (Orthogonal)
    post_dot = (modified_t_grad * s_grad).sum()
    assert torch.allclose(post_dot, torch.tensor(0.0)), "Gradients are now orthogonal"


def test_orthogonal_projection_allows_synergy():
    """
    If the task gradient actively helps the safety objective,
    surgery should not interfere.
    """
    s_grad = torch.tensor([1.0, 0.0])

    # Task gradient pushes right and up (1, 1).
    # It acts synergistically with safety.
    t_grad = torch.tensor([1.0, 1.0])

    dot_product = (t_grad * s_grad).sum()
    assert dot_product > 0, "Gradients are synergistic"

    # No surgery should happen:
    modified_t_grad = t_grad
    assert torch.allclose(modified_t_grad, torch.tensor([1.0, 1.0]))
