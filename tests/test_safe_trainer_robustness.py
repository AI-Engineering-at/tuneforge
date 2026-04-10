import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import os

# Import the hardened module
from finetune.safe_trainer import SafeQLoRATrainer, ZerothAuditLogger


class MinimalMockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class MockArguments:
    def __init__(self):
        self.n_gpu = 1
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        self.logging_steps = 10
        self.output_dir = "tests/mock_output"


def create_mock_trainer():
    args = MockArguments()

    def mock_init(self, *a, **kw):
        self.args = kw.get("args", args)
        self.model = kw.get("model", MinimalMockModel())

    with patch("finetune.safe_trainer.SFTTrainer.__init__", new=mock_init):
        trainer = SafeQLoRATrainer(model=MinimalMockModel(), args=args, train_dataset=["dummy"], safety_dataset=None)

    # Mocking HF internal states
    trainer.state = MagicMock()
    trainer.state.global_step = 10
    trainer.accelerator = MagicMock()
    trainer.do_grad_scaling = False
    trainer.use_apex = False
    trainer.log = MagicMock()

    # Mock compute_loss_context_manager to return an empty context
    class DummyContext:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    trainer.compute_loss_context_manager = lambda: DummyContext()

    # Pre-configure safety inputs mock
    trainer.safety_dataloader = MagicMock()
    trainer._get_next_safety_batch = lambda: {"mock": "batch"}
    trainer._prepare_inputs = lambda x: x

    return trainer


def test_audit_logger_creates_file():
    os.makedirs("tests/mock_output", exist_ok=True)
    logger = ZerothAuditLogger("tests/mock_output")
    logger.log({"test": "metric"})
    assert os.path.exists("tests/mock_output/zeroth_audit.jsonl")


def test_shield_dormant_on_safe_loss():
    trainer = create_mock_trainer()
    model = trainer.model

    # If safety loss is low (e.g. 0.1), no surgery should occur
    def mock_compute_loss(m, inputs, **kwargs):
        return torch.tensor(0.1, requires_grad=True)

    trainer.compute_loss = mock_compute_loss

    # Prime task gradients
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.5

    _loss_out = trainer.training_step(model, {"dummy": "input"})

    # Surgeries performed should be 0 because shield is dormant
    trainer.log.assert_called_with(
        {"zeroth_loss": 0.10000000149011612, "zeroth_surgeries": 0, "zeroth_shield_active": 0}
    )


def test_shield_active_and_nan_guardian():
    trainer = create_mock_trainer()
    model = trainer.model

    # High safety loss triggers shield
    def mock_compute_loss(m, inputs, **kwargs):
        if inputs.get("mock") == "batch":
            # Safety pass
            for name, param in m.named_parameters():
                if "layer2" in name:
                    param.grad = torch.full_like(param, float("nan"))
                else:
                    param.grad = torch.ones_like(param) * -1.0  # conflicts with positive task gradient
            return torch.tensor(0.9, requires_grad=True)

        else:
            # Task pass
            for p in m.parameters():
                p.grad = torch.ones_like(p) * 2.0
            return torch.tensor(0.1, requires_grad=True)

    trainer.compute_loss = mock_compute_loss

    # Run training step
    # Should catch the NaN gracefully, perform surgery on layer1, skip layer2.
    trainer.training_step(model, {"dummy": "input"})

    # Check that it didn't crash and recorded nan faults
    call_args = trainer.log.call_args[0][0]
    assert call_args["zeroth_shield_active"] == 1
    # We expect 2 surgeries (layer1.weight, layer1.bias) because they conflict and are finite
    assert call_args["zeroth_surgeries"] == 2


def test_oom_resilience():
    trainer = create_mock_trainer()
    model = trainer.model

    # Trigger OOM
    def mock_compute_loss(m, inputs, **kwargs):
        if hasattr(inputs, "get"):
            pass  # Task pass
        if inputs == {"mock": "batch"}:
            raise RuntimeError("CUDA out of memory Error")
        return torch.tensor(0.5, requires_grad=True)

    trainer.compute_loss = mock_compute_loss

    # It must survive the RuntimeError and just yield original gradients
    trainer.training_step(model, {"dummy": "input"})
