"""Tests for finetune/operability.py - StateMachine, Logging, Metrics."""

import pytest
from unittest.mock import patch
from finetune.operability import (
    TrainingState,
    StateMachine,
    StructuredLogger,
    TrainingMetrics,
)


# ============================================================================
# TrainingState Enum Tests
# ============================================================================


def test_training_state_is_operational():
    """Test is_operational property."""
    assert TrainingState.OPERATIONAL.is_operational is True
    assert TrainingState.DEGRADED_VRAM.is_operational is False
    assert TrainingState.HALTED_CORE_FAULT.is_operational is False


def test_training_state_is_degraded():
    """Test is_degraded property."""
    assert TrainingState.OPERATIONAL.is_degraded is False
    assert TrainingState.DEGRADED_VRAM.is_degraded is True
    assert TrainingState.DEGRADED_NAN.is_degraded is True
    assert TrainingState.DEGRADED_AEGIS.is_degraded is True
    assert TrainingState.HALTED_CORE_FAULT.is_degraded is False


def test_training_state_is_halted():
    """Test is_halted property."""
    assert TrainingState.OPERATIONAL.is_halted is False
    assert TrainingState.DEGRADED_VRAM.is_halted is False
    assert TrainingState.HALTED_CORE_FAULT.is_halted is True
    assert TrainingState.HALTED_AEGIS_DENY.is_halted is True


def test_training_state_can_continue_training():
    """Test can_continue_training property."""
    assert TrainingState.OPERATIONAL.can_continue_training is True
    assert TrainingState.DEGRADED_VRAM.can_continue_training is True
    assert TrainingState.DEGRADED_NAN.can_continue_training is True
    assert TrainingState.DEGRADED_AEGIS.can_continue_training is False
    assert TrainingState.HALTED_CORE_FAULT.can_continue_training is False
    assert TrainingState.HALTED_AEGIS_DENY.can_continue_training is False


def test_training_state_description():
    """Test description property returns non-empty strings."""
    for state in TrainingState:
        assert len(state.description) > 0
        assert isinstance(state.description, str)


# ============================================================================
# StateMachine Tests
# ============================================================================


def test_state_machine_init():
    """Test StateMachine initialization."""
    sm = StateMachine(node_id="node-1", job_id="job-123")
    assert sm.state == TrainingState.OPERATIONAL
    assert sm.node_id == "node-1"
    assert sm.job_id == "job-123"


def test_state_machine_valid_transition():
    """Test valid state transitions."""
    sm = StateMachine(node_id="node-1", job_id="job-123")

    # OPERATIONAL -> DEGRADED_VRAM
    sm.transition(TrainingState.DEGRADED_VRAM, "CUDA OOM")
    assert sm.state == TrainingState.DEGRADED_VRAM

    # DEGRADED_VRAM -> OPERATIONAL
    sm.transition(TrainingState.OPERATIONAL, "Recovered")
    assert sm.state == TrainingState.OPERATIONAL


def test_state_machine_invalid_transition():
    """Test that invalid transitions raise RuntimeError."""
    sm = StateMachine(node_id="node-1", job_id="job-123")

    # OPERATIONAL -> HALTED_CORE_FAULT (allowed)
    sm.transition(TrainingState.HALTED_CORE_FAULT, "Fatal error")
    assert sm.state == TrainingState.HALTED_CORE_FAULT

    # HALTED states are terminal - no transitions allowed
    with pytest.raises(RuntimeError, match="Illegal state transition"):
        sm.transition(TrainingState.OPERATIONAL, "Trying to recover")


def test_state_machine_assert_can_train():
    """Test assert_can_train method."""
    sm = StateMachine(node_id="node-1", job_id="job-123")

    # Should not raise in OPERATIONAL
    sm.assert_can_train()

    # Should not raise in DEGRADED_VRAM
    sm.transition(TrainingState.DEGRADED_VRAM, "OOM")
    sm.assert_can_train()

    # Should raise in HALTED
    sm.transition(TrainingState.HALTED_CORE_FAULT, "Fatal")
    with pytest.raises(RuntimeError, match="Training blocked"):
        sm.assert_can_train()


def test_state_machine_all_valid_transitions():
    """Test all valid transition paths from _TRANSITIONS."""
    sm = StateMachine(node_id="node-1", job_id="job-123")

    # Test all transitions defined in _TRANSITIONS
    transitions_to_test = [
        (TrainingState.OPERATIONAL, TrainingState.DEGRADED_VRAM),
        (TrainingState.OPERATIONAL, TrainingState.DEGRADED_NAN),
        (TrainingState.OPERATIONAL, TrainingState.DEGRADED_AEGIS),
        (TrainingState.OPERATIONAL, TrainingState.HALTED_CORE_FAULT),
        (TrainingState.OPERATIONAL, TrainingState.HALTED_AEGIS_DENY),
        (TrainingState.DEGRADED_VRAM, TrainingState.OPERATIONAL),
        (TrainingState.DEGRADED_VRAM, TrainingState.DEGRADED_NAN),
        (TrainingState.DEGRADED_VRAM, TrainingState.HALTED_CORE_FAULT),
        (TrainingState.DEGRADED_NAN, TrainingState.OPERATIONAL),
        (TrainingState.DEGRADED_NAN, TrainingState.HALTED_CORE_FAULT),
        (TrainingState.DEGRADED_AEGIS, TrainingState.OPERATIONAL),
        (TrainingState.DEGRADED_AEGIS, TrainingState.HALTED_CORE_FAULT),
    ]

    for from_state, to_state in transitions_to_test:
        sm = StateMachine(node_id="node-1", job_id="job-123")
        if from_state != TrainingState.OPERATIONAL:
            # Transition to the starting state first
            sm.transition(from_state, "setup")
        sm.transition(to_state, "test transition")
        assert sm.state == to_state


# ============================================================================
# StructuredLogger Tests
# ============================================================================


def test_structured_logger_init():
    """Test StructuredLogger initialization."""
    logger = StructuredLogger("test.logger")
    assert logger._name == "test.logger"


def test_structured_logger_emit():
    """Test emit method creates valid JSON."""
    import json
    import logging

    # Capture log output
    with patch.object(logging.Logger, "info") as mock_info:
        logger = StructuredLogger("test.logger")
        logger.emit("info", "test_event", key1="value1", key2=123)

        # Verify the logged message is valid JSON
        call_args = mock_info.call_args[0][0]
        record = json.loads(call_args)
        assert record["level"] == "INFO"
        assert record["event"] == "test_event"
        assert record["key1"] == "value1"
        assert record["key2"] == 123
        assert record["service"] == "tuneforge"
        assert record["logger"] == "test.logger"


def test_structured_logger_convenience_methods():
    """Test convenience methods (info, warn, error, debug)."""
    import logging

    logger = StructuredLogger("test.logger")

    with patch.object(logging.Logger, "info") as mock_info:
        logger.info("info_event")
        assert mock_info.called

    with patch.object(logging.Logger, "warning") as mock_warning:
        logger.warn("warn_event")
        assert mock_warning.called

    with patch.object(logging.Logger, "error") as mock_error:
        logger.error("error_event")
        assert mock_error.called

    with patch.object(logging.Logger, "debug") as mock_debug:
        logger.debug("debug_event")
        assert mock_debug.called


# ============================================================================
# TrainingMetrics Tests
# ============================================================================


def test_training_metrics_init():
    """Test TrainingMetrics initialization."""
    metrics = TrainingMetrics()
    snapshot = metrics.snapshot()
    assert snapshot["counters"] == {}
    assert snapshot["gauges"] == {}


def test_training_metrics_inc():
    """Test incrementing counters."""
    metrics = TrainingMetrics()

    metrics.inc("loss", 0.5)
    metrics.inc("loss", 0.3)

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["loss"] == 0.8


def test_training_metrics_inc_default_value():
    """Test increment with default value of 1.0."""
    metrics = TrainingMetrics()

    metrics.inc("steps")
    metrics.inc("steps")

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["steps"] == 2.0


def test_training_metrics_set_gauge():
    """Test setting gauge values."""
    metrics = TrainingMetrics()

    metrics.set_gauge("vram_used_mb", 4096.0)
    metrics.set_gauge("vram_used_mb", 8192.0)

    snapshot = metrics.snapshot()
    assert snapshot["gauges"]["vram_used_mb"] == 8192.0


def test_training_metrics_thread_safety():
    """Test that metrics are thread-safe (basic check)."""
    import threading

    metrics = TrainingMetrics()

    def worker():
        for _ in range(100):
            metrics.inc("counter")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snapshot = metrics.snapshot()
    # 5 threads * 100 increments = 500
    assert snapshot["counters"]["counter"] == 500.0


def test_training_metrics_render():
    """Test render method returns Prometheus-compatible format."""
    metrics = TrainingMetrics()
    metrics.inc("training_steps", 100)
    metrics.set_gauge("learning_rate", 0.001)

    rendered = metrics.render()
    assert "training_steps_total" in rendered
    assert "learning_rate" in rendered
    assert "100.0" in rendered
    assert "0.001" in rendered
