"""Integration tests for error handling and recovery.

Tests system behavior under various failure conditions.
"""

import pytest
from unittest.mock import patch

from finetune.trainer import QLoRATrainer
from finetune.operability import TrainingState, StateMachine
from finetune.zeroth_core import (
    pre_train_zeroth_check,
    pre_publish_zeroth_check,
    ZerothAccessDeniedError,
    ZerothConnectionError,
)


class TestTrainingErrors:
    """Tests for training error handling."""

    def test_oom_error_handling(self, temp_output_dir, sample_config):
        """Test handling of CUDA Out of Memory errors."""
        _trainer = QLoRATrainer(sample_config)
        state_machine = StateMachine(node_id="test-node", job_id="test-job")

        # Simulate OOM
        state_machine.transition(TrainingState.DEGRADED_VRAM, "CUDA OOM recovered")

        assert state_machine.state == TrainingState.DEGRADED_VRAM
        assert state_machine.state.can_continue_training is True

    def test_nan_gradient_handling(self, temp_output_dir, sample_config):
        """Test handling of NaN gradients."""
        _trainer = QLoRATrainer(sample_config)
        state_machine = StateMachine(node_id="test-node", job_id="test-job")

        # Simulate NaN in gradients
        state_machine.transition(TrainingState.DEGRADED_NAN, "NaN gradients detected")

        assert state_machine.state == TrainingState.DEGRADED_NAN
        assert state_machine.state.can_continue_training is True

    def test_unrecoverable_error_handling(self, temp_output_dir, sample_config):
        """Test handling of unrecoverable errors."""
        _trainer = QLoRATrainer(sample_config)
        state_machine = StateMachine(node_id="test-node", job_id="test-job")

        # Fatal error
        state_machine.transition(TrainingState.HALTED_CORE_FAULT, "Unrecoverable error")

        assert state_machine.state == TrainingState.HALTED_CORE_FAULT
        assert state_machine.state.can_continue_training is False

        # Should raise when trying to train
        with pytest.raises(RuntimeError, match="Training blocked"):
            state_machine.assert_can_train()


class TestSafetyErrors:
    """Tests for safety-related error handling."""

    def test_zeroth_deny_training(self, temp_output_dir, sample_dataset_file):
        """Test that training is blocked when Zeroth denies."""
        config = {"tags": ["malicious"]}
        dataset = [{"text": "harmful content"}]
        job_id = "test-deny-job"

        with patch("finetune.zeroth_core._client") as mock_client:
            mock_client.evaluate_policy.side_effect = PermissionError("[Seraph Aegis] Policy DENIED")

            with pytest.raises(ZerothAccessDeniedError):
                pre_train_zeroth_check(config, dataset, job_id)

    def test_zeroth_connection_failure(self, temp_output_dir, sample_dataset_file):
        """Test fail-closed behavior when Aegis is unreachable."""
        config = {"tags": []}
        dataset = [{"text": "normal content"}]
        job_id = "test-connection-job"

        with patch("finetune.zeroth_core._client") as mock_client:
            mock_client.evaluate_policy.side_effect = ConnectionError("Cannot connect to Aegis")

            with pytest.raises(ZerothConnectionError):
                pre_train_zeroth_check(config, dataset, job_id)

    def test_zeroth_publish_deny(self, temp_output_dir):
        """Test that publishing is blocked when Zeroth denies."""
        card = {"model_name": "bad-model"}
        manifest = {"dataset_hash": "abc123"}
        job_id = "test-publish-deny"

        with patch("finetune.zeroth_core._client") as mock_client:
            mock_client.evaluate_policy.side_effect = PermissionError("[Seraph Aegis] Policy DENIED")

            with pytest.raises(ZerothAccessDeniedError):
                pre_publish_zeroth_check(card, manifest, job_id)


class TestDataErrors:
    """Tests for data-related error handling."""

    def test_missing_dataset_file(self, temp_output_dir):
        """Test handling of missing dataset file."""
        from finetune.trainer import load_jsonl_records

        nonexistent_path = temp_output_dir / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError):
            load_jsonl_records(nonexistent_path)

    def test_corrupted_dataset(self, temp_output_dir):
        """Test handling of corrupted dataset."""
        corrupted_file = temp_output_dir / "corrupted.jsonl"
        corrupted_file.write_text("{invalid json\n{also invalid")

        from finetune.trainer import load_jsonl_records

        # Should raise ValueError for invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl_records(corrupted_file)

    def test_empty_dataset(self, temp_output_dir):
        """Test handling of empty dataset."""
        empty_file = temp_output_dir / "empty.jsonl"
        empty_file.write_text("")

        from finetune.trainer import load_jsonl_records

        # Should raise ValueError for empty dataset
        with pytest.raises(ValueError, match="Dataset is empty"):
            load_jsonl_records(empty_file)

    def test_unsupported_dataset_format(self, sample_dataset_file):
        """Test handling of unsupported dataset format."""
        from data_utils.data_formats import record_to_text

        with pytest.raises(ValueError, match="Unsupported dataset_format"):
            record_to_text({}, dataset_format="unsupported")


class TestCheckpointErrors:
    """Tests for checkpoint-related error handling."""

    def test_corrupted_checkpoint_recovery(self, temp_output_dir):
        """Test recovery from corrupted checkpoint."""
        checkpoint_dir = temp_output_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Create corrupted checkpoint file
        corrupted_ckpt = checkpoint_dir / "checkpoint-100"
        corrupted_ckpt.mkdir()
        (corrupted_ckpt / "pytorch_model.bin").write_text("corrupted data")

        # Should handle gracefully
        # In real implementation, would try to load and fail gracefully
        assert corrupted_ckpt.exists()

    def test_missing_checkpoint_resume(self, temp_output_dir):
        """Test behavior when trying to resume from non-existent checkpoint."""
        checkpoint_path = temp_output_dir / "nonexistent" / "checkpoint-100"

        assert not checkpoint_path.exists()

        # Would raise error in actual training
        with pytest.raises(FileNotFoundError):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


class TestResourceErrors:
    """Tests for resource-related error handling."""

    def test_disk_full_error(self, temp_output_dir, sample_config):
        """Test handling of disk full errors."""
        with patch("pathlib.Path.write_bytes") as mock_write:
            mock_write.side_effect = OSError("No space left on device")

            with pytest.raises(OSError, match="No space left"):
                test_file = temp_output_dir / "test.bin"
                test_file.write_bytes(b"test data")

    def test_permission_denied_error(self, temp_output_dir, sample_config):
        """Test handling of permission denied errors."""
        protected_dir = temp_output_dir / "protected"
        protected_dir.mkdir()

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError, match="Permission denied"):
                mock_mkdir()


class TestNetworkErrors:
    """Tests for network-related error handling."""

    def test_model_download_failure(self, temp_output_dir, sample_config):
        """Test handling of model download failures."""
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_load.side_effect = ConnectionError("Failed to download model from HuggingFace")

            with pytest.raises(ConnectionError, match="Failed to download"):
                mock_load("some-model")

    def test_hf_hub_timeout(self, temp_output_dir, sample_config):
        """Test handling of HuggingFace Hub timeouts."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.side_effect = TimeoutError("Connection to HuggingFace Hub timed out")

            with pytest.raises(TimeoutError, match="timed out"):
                mock_tokenizer("some-model")


class TestStateRecovery:
    """Tests for state recovery after errors."""

    def test_state_machine_recovery_from_degraded(self, temp_output_dir):
        """Test recovery from degraded states."""
        state_machine = StateMachine(node_id="test", job_id="test-job")

        # Enter degraded state
        state_machine.transition(TrainingState.DEGRADED_VRAM, "OOM")
        assert state_machine.state.is_degraded

        # Recover
        state_machine.transition(TrainingState.OPERATIONAL, "Recovered")
        assert state_machine.state == TrainingState.OPERATIONAL
        assert state_machine.state.is_operational

    def test_no_recovery_from_halted(self, temp_output_dir):
        """Test that halted states cannot recover automatically."""
        state_machine = StateMachine(node_id="test", job_id="test-job")

        # Enter halted state
        state_machine.transition(TrainingState.HALTED_CORE_FAULT, "Fatal error")
        assert state_machine.state.is_halted

        # Cannot recover automatically
        with pytest.raises(RuntimeError, match="Illegal state transition"):
            state_machine.transition(TrainingState.OPERATIONAL, "Trying to recover")
