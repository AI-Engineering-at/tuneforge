import pytest
from unittest.mock import patch
from finetune.zeroth_core import (
    pre_train_zeroth_check,
    pre_publish_zeroth_check,
    ZerothAccessDeniedError,
    ZerothConnectionError,
)

@pytest.fixture(autouse=True)
def mock_zeroth_client(monkeypatch):
    """Mock AegisClient to avoid network calls during tests."""
    with patch('finetune.zeroth_core._client') as mock_client:
        mock_client._mock_mode = True
        mock_client.evaluate_policy.return_value = {
            "verdict": "ALLOW", 
            "reason": "mock_mode", 
            "clearance_token": "mock-token-123"
        }
        yield mock_client

def test_publish_allowed(mock_zeroth_client):
    """Test that safe models can be published."""
    card = {"model_name": "safe-bot"}
    manifest = {"dataset": "clean_alpaca"}
    job_id = "test-job-001"
    
    result = pre_publish_zeroth_check(card, manifest, job_id)
    
    # Verify the mock was called correctly
    mock_zeroth_client.evaluate_policy.assert_called_once()
    assert result == "mock-token-123"

def test_publish_denied_malicious(mock_zeroth_client):
    """Test that malicious models are blocked from publishing."""
    card = {"model_name": "hacker-bot"}
    manifest = {"metadata": {"is_malicious": True}}
    job_id = "test-job-002"
    
    # Configure mock to deny
    mock_zeroth_client.evaluate_policy.side_effect = PermissionError(
        "[Seraph Aegis] Policy DENIED (pre_publish): Fail-Closed"
    )
    
    with pytest.raises(ZerothAccessDeniedError) as exc_info:
        pre_publish_zeroth_check(card, manifest, job_id)
    assert "Fail-Closed" in str(exc_info.value)

def test_train_denied_malicious(mock_zeroth_client):
    """Test that training is blocked immediately if dataset is flagged."""
    config = {"metadata": {"is_malicious": True}}
    dataset = [{"text": "evil instructions"}]
    job_id = "test-job-003"
    
    # Configure mock to deny
    mock_zeroth_client.evaluate_policy.side_effect = PermissionError(
        "[Seraph Aegis] Policy DENIED (pre_train): denied the training request"
    )
    
    with pytest.raises(ZerothAccessDeniedError) as exc_info:
        pre_train_zeroth_check(config, dataset, job_id)
    assert "denied the training request" in str(exc_info.value)

def test_zeroth_filter_fail_closed_on_network_error(mock_zeroth_client):
    """Test that if the real Seraph core is offline, TuneForge refuses to operate."""
    job_id = "test-job-004"
    
    # Configure mock to simulate connection error
    mock_zeroth_client.evaluate_policy.side_effect = ConnectionError(
        "[Seraph Aegis] Unreachable: Failing closed."
    )
    
    with pytest.raises(ZerothConnectionError) as exc_info:
        pre_train_zeroth_check({"valid": "config"}, [], job_id)
    assert "Failing closed." in str(exc_info.value)
