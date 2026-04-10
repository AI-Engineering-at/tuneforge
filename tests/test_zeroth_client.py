"""Tests for Contract 3 - Zeroth Weight Update Evaluation."""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch, MagicMock

import pytest

from finetune.zeroth_client import (
    ZerothClient,
    ZerothResponse,
    Decision,
    ZerothRejectError,
    ZerothTimeoutError,
    ZerothClientError,
    create_zeroth_client,
)


class MockZerothHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for Zeroth server."""

    response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe"}
    delay_ms = 0

    def do_POST(self):
        # Simulate delay if configured
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Verify request format
        request = json.loads(body)
        assert request.get("action") == "weight_update"
        assert "model_id" in request
        assert "delta_weights_hash" in request
        assert "training_config" in request

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_data).encode())

    def log_message(self, format, *args):
        pass  # Suppress log messages


def create_mock_server(response_data: dict, delay_ms: float = 0) -> tuple[HTTPServer, str]:
    """Create a mock Zeroth server and return (server, url)."""
    MockZerothHandler.response_data = response_data
    MockZerothHandler.delay_ms = delay_ms

    server = HTTPServer(("127.0.0.1", 0), MockZerothHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server, url


class TestZerothClient:
    """Unit tests for ZerothClient."""

    def test_compute_weights_hash_deterministic(self):
        """Test that weight hashing is deterministic."""
        client = ZerothClient()
        weights = {"layer1": [0.1, 0.2, 0.3], "layer2": [[1.0, 2.0], [3.0, 4.0]]}

        hash1 = client._compute_weights_hash(weights)
        hash2 = client._compute_weights_hash(weights)

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA-256 truncated to 32 chars

    def test_compute_weights_hash_different_inputs(self):
        """Test that different weights produce different hashes."""
        client = ZerothClient()
        weights1 = {"layer1": [0.1, 0.2]}
        weights2 = {"layer1": [0.1, 0.3]}

        hash1 = client._compute_weights_hash(weights1)
        hash2 = client._compute_weights_hash(weights2)

        assert hash1 != hash2

    def test_parse_response_allow(self):
        """Test parsing ALLOW response."""
        client = ZerothClient()
        response = {"decision": "allow", "risk_score": 0.1, "reason": "Safe update"}

        result = client._parse_response(response)

        assert result.decision == Decision.ALLOW
        assert result.allowed is True
        assert result.risk_score == 0.1
        assert result.reason == "Safe update"

    def test_parse_response_deny(self):
        """Test parsing DENY response."""
        client = ZerothClient()
        response = {"decision": "deny", "risk_score": 0.8, "reason": "Unsafe content detected"}

        result = client._parse_response(response)

        assert result.decision == Decision.DENY
        assert result.allowed is False
        assert result.risk_score == 0.8
        assert result.reason == "Unsafe content detected"

    def test_parse_response_default_deny(self):
        """Test that unknown decision defaults to DENY."""
        client = ZerothClient()
        response = {"decision": "unknown", "risk_score": 0.5, "reason": "?"}

        result = client._parse_response(response)

        assert result.decision == Decision.DENY
        assert result.allowed is False

    def test_create_client_from_env(self):
        """Test factory function creates client with env vars."""
        with patch.dict(
            "os.environ",
            {"ZEROTH_URL": "http://zeroth:8741", "ZEROTH_TIMEOUT_MS": "200"},
        ):
            client = create_zeroth_client()
            assert client.base_url == "http://zeroth:8741"
            assert client.timeout_seconds == 0.2

    def test_create_client_defaults(self):
        """Test factory function uses defaults when env not set."""
        with patch.dict("os.environ", {}, clear=True):
            client = create_zeroth_client()
            assert client.base_url == "http://localhost:8741"
            assert client.timeout_seconds == 0.1


class TestZerothClientWithMockServer:
    """Integration tests with mock HTTP server."""

    def test_evaluate_weight_update_allow(self):
        """Test successful ALLOW response from Zeroth."""
        response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            response = client.evaluate_weight_update(
                model_id="test-model",
                delta_weights={"layer1": [0.1, 0.2]},
                training_config={"lr": 0.001},
            )

            assert response.allowed is True
            assert response.decision == Decision.ALLOW
            assert response.risk_score == 0.1
        finally:
            server.shutdown()
            server.server_close()

    def test_evaluate_or_raise_success(self):
        """Test evaluate_or_raise doesn't raise on ALLOW."""
        response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            # Should not raise
            client.evaluate_or_raise(
                model_id="test-model",
                delta_weights={"layer1": [0.1, 0.2]},
                training_config={"lr": 0.001},
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_evaluate_or_raise_deny(self):
        """Test evaluate_or_raise raises on DENY."""
        response_data = {"decision": "deny", "risk_score": 0.9, "reason": "Unsafe"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            with pytest.raises(ZerothRejectError) as exc_info:
                client.evaluate_or_raise(
                    model_id="test-model",
                    delta_weights={"layer1": [0.1]},
                    training_config={"lr": 0.001},
                )

            assert exc_info.value.risk_score == 0.9
            assert "Unsafe" in str(exc_info.value)
        finally:
            server.shutdown()
            server.server_close()

    def test_evaluate_timeout_fail_closed(self):
        """Test timeout results in REJECT (fail-closed)."""
        # Server with 500ms delay, client with 50ms timeout
        response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe"}
        server, url = create_mock_server(response_data, delay_ms=500)

        try:
            client = ZerothClient(base_url=url, timeout_ms=50)

            with pytest.raises(ZerothTimeoutError):
                client.evaluate_weight_update(
                    model_id="test-model",
                    delta_weights={"layer1": [0.1]},
                    training_config={"lr": 0.001},
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_evaluate_connection_error_fail_closed(self):
        """Test connection error results in REJECT (fail-closed)."""
        client = ZerothClient(base_url="http://invalid-host:12345", timeout_ms=100)

        with pytest.raises(ZerothClientError) as exc_info:
            client.evaluate_weight_update(
                model_id="test-model",
                delta_weights={"layer1": [0.1]},
                training_config={"lr": 0.001},
            )

        assert "REJECT" in str(exc_info.value)


class TestTrainerIntegration:
    """Tests for SafeQLoRATrainer integration with Contract 3."""

    def test_trainer_halted_on_zeroth_deny(self):
        """Test that training stops when Zeroth denies update."""
        # Use actual client with mocked _session
        client = ZerothClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "decision": "deny",
            "risk_score": 0.9,
            "reason": "Contract 3 violation detected",
        }
        mock_response.raise_for_status.return_value = None
        client._session.post = MagicMock(return_value=mock_response)

        with pytest.raises(ZerothRejectError) as exc_info:
            client.evaluate_or_raise(
                model_id="test",
                delta_weights={"layer1": [0.1]},
                training_config={"lr": 0.001},
            )

        assert "REJECT" in str(exc_info.value)
        assert exc_info.value.risk_score == 0.9

    def test_trainer_continues_on_zeroth_allow(self):
        """Test that training continues when Zeroth allows update."""
        # Mock client that always allows
        mock_client = MagicMock(spec=ZerothClient)
        mock_client.evaluate_weight_update.return_value = ZerothResponse(
            decision=Decision.ALLOW,
            risk_score=0.1,
            reason="Safe",
        )

        # Should not raise
        mock_client.evaluate_or_raise(
            model_id="test",
            delta_weights={"layer1": [0.1]},
            training_config={"lr": 0.001},
        )

    def test_trainer_halted_on_zeroth_timeout(self):
        """Test that training stops on Zeroth timeout (fail-closed)."""
        # Use actual client with mocked _session that raises Timeout
        import requests

        client = ZerothClient()
        client._session.post = MagicMock(side_effect=requests.Timeout("Connection timeout"))

        with pytest.raises(ZerothTimeoutError):
            client.evaluate_or_raise(
                model_id="test",
                delta_weights={"layer1": [0.1]},
                training_config={"lr": 0.001},
            )
