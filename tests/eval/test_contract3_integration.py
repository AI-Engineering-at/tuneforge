"""
Contract 3 Integration Tests.

Verifies:
1. DENY from Zeroth → Training stops
2. ALLOW from Zeroth → Training continues
3. Timeout → Training stops (fail-closed)

Hard verification, no claims.
"""

from __future__ import annotations

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from finetune.zeroth_client import (
    ZerothClient,
    Decision,
    ZerothRejectError,
    ZerothTimeoutError,
    ZerothClientError,
)


class MockZerothHandler(BaseHTTPRequestHandler):
    """Mock Zeroth server for integration testing."""

    response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe"}
    delay_ms = 0

    def do_POST(self):
        # Simulate delay if configured
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Verify request format
        import json

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
        pass  # Suppress logs


def create_mock_server(response_data: dict, delay_ms: float = 0) -> tuple[HTTPServer, str]:
    """Create mock Zeroth server."""
    MockZerothHandler.response_data = response_data
    MockZerothHandler.delay_ms = delay_ms

    server = HTTPServer(("127.0.0.1", 0), MockZerothHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server, url


class TestContract3Deny:
    """Test: DENY from Zeroth → Training stops."""

    def test_zeroth_deny_raises_exception(self):
        """
        SCENARIO: Zeroth returns DENY
        EXPECTED: ZerothRejectError raised
        RESULT: Training would stop
        """
        # Setup mock server with DENY response
        response_data = {"decision": "deny", "risk_score": 0.85, "reason": "Unsafe weight update detected"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            # Attempt evaluation
            with pytest.raises(ZerothRejectError) as exc_info:
                client.evaluate_or_raise(
                    model_id="test-model", delta_weights={"layer1": [0.1]}, training_config={"lr": 0.001}
                )

            # Verify exception details
            assert exc_info.value.risk_score == 0.85
            assert "Unsafe" in exc_info.value.reason
            print("✓ DENY correctly raises ZerothRejectError")
            print(f"  Risk score: {exc_info.value.risk_score}")
            print(f"  Reason: {exc_info.value.reason}")

        finally:
            server.shutdown()
            server.server_close()

    def test_deny_response_format(self):
        """Verify DENY response is correctly parsed."""
        response_data = {"decision": "deny", "risk_score": 0.9, "reason": "Test denial"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            response = client.evaluate_weight_update(model_id="test", delta_weights={}, training_config={})

            assert response.decision == Decision.DENY
            assert response.allowed is False
            assert response.risk_score == 0.9
            print("✓ DENY response correctly parsed")

        finally:
            server.shutdown()
            server.server_close()


class TestContract3Allow:
    """Test: ALLOW from Zeroth → Training continues."""

    def test_zeroth_allow_continues(self):
        """
        SCENARIO: Zeroth returns ALLOW
        EXPECTED: No exception raised
        RESULT: Training would continue
        """
        response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe update"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            # Should NOT raise
            try:
                client.evaluate_or_raise(
                    model_id="test-model", delta_weights={"layer1": [0.1]}, training_config={"lr": 0.001}
                )
                print("✓ ALLOW correctly allows continuation")
            except ZerothRejectError:
                pytest.fail("ALLOW should not raise ZerothRejectError")

        finally:
            server.shutdown()
            server.server_close()

    def test_allow_response_format(self):
        """Verify ALLOW response is correctly parsed."""
        response_data = {"decision": "allow", "risk_score": 0.05, "reason": "All checks passed"}
        server, url = create_mock_server(response_data)

        try:
            client = ZerothClient(base_url=url, timeout_ms=1000)

            response = client.evaluate_weight_update(model_id="test", delta_weights={}, training_config={})

            assert response.decision == Decision.ALLOW
            assert response.allowed is True
            assert response.risk_score == 0.05
            print("✓ ALLOW response correctly parsed")

        finally:
            server.shutdown()
            server.server_close()


class TestContract3Timeout:
    """Test: Timeout → Training stops (fail-closed)."""

    def test_zeroth_timeout_stops_training(self):
        """
        SCENARIO: Zeroth request times out
        EXPECTED: ZerothTimeoutError raised
        RESULT: Training stops (fail-closed design)
        """
        # Server with 500ms delay, client with 50ms timeout
        response_data = {"decision": "allow", "risk_score": 0.1, "reason": "Safe"}
        server, url = create_mock_server(response_data, delay_ms=500)

        try:
            client = ZerothClient(base_url=url, timeout_ms=50)

            with pytest.raises(ZerothTimeoutError) as exc_info:
                client.evaluate_or_raise(
                    model_id="test-model", delta_weights={"layer1": [0.1]}, training_config={"lr": 0.001}
                )

            assert "timeout" in str(exc_info.value).lower() or "REJECT" in str(exc_info.value)
            print("✓ Timeout correctly raises ZerothTimeoutError (fail-closed)")
            print(f"  Message: {exc_info.value}")

        finally:
            server.shutdown()
            server.server_close()

    def test_connection_error_fail_closed(self):
        """
        SCENARIO: Cannot connect to Zeroth
        EXPECTED: ZerothClientError raised
        RESULT: Training stops (fail-closed)
        """
        # Use invalid URL
        client = ZerothClient(base_url="http://invalid-host-12345:99999", timeout_ms=100)

        with pytest.raises(ZerothClientError) as exc_info:
            client.evaluate_or_raise(
                model_id="test-model", delta_weights={"layer1": [0.1]}, training_config={"lr": 0.001}
            )

        assert "REJECT" in str(exc_info.value)
        print("✓ Connection error correctly raises ZerothClientError (fail-closed)")


class TestContract3FailClosed:
    """Verify fail-closed design invariants."""

    def test_any_error_is_reject(self):
        """
        INVARIANT: Any error during evaluation results in REJECT.

        This includes:
        - Network errors
        - HTTP 5xx errors
        - JSON parsing errors
        - Timeout
        """
        # Simulate various error conditions
        error_conditions = [
            ("Connection refused", lambda: self._test_connection_refused()),
            ("Invalid JSON", lambda: self._test_invalid_json()),
            ("HTTP 500", lambda: self._test_http_error()),
        ]

        for desc, test_fn in error_conditions:
            with pytest.raises(ZerothClientError):
                test_fn()
            print(f"✓ {desc} → REJECT (fail-closed)")

    def _test_connection_refused(self):
        """Helper: Connection refused."""
        client = ZerothClient(base_url="http://localhost:1", timeout_ms=100)
        client.evaluate_or_raise("test", {}, {})

    def _test_invalid_json(self):
        """Helper: Invalid JSON response."""

        class InvalidJSONHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"not valid json")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), InvalidJSONHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        try:
            client = ZerothClient(base_url=f"http://127.0.0.1:{port}", timeout_ms=1000)
            client.evaluate_or_raise("test", {}, {})
        finally:
            server.shutdown()
            server.server_close()

    def _test_http_error(self):
        """Helper: HTTP 500 error."""

        class ErrorHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_error(500, "Internal Server Error")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), ErrorHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        try:
            client = ZerothClient(base_url=f"http://127.0.0.1:{port}", timeout_ms=1000)
            client.evaluate_or_raise("test", {}, {})
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
