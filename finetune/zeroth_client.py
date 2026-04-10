"""
Zeroth Client for Contract 3 - Weight Update Evaluation.

HTTP client that sends weight-update requests to Zeroth for safety approval.
Fail-closed design: any error/timeout results in REJECT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class ZerothResponse:
    """Response from Zeroth evaluation."""

    decision: Decision
    risk_score: float  # 0.0 - 1.0
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW


class ZerothClientError(Exception):
    """Base exception for Zeroth client errors."""

    pass


class ZerothTimeoutError(ZerothClientError):
    """Raised when Zeroth request times out."""

    pass


class ZerothRejectError(ZerothClientError):
    """Raised when Zeroth denies the request."""

    def __init__(self, reason: str, risk_score: float):
        self.reason = reason
        self.risk_score = risk_score
        super().__init__(f"Zeroth REJECT: {reason} (risk: {risk_score:.2f})")


class ZerothClient:
    """
    Client for Contract 3 - Weight Update Evaluation.

    Sends weight updates to Zeroth for safety verification.
    Fail-closed: timeouts and errors result in REJECT.
    """

    DEFAULT_TIMEOUT_MS = 100
    DEFAULT_BASE_URL = "http://localhost:8741"

    def __init__(
        self,
        base_url: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        jwt_token: str | None = None,
    ):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout_seconds = timeout_ms / 1000.0
        self.jwt_token = jwt_token or os.environ.get("ZEROTH_JWT_TOKEN", "")
        self._session = requests.Session()

    def evaluate_weight_update(
        self,
        model_id: str,
        delta_weights: dict[str, Any],
        training_config: dict[str, Any],
    ) -> ZerothResponse:
        """
        Evaluate a weight update with Zeroth.

        Args:
            model_id: Unique identifier for the model
            delta_weights: Dictionary of weight parameter changes
            training_config: Training configuration used

        Returns:
            ZerothResponse with decision and risk score

        Raises:
            ZerothRejectError: If Zeroth denies the update
            ZerothTimeoutError: If request times out (fail-closed → REJECT)
            ZerothClientError: For other errors (fail-closed → REJECT)
        """
        # Compute hash of delta weights for efficient transfer
        delta_weights_hash = self._compute_weights_hash(delta_weights)

        payload = {
            "action": "weight_update",
            "model_id": model_id,
            "delta_weights_hash": delta_weights_hash,
            "training_config": training_config,
        }

        start_time = time.time()
        try:
            headers = {"Content-Type": "application/json"}
            if self.jwt_token:
                headers["Authorization"] = f"Bearer {self.jwt_token}"
            
            response = self._session.post(
                f"{self.base_url}/evaluate",
                json=payload,
                timeout=self.timeout_seconds,
                headers=headers,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Zeroth evaluation completed",
                extra={
                    "model_id": model_id,
                    "elapsed_ms": elapsed_ms,
                    "status_code": response.status_code,
                },
            )

            response.raise_for_status()
            result = response.json()

            return self._parse_response(result)

        except requests.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "Zeroth evaluation TIMEOUT - fail-closed REJECT",
                extra={
                    "model_id": model_id,
                    "timeout_ms": self.timeout_seconds * 1000,
                    "elapsed_ms": elapsed_ms,
                },
            )
            raise ZerothTimeoutError(f"Zeroth timeout after {self.timeout_seconds * 1000:.0f}ms - REJECT")

        except requests.RequestException as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "Zeroth evaluation ERROR - fail-closed REJECT",
                extra={
                    "model_id": model_id,
                    "error": str(e),
                    "elapsed_ms": elapsed_ms,
                },
            )
            raise ZerothClientError(f"Zeroth request failed: {e} - REJECT")

    def evaluate_or_raise(
        self,
        model_id: str,
        delta_weights: dict[str, Any],
        training_config: dict[str, Any],
    ) -> None:
        """
        Evaluate and raise exception if denied.

        Convenience method for trainer integration.
        Raises ZerothRejectError if decision is DENY.
        """
        response = self.evaluate_weight_update(
            model_id=model_id,
            delta_weights=delta_weights,
            training_config=training_config,
        )

        if not response.allowed:
            raise ZerothRejectError(
                reason=response.reason,
                risk_score=response.risk_score,
            )

    def _compute_weights_hash(self, delta_weights: dict[str, Any]) -> str:
        """Compute SHA-256 hash of delta weights."""
        # Serialize weights deterministically
        weights_json = json.dumps(delta_weights, sort_keys=True, default=str)
        return hashlib.sha256(weights_json.encode()).hexdigest()[:32]

    def _parse_response(self, result: dict[str, Any]) -> ZerothResponse:
        """Parse Zeroth response into ZerothResponse."""
        decision_str = result.get("decision", "deny").lower()
        decision = Decision.ALLOW if decision_str == "allow" else Decision.DENY

        return ZerothResponse(
            decision=decision,
            risk_score=float(result.get("risk_score", 1.0)),
            reason=result.get("reason", "No reason provided"),
        )


def create_zeroth_client(
    base_url: str | None = None,
    timeout_ms: int | None = None,
) -> ZerothClient:
    """Factory function to create ZerothClient from config."""
    import os

    base_url = base_url or os.getenv("ZEROTH_URL", ZerothClient.DEFAULT_BASE_URL)
    timeout_ms = timeout_ms or int(os.getenv("ZEROTH_TIMEOUT_MS", "100"))

    return ZerothClient(base_url=base_url, timeout_ms=timeout_ms)
