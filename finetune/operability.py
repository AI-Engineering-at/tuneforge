"""
finetune/operability.py
=======================
Enterprise-grade operability layer for TuneForge.

Provides:
  - Formal Degraded-State Taxonomy (state machine)
  - Structured JSON logging (OpenTelemetry-compatible)
  - Seraph Aegis auth-gated API calls
  - Metric emission helpers

State Hierarchy (ordered by severity):
  OPERATIONAL          All systems nominal.
  DEGRADED_VRAM        CUDA OOM handled; training proceeds in reduced mode.
  DEGRADED_NAN         NaN gradients detected; affected layers skipped.
  DEGRADED_AEGIS       Cannot reach Seraph Aegis; running in offline mode
                       (only if ZEROTH_MOCK_MODE=1, else HALTED).
  HALTED_CORE_FAULT    Unrecoverable internal fault. Training aborted.
  HALTED_AEGIS_DENY    Seraph Aegis denied the request. Fail-closed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional

import requests


# ---------------------------------------------------------------------------
# 1.  Degraded-State Taxonomy
# ---------------------------------------------------------------------------

class TrainingState(str, Enum):
    OPERATIONAL        = "OPERATIONAL"
    DEGRADED_VRAM      = "DEGRADED_VRAM"
    DEGRADED_NAN       = "DEGRADED_NAN"
    DEGRADED_AEGIS     = "DEGRADED_AEGIS"
    HALTED_CORE_FAULT  = "HALTED_CORE_FAULT"
    HALTED_AEGIS_DENY  = "HALTED_AEGIS_DENY"

    @property
    def is_operational(self) -> bool:
        return self == TrainingState.OPERATIONAL

    @property
    def is_degraded(self) -> bool:
        return self.value.startswith("DEGRADED_")

    @property
    def is_halted(self) -> bool:
        return self.value.startswith("HALTED_")

    @property
    def can_continue_training(self) -> bool:
        """Training may proceed only in these states."""
        return self in (
            TrainingState.OPERATIONAL,
            TrainingState.DEGRADED_VRAM,
            TrainingState.DEGRADED_NAN,
        )

    @property
    def description(self) -> str:
        return {
            TrainingState.OPERATIONAL:        "All systems nominal.",
            TrainingState.DEGRADED_VRAM:      "CUDA OOM recovered; safety micro-batching active.",
            TrainingState.DEGRADED_NAN:       "NaN gradients detected; affected layers quarantined.",
            TrainingState.DEGRADED_AEGIS:     "Seraph Aegis unreachable; offline mock-mode only.",
            TrainingState.HALTED_CORE_FAULT:  "HALTED: Unrecoverable internal fault. Restart required.",
            TrainingState.HALTED_AEGIS_DENY:  "HALTED: Seraph Aegis denied this job. Fail-closed.",
        }[self]


class StateMachine:
    """Thread-safe state machine for TuneForge training jobs."""

    # Valid transitions.  Key → set of allowed successor states.
    _TRANSITIONS: Dict[TrainingState, set] = {
        TrainingState.OPERATIONAL:       {TrainingState.DEGRADED_VRAM,
                                          TrainingState.DEGRADED_NAN,
                                          TrainingState.DEGRADED_AEGIS,
                                          TrainingState.HALTED_CORE_FAULT,
                                          TrainingState.HALTED_AEGIS_DENY},
        TrainingState.DEGRADED_VRAM:     {TrainingState.OPERATIONAL,
                                          TrainingState.DEGRADED_NAN,
                                          TrainingState.HALTED_CORE_FAULT},
        TrainingState.DEGRADED_NAN:      {TrainingState.OPERATIONAL,
                                          TrainingState.HALTED_CORE_FAULT},
        TrainingState.DEGRADED_AEGIS:    {TrainingState.OPERATIONAL,
                                          TrainingState.HALTED_CORE_FAULT},
        # HALTED states are terminal – no recovery without operator action.
        TrainingState.HALTED_CORE_FAULT: set(),
        TrainingState.HALTED_AEGIS_DENY: set(),
    }

    def __init__(self, node_id: str, job_id: str):
        self.node_id = node_id
        self.job_id  = job_id
        self._state  = TrainingState.OPERATIONAL
        self._lock   = Lock()
        self._log    = StructuredLogger("tuneforge.state_machine")

    @property
    def state(self) -> TrainingState:
        return self._state

    def transition(self, new_state: TrainingState, reason: str) -> None:
        with self._lock:
            allowed = self._TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                raise RuntimeError(
                    f"Illegal state transition {self._state} → {new_state}. "
                    f"Allowed: {allowed}"
                )
            old = self._state
            self._state = new_state

            level = "info" if new_state.is_operational else \
                    "warning" if new_state.is_degraded else "error"
            self._log.emit(
                level=level,
                event="state_transition",
                from_state=old.value,
                to_state=new_state.value,
                reason=reason,
                node_id=self.node_id,
                job_id=self.job_id,
            )

    def assert_can_train(self) -> None:
        """Call before every training step. Raises if halted."""
        if not self._state.can_continue_training:
            raise RuntimeError(
                f"Training blocked in state {self._state.value}: "
                f"{self._state.description}"
            )


# ---------------------------------------------------------------------------
# 2.  Structured Logger (OpenTelemetry-compatible JSON)
# ---------------------------------------------------------------------------

class StructuredLogger:
    """
    Emits structured JSON log lines to stderr / log aggregator.
    Every record includes: timestamp, level, event, service, trace_id,
    and any caller-supplied fields.

    Compatible with ELK, Loki, and any OpenTelemetry log ingester.
    """

    def __init__(self, name: str):
        self._name  = name
        self._inner = logging.getLogger(name)

    def emit(self, level: str, event: str, **fields: Any) -> None:
        record = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level":      level.upper(),
            "service":    "tuneforge",
            "logger":     self._name,
            "event":      event,
            **fields,
        }
        line = json.dumps(record, default=str)
        getattr(self._inner, level.lower(), self._inner.info)(line)

    # Convenience aliases
    def info(self, event: str, **kw):  self.emit("info",    event, **kw)
    def warn(self, event: str, **kw):  self.emit("warning", event, **kw)
    def error(self, event: str, **kw): self.emit("error",   event, **kw)
    def debug(self, event: str, **kw): self.emit("debug",   event, **kw)


# ---------------------------------------------------------------------------
# 3.  Seraph Aegis API Client
# ---------------------------------------------------------------------------

class AegisClient:
    """
    Authenticated client for the Seraph Aegis Rust microservice.

    Auth-Default: every request carries a JWT Bearer token.
    Fail-Closed:  if Aegis denies or is unreachable and ZEROTH_MOCK_MODE
                  is not set, the result is HALTED.
    """

    def __init__(self):
        self._url         = os.environ.get("AEGIS_API_URL", "http://localhost:8741")
        self._token       = os.environ.get("AEGIS_JWT_TOKEN", "")
        self._mock_mode   = os.environ.get("ZEROTH_MOCK_MODE", "0") == "1"
        self._log         = StructuredLogger("tuneforge.aegis_client")
        self._timeout_sec = int(os.environ.get("AEGIS_TIMEOUT_SEC", "5"))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  "application/json",
            "X-Node-ID":     os.environ.get("TUNEFORGE_NODE_ID", "unknown"),
        }

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._url}{path}"
        resp = requests.post(url, json=payload, headers=self._headers(),
                             timeout=self._timeout_sec)
        resp.raise_for_status()
        return resp.json()

    def evaluate_policy(
        self,
        job_id:       str,
        tags:         list[str],
        phase:        str,          # "pre_train" | "pre_publish"
        dataset_hash: str,
        node_id:      Optional[str] = None,
    ) -> dict:
        """
        Calls POST /v1/policy/evaluate on Seraph Aegis.
        Returns the PolicyResponse dict.
        Raises RuntimeError on denial.
        """
        if self._mock_mode:
            self._log.warn(
                "aegis_mock_mode",
                job_id=job_id, phase=phase,
                message="ZEROTH_MOCK_MODE=1: Bypassing real Aegis check."
            )
            return {"verdict": "ALLOW", "reason": "mock_mode", "clearance_token": "mock"}

        payload = {
            "node_id":      node_id or os.environ.get("TUNEFORGE_NODE_ID", "unknown"),
            "job_id":       job_id,
            "tags":         tags,
            "phase":        phase,
            "dataset_hash": dataset_hash,
        }

        try:
            result = self._post("/v1/policy/evaluate", payload)
        except requests.RequestException as exc:
            self._log.error(
                "aegis_unreachable",
                job_id=job_id, phase=phase, error=str(exc)
            )
            raise ConnectionError(f"[Seraph Aegis] Unreachable: {exc}") from exc

        verdict = result.get("verdict", "DENY")
        self._log.emit(
            level="info" if verdict == "ALLOW" else "error",
            event="aegis_policy_result",
            verdict=verdict,
            rule=result.get("triggered_rule"),
            reason=result.get("reason"),
            job_id=job_id,
            phase=phase,
        )

        if verdict != "ALLOW":
            raise PermissionError(
                f"[Seraph Aegis] Policy DENIED ({phase}): {result.get('reason')}"
            )

        return result


# ---------------------------------------------------------------------------
# 4.  Metric helpers (Prometheus-compatible text format)
# ---------------------------------------------------------------------------

class TrainingMetrics:
    """
    Simple in-process counter/gauge store.
    Dump with `.render()` for /metrics endpoint or append to JSONL.
    """

    def __init__(self):
        self._lock    = Lock()
        self._counters: Dict[str, float] = {}
        self._gauges:   Dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges":   dict(self._gauges),
                "timestamp": time.time(),
            }

    def render(self) -> str:
        """Prometheus text format."""
        lines = []
        snap = self.snapshot()
        for k, v in snap["counters"].items():
            lines.append(f"tuneforge_{k}_total {v}")
        for k, v in snap["gauges"].items():
            lines.append(f"tuneforge_{k} {v}")
        return "\n".join(lines)

    def emit_jsonl(self, path: str) -> None:
        """Append current snapshot to a JSONL metrics file."""
        snap = self.snapshot()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snap) + "\n")
        except OSError as exc:
            logging.getLogger("tuneforge.metrics").error(
                "Failed to write metrics JSONL: %s", exc
            )
