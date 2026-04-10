"""
finetune/zeroth_core.py
========================
Zeroth-Law-Core Seraph Aegis integration seam for TuneForge.

This module is the single choke-point between TuneForge and the
Seraph Aegis Rust microservice.  ALL traffic goes through here.

Fail-Closed by design:
  - If Aegis is unreachable AND mock mode is off → raise, block job.
  - If Aegis explicitly denies → raise, block job, transition to HALTED.
  - If Aegis allows → return clearance token for audit trail.

State transitions are propagated back to the caller via the
`StateMachine` in `operability.py`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from finetune.operability import AegisClient, StructuredLogger, TrainingState

logger = StructuredLogger("tuneforge.zeroth_core")


class ZerothAccessDeniedError(Exception):
    """Raised when Seraph Aegis blocks an action (policy DENY or fail-closed)."""


class ZerothConnectionError(Exception):
    """Raised when Seraph Aegis is unreachable and mock mode is disabled."""


# ---------------------------------------------------------------------------
# Dataset hashing helper
# ---------------------------------------------------------------------------


def _hash_dataset(records: List[Dict[str, Any]]) -> str:
    """
    Produce a deterministic SHA-256 fingerprint of a dataset.
    Used as the `dataset_hash` field in policy requests so Aegis can
    cross-reference against its restricted-dataset blocklist.
    """
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core gate functions
# ---------------------------------------------------------------------------

_client = AegisClient()


def pre_train_zeroth_check(
    config: dict,
    dataset_records: List[Dict[str, Any]],
    job_id: str,
    state_machine=None,  # Optional[StateMachine]  – avoids circular import
) -> str:
    """
    Must be called BEFORE any optimizer step is run.

    Parameters
    ----------
    config          : Training config dict (passed for tag extraction).
    dataset_records : The full training dataset (used for hash generation).
    job_id          : Unique job identifier for the audit trail.
    state_machine   : If provided, transitions to HALTED_AEGIS_DENY on denial.

    Returns
    -------
    clearance_token : Opaque token from Aegis for the audit log.

    Raises
    ------
    ZerothAccessDeniedError  on policy DENY or fail-closed.
    ZerothConnectionError    if Aegis is unreachable and mock mode is off.
    """
    logger.info("pre_train_check", job_id=job_id, phase="pre_train")

    tags = _extract_tags(config)
    dataset_hash = _hash_dataset(dataset_records)

    try:
        result = _client.evaluate_policy(
            job_id=job_id,
            tags=tags,
            phase="pre_train",
            dataset_hash=dataset_hash,
        )
    except ConnectionError as exc:
        logger.error("aegis_connection_failed", job_id=job_id, error=str(exc))
        if state_machine:
            state_machine.transition(TrainingState.DEGRADED_AEGIS, str(exc))
        raise ZerothConnectionError(str(exc)) from exc
    except PermissionError as exc:
        logger.error("aegis_denied", job_id=job_id, error=str(exc))
        if state_machine:
            state_machine.transition(TrainingState.HALTED_AEGIS_DENY, str(exc))
        raise ZerothAccessDeniedError(str(exc)) from exc

    token = result.get("clearance_token", "")
    logger.info("pre_train_cleared", job_id=job_id, clearance_token=token)
    return token


def pre_publish_zeroth_check(
    model_card: dict,
    manifest: dict,
    job_id: str,
    state_machine=None,
) -> str:
    """
    Must be called BEFORE any model artifact is saved or exported.

    Raises ZerothAccessDeniedError on denial (fail-closed).
    """
    logger.info("pre_publish_check", job_id=job_id, phase="pre_publish")

    tags = _extract_tags({**model_card, **manifest})
    dataset_hash = manifest.get("dataset_hash", "unknown")

    try:
        result = _client.evaluate_policy(
            job_id=job_id,
            tags=tags,
            phase="pre_publish",
            dataset_hash=dataset_hash,
        )
    except ConnectionError as exc:
        logger.error("aegis_connection_failed_publish", job_id=job_id, error=str(exc))
        if state_machine:
            state_machine.transition(TrainingState.HALTED_AEGIS_DENY, str(exc))
        raise ZerothConnectionError(str(exc)) from exc
    except PermissionError as exc:
        logger.error("aegis_denied_publish", job_id=job_id, error=str(exc))
        if state_machine:
            state_machine.transition(TrainingState.HALTED_AEGIS_DENY, str(exc))
        raise ZerothAccessDeniedError(str(exc)) from exc

    token = result.get("clearance_token", "")
    logger.info("pre_publish_cleared", job_id=job_id, clearance_token=token)
    return token


# ---------------------------------------------------------------------------
# Tag extraction helper
# ---------------------------------------------------------------------------


def _extract_tags(config: dict) -> List[str]:
    """
    Extract Seraph Aegis policy tags from a config / metadata dict.
    Tags drive the ALLOW / DENY evaluation in the Rust rule engine.
    """
    raw_tags: List[str] = list(config.get("tags", []))

    # Implicit tag derivation from known config fields
    domain = config.get("domain", "")
    if domain:
        raw_tags.append(domain.lower())

    dataset_name = config.get("dataset", config.get("dataset_name", ""))
    if "restricted" in dataset_name.lower():
        raw_tags.append("restricted_dataset")

    return list(set(raw_tags))  # deduplicate
