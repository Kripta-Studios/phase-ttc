"""Durable full-job optimizer reservations, including replayed and uncertain work."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify

MAX_UPDATES = 900_000
MAX_TECHNICAL_UPDATES = 3_000


@dataclass(frozen=True)
class FitSpec:
    """Prospective identity for a prescribed optimizer fit, never an outer-selected run."""

    job_id: str
    stage: str
    arm: str
    fold: int
    seed: int
    endpoint: int
    recipe_sha256: str
    training_view_sha256: str
    technical: bool = False

    def __post_init__(self) -> None:
        if (
            not self.job_id
            or not self.stage
            or not self.arm
            or self.fold < 0
            or self.seed < 0
            or not 1 <= self.endpoint <= MAX_UPDATES
            or any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in (self.recipe_sha256, self.training_view_sha256)
            )
        ):
            raise ValueError("invalid fixed fit specification")


class UpdateLedger:
    """One campaign owner; reserve every remaining prescribed update before a job.

    Before each optimizer step, an atomic intent charges its budget. Successful
    completion is then confirmed. A process death between these two publications
    leaves an explicitly uncertain attempt, never a fabricated completed update.
    Such attempts conservatively consume budget and are reported separately.
    Restarting from an earlier valid checkpoint charges all repeated work again.
    """

    def __init__(self, path: Path, owner_root: Path) -> None:
        self.path, self.owner_root = path, owner_root
        self._assert_owner()
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            signature = state.pop("sha256")
            if object_digest(state) != signature:
                raise ValueError("optimizer ledger signature mismatch")
            self.state: dict[str, Any] = state
        else:
            self.state = {"version": 1, "jobs": {}, "pending": None, "ridge_fits": []}
        self._validate_budget()

    def _assert_owner(self) -> None:
        record = json.loads((self.owner_root / "ACTIVE_OWNER.json").read_text())
        if record["pid"] != os.getpid() or record["created"] != psutil.Process().create_time():
            raise RuntimeError("optimizer ledger requires the current exclusive campaign owner")

    def totals(self) -> dict[str, int]:
        """Report confirmed work, uncertain charged attempts and future reservations."""
        jobs = list(self.state["jobs"].values())
        return {
            "confirmed_optimizer_updates": sum(job["confirmed"] for job in jobs),
            "uncertain_optimizer_attempts": sum(job["uncertain"] for job in jobs)
            + int(self.state["pending"] is not None),
            "charged_optimizer_attempts": sum(job["charged"] for job in jobs),
            "reserved_remaining_updates": sum(job["remaining"] for job in jobs),
            "technical_charged_and_reserved": sum(
                job["charged"] + job["remaining"] for job in jobs if job["spec"]["technical"]
            ),
            "optimizer_fits_started": sum(job["charged"] > 0 for job in jobs),
            "optimizer_endpoints_complete": sum(job["status"] == "COMPLETE" for job in jobs),
            "ridge_fits": len(self.state["ridge_fits"]),
        }

    def _validate_budget(self) -> None:
        totals = self.totals()
        if (
            totals["charged_optimizer_attempts"] + totals["reserved_remaining_updates"]
            > MAX_UPDATES
        ):
            raise RuntimeError("UPDATE_BUDGET_BLOCKED: entire remaining job exceeds900000")
        if totals["technical_charged_and_reserved"] > MAX_TECHNICAL_UPDATES:
            raise RuntimeError("UPDATE_BUDGET_BLOCKED: technical budget exceeds3000")
        for job in self.state["jobs"].values():
            if (
                min(job[k] for k in ("charged", "confirmed", "uncertain", "remaining", "update"))
                < 0
            ):
                raise ValueError("negative optimizer ledger accounting")
            if job["confirmed"] + job["uncertain"] > job["charged"]:
                raise ValueError("optimizer accounting exceeds charged attempts")

    def _publish(self) -> None:
        self._assert_owner()
        self._validate_budget()
        atomic_json(self.path, {**self.state, "sha256": object_digest(self.state)})

    def reserve(self, spec: FitSpec) -> None:
        """Atomically bind the complete remaining fit before its first optimizer step."""
        jobs = self.state["jobs"]
        if spec.job_id in jobs:
            if jobs[spec.job_id]["spec"] != asdict(spec):
                raise ValueError("existing fit recipe or endpoint changed")
            return
        jobs[spec.job_id] = {
            "spec": asdict(spec),
            "status": "RESERVED",
            "update": 0,
            "remaining": spec.endpoint,
            "charged": 0,
            "confirmed": 0,
            "uncertain": 0,
            "checkpoint": None,
            "checkpoint_update": 0,
            "endpoint": None,
        }
        try:
            self._publish()
        except BaseException:
            jobs.pop(spec.job_id)
            raise

    def begin_step(self, job_id: str) -> int:
        """Publish an intent before optimizer.step; return the intended logical update."""
        job = self.state["jobs"][job_id]
        if self.state["pending"] is not None or job["remaining"] <= 0:
            raise ValueError("another step is pending or the fixed endpoint was reached")
        if job["status"] not in ("RESERVED", "RUNNING"):
            raise ValueError("fit is not in a trainable state")
        job["charged"] += 1
        job["remaining"] -= 1
        job["status"] = "RUNNING"
        self.state["pending"] = {"job_id": job_id, "update": job["update"] + 1}
        self._publish()
        return job["update"] + 1

    def confirm_step(self, job_id: str) -> None:
        """Call only after optimizer work and required device synchronization succeed."""
        pending = self.state["pending"]
        if pending is None or pending["job_id"] != job_id:
            raise ValueError("no matching optimizer intent")
        job = self.state["jobs"][job_id]
        job["confirmed"] += 1
        job["update"] = pending["update"]
        self.state["pending"] = None
        self._publish()

    def checkpoint(self, job_id: str, path: Path, update: int) -> None:
        """Bind a flushed immutable full-state checkpoint to its exact logical update."""
        job = self.state["jobs"][job_id]
        if self.state["pending"] is not None or update != job["update"]:
            raise ValueError("checkpoint is not the latest confirmed logical state")
        record = binding(path)
        job["checkpoint"], job["checkpoint_update"] = record, update
        self._publish()

    def recover(self, job_id: str) -> dict[str, Any] | None:
        """Recover only the latest bound checkpoint and reserve all replay work again."""
        job = self.state["jobs"][job_id]
        if job["status"] == "COMPLETE":
            raise ValueError("completed endpoint cannot restart")
        pending = self.state["pending"]
        if pending is not None and pending["job_id"] != job_id:
            raise ValueError("another fit owns the unresolved optimizer attempt")
        if job["checkpoint"] is not None:
            verify(job["checkpoint"])
        old_state = json.loads(json.dumps(self.state))
        if pending is not None:
            job["uncertain"] += 1
            self.state["pending"] = None
        job["update"] = job["checkpoint_update"]
        job["remaining"] = job["spec"]["endpoint"] - job["update"]
        try:
            self._publish()
        except BaseException:
            self.state = old_state
            raise
        return job["checkpoint"]

    def complete(self, job_id: str) -> None:
        """Certify only the prescribed final update and its verified full-state payload."""
        job = self.state["jobs"][job_id]
        endpoint = job["spec"]["endpoint"]
        if (
            self.state["pending"] is not None
            or job["update"] != endpoint
            or job["checkpoint_update"] != endpoint
            or job["checkpoint"] is None
            or job["remaining"] != 0
        ):
            raise ValueError("partial checkpoint cannot become a completed endpoint")
        verify(job["checkpoint"])
        job["status"], job["endpoint"] = "COMPLETE", job["checkpoint"]
        self._publish()
