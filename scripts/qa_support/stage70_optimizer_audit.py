"""Cross-process bounded optimizer accounting for CPU pytest, without project imports.

The parent reserves the entire allowance in the campaign ledger before creating
this database. SQLite serializes intents from inherited Python subprocesses. An
intent without its post-hook is conservatively charged as uncertain work.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
import weakref
from pathlib import Path
from typing import Any

import psutil


def initialize(path: Path, limit: int, identity: str) -> None:
    """Create a fresh quota only after the caller's durable campaign reservation."""
    if path.exists() or limit < 1 or len(identity) != 64:
        raise ValueError("fresh audit database, positive quota and SHA256 required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE quota (limit_updates INTEGER, identity TEXT);"
            "CREATE TABLE processes (token TEXT PRIMARY KEY, pid INTEGER, started REAL);"
            "CREATE TABLE steps (id INTEGER PRIMARY KEY, optimizer TEXT, process TEXT, "
            "test_id TEXT, confirmed INTEGER DEFAULT 0);"
        )
        connection.execute("INSERT INTO quota VALUES (?, ?)", (limit, identity))


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=60)
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def snapshot(path: Path) -> dict[str, Any]:
    """Read exact attempts and optimizer instances, without claiming model endpoints."""
    with _connect(path) as connection:
        limit, identity = connection.execute("SELECT * FROM quota").fetchone()
        charged, confirmed = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(confirmed), 0) FROM steps"
        ).fetchone()
        fits = connection.execute("SELECT COUNT(DISTINCT optimizer) FROM steps").fetchone()[0]
        details = [
            {"optimizer": row[0], "attempts": row[1], "confirmed": row[2]}
            for row in connection.execute(
                "SELECT optimizer, COUNT(*), SUM(confirmed) FROM steps GROUP BY optimizer"
            )
        ]
    return {
        "identity": identity,
        "limit": limit,
        "charged": charged,
        "confirmed": confirmed,
        "uncertain": charged - confirmed,
        "optimizer_instances_started": fits,
        "optimizer_instances": details,
    }


class Recorder:
    """Publish intents before work and completion only after a successful CPU step."""

    def __init__(self, path: Path, identity: str) -> None:
        self.path = path.resolve()
        self.process = uuid.uuid4().hex
        self.optimizers: weakref.WeakKeyDictionary[object, str] = weakref.WeakKeyDictionary()
        self.pending: dict[int, int] = {}
        with _connect(self.path) as connection:
            if connection.execute("SELECT identity FROM quota").fetchone()[0] != identity:
                raise ValueError("QA accounting identity mismatch")
            connection.execute(
                "INSERT INTO processes VALUES (?, ?, ?)",
                (self.process, os.getpid(), psutil.Process().create_time()),
            )

    def before(self, optimizer: object) -> None:
        """Charge under an exclusive transaction, including across subprocesses."""
        key = id(optimizer)
        if key in self.pending:
            raise RuntimeError("nested step on the same optimizer is not auditable")
        if optimizer not in self.optimizers:
            self.optimizers[optimizer] = uuid.uuid4().hex
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            limit = connection.execute("SELECT limit_updates FROM quota").fetchone()[0]
            count = connection.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
            if count >= limit:
                raise RuntimeError("QA_UPDATE_BUDGET_BLOCKED: reserved allowance exhausted")
            cursor = connection.execute(
                "INSERT INTO steps (optimizer, process, test_id) VALUES (?, ?, ?)",
                (
                    self.optimizers[optimizer],
                    self.process,
                    os.environ.get("PYTEST_CURRENT_TEST", "outside_pytest_call"),
                ),
            )
            assert cursor.lastrowid is not None
            self.pending[key] = cursor.lastrowid

    def after(self, optimizer: object) -> None:
        """Confirm only the matching charged attempt; exceptions leave uncertain work."""
        key = id(optimizer)
        attempt = self.pending[key]
        with _connect(self.path) as connection:
            cursor = connection.execute(
                "UPDATE steps SET confirmed=1 WHERE id=? AND process=? AND confirmed=0",
                (attempt, self.process),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("missing or duplicated optimizer confirmation")
        del self.pending[key]


def install() -> None:
    """Install public Torch hooks for this explicitly enabled CPU-only QA process."""
    import torch
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_available():
        raise RuntimeError("QA optimizer accounting requires the registered CPU-only runtime")
    recorder = Recorder(
        Path(os.environ["STAGE70_QA_OPTIMIZER_DB"]), os.environ["STAGE70_QA_IDENTITY"]
    )
    register_optimizer_step_pre_hook(lambda optimizer, _args, _kwargs: recorder.before(optimizer))
    register_optimizer_step_post_hook(lambda optimizer, _args, _kwargs: recorder.after(optimizer))
