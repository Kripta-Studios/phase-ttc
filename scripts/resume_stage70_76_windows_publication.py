"""Visible execution amendment: bounded retries for Windows ledger publication only."""

# ruff: noqa: E402 -- same repository bootstrap as the frozen launcher
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts import risk_geometry_v10 as publication

SCOPE = "WINDOWS_LEDGER_PUBLICATION_RETRY_ONLY"


def guarded_publication(
    original: Callable[[Path, object], None],
    ledger: Path,
    journal: Callable[[dict[str, Any]], None],
    *,
    pause: Callable[[float], None] = time.sleep,
) -> Callable[[Path, object], None]:
    """Retry identical JSON publication, never an optimizer action or another path."""

    def publish(path: Path, value: object) -> None:
        if path.absolute() != ledger:
            original(path, value)
            return
        for attempt in range(101):
            try:
                original(path, value)
                return
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 100:
                    raise
                journal(
                    {
                        "kind": SCOPE,
                        "path": str(path),
                        "retry": attempt + 1,
                        "winerror": error.winerror,
                        "pid": os.getpid(),
                        "time_ns": time.time_ns(),
                        "optimizer_invoked_by_retry": False,
                    }
                )
                pause(0.1)

    return publish


def main() -> None:
    """Validate the signed adapter amendment, then enter the unchanged resume launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if "--resume" not in forwarded or "--execute-authorized" not in forwarded:
        parser.error("this adapter only supports the existing authorized full-graph resume")
    root = ROOT / "artifacts/stage70_76_architecture"
    amendment_path = args.amendment.resolve(strict=True)
    if amendment_path.parent != root / "execution_amendments":
        raise ValueError("amendment escaped the campaign's explicit amendment directory")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    signature = amendment.pop("artifact_sha256", None)
    if (
        publication.object_digest(amendment) != signature
        or amendment.get("scope") != SCOPE
        or amendment.get("retry_policy") != {"maximum_retries": 100, "delay_seconds": 0.1}
        or amendment.get("scientific_recipe_changed") is not False
        or amendment.get("adapter") != publication.binding(Path(__file__))
        or publication.verify(amendment["training_lock"]) != root / "TRAINING_LOCK.json"
    ):
        raise ValueError("execution amendment, adapter or original scientific lock changed")
    for source in amendment["tests"]:
        publication.verify(source)
    for source in amendment["incidents"]:
        publication.verify(source)
    journal_path = root / "execution_amendments" / f"PUBLICATION_RETRIES_{os.getpid()}.jsonl"

    def journal(value: dict[str, Any]) -> None:
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    # Install before importing the frozen trainer/ledger so their imported function
    # references use this explicitly declared execution adapter. Every original
    # file hash and all checkpoint/model/optimizer identities remain validated.
    publication.atomic_json = guarded_publication(
        publication.atomic_json, root / "UPDATE_LEDGER.json", journal
    )
    launcher = ROOT / "scripts/run_scientific_recovery_v10_stage70_76.py"
    print(
        f"Execution amendment {signature}: bounded Windows ledger publication retries", flush=True
    )
    sys.argv = [str(launcher), *forwarded]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
