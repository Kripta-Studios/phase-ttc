"""Exact test continuation cannot drop failures or accept a changed source inventory."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest

SPEC = importlib.util.spec_from_file_location(
    "qa_capture_fixture",
    Path(__file__).resolve().parents[2] / "scripts/qa_support/stage70_capture.py",
)
assert SPEC is not None and SPEC.loader is not None
PLUGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLUGIN)


def test_signed_pending_selection_preserves_order_and_rejects_changed_inventory(
    tmp_path, monkeypatch
):
    payload = {"full_nodeids": ["a", "b", "c"], "pending_ids": ["c", "b"]}
    signature = object_digest(payload)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({**payload, "sha256": signature}))
    monkeypatch.setenv("STAGE70_QA_PENDING_SELECTION", str(path))
    monkeypatch.setenv("STAGE70_QA_PENDING_SHA256", signature)
    removed = []
    config = SimpleNamespace(
        hook=SimpleNamespace(pytest_deselected=lambda items: removed.extend(items))
    )
    items = [SimpleNamespace(nodeid=name) for name in ("a", "b", "c")]
    PLUGIN.pytest_collection_modifyitems(config, items)
    assert [item.nodeid for item in items] == ["b", "c"]
    assert [item.nodeid for item in removed] == ["a"]
    with pytest.raises(pytest.UsageError, match="inventory changed"):
        PLUGIN.pytest_collection_modifyitems(config, items)
    monkeypatch.setenv("STAGE70_QA_PENDING_SHA256", "a" * 64)
    with pytest.raises(pytest.UsageError, match="signature"):
        PLUGIN.pytest_collection_modifyitems(config, items)
