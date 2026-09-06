"""Integrity checks for the bounded Stage70 audit, not trainer integration QA."""

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.campaign_v11 import ResourcePolicy, ResourceReservation
from scripts.audit_stage70_76_preflight import checked_members, phase, resource_record


@pytest.mark.parametrize("name", ["../bad", "/bad", "C:/bad", "a\\bad", "a/../bad"])
def test_rejects_malicious_members(name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, b"x")
        archive.writestr("ESSENTIAL_MANIFEST.json", "{}")
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError):
        checked_members(archive)


def test_verifies_exact_manifest():
    payload = b"safe inference evidence"
    manifest = {"rows.csv": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("rows.csv", payload)
        archive.writestr("ESSENTIAL_MANIFEST.json", json.dumps(manifest))
    with zipfile.ZipFile(stream) as archive:
        assert checked_members(archive)["verified_entries"] == 1


def test_rejects_hash_mismatch():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("rows.csv", b"x")
        archive.writestr(
            "ESSENTIAL_MANIFEST.json", json.dumps({"rows.csv": {"bytes": 1, "sha256": "bad"}})
        )
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match="identity mismatch"):
        checked_members(archive)


def test_reservation_requires_margin_after_allocation(monkeypatch):
    minimum = 40_000_000_000
    assert ResourcePolicy().disk_free_bytes == minimum
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(10**15, 0, minimum + 10)
    )
    assert resource_record(Path.cwd(), 10)["passes"]
    assert not resource_record(Path.cwd(), 11)["passes"]
    with pytest.raises(ValueError, match="negative reservation"):
        resource_record(Path.cwd(), -1)


def test_host_memory_minimum_is_fixed_and_inclusive():
    minimum = ResourcePolicy().host_ram_available_bytes
    assert minimum == 4 * 1024**3
    assert ResourceReservation("host_RAM", minimum + 1, 1, minimum).passes
    assert not ResourceReservation("host_RAM", minimum, 1, minimum).passes


def test_reservation_rejects_invalid_quantities():
    with pytest.raises(ValueError, match="negative"):
        ResourceReservation("disk", 100, -1, 10)
    with pytest.raises(ValueError, match="negative"):
        ResourcePolicy(disk_free_bytes=-1)


def test_signed_phase_domain():
    np.testing.assert_allclose(phase(np.array([-0.1, 0.2])), [-np.log(2), np.log(2)])
    for invalid in (0.0, 0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="invalid TTC"):
            phase(np.array([invalid]))
