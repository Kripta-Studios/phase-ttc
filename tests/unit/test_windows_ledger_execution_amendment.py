"""Zero-optimizer retry equivalence and fail-closed scope checks."""

import ctypes
import runpy
import sys
import threading
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json

guarded_publication = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/resume_stage70_76_windows_publication.py")
)["guarded_publication"]


def sharing_error():
    error = PermissionError("synthetic Windows sharing failure")
    error.winerror = 5
    return error


@pytest.mark.skipif(sys.platform != "win32", reason="actual Windows sharing contract")
def test_real_windows_handle_conflict_recovers_identical_payload(tmp_path):
    target = tmp_path / "UPDATE_LEDGER.json"
    reference = tmp_path / "REFERENCE.json"
    atomic_json(target, {"update": 0})
    value = {"update": 3500}
    atomic_json(reference, value)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    # GENERIC_READ; share read/write but deliberately withhold FILE_SHARE_DELETE.
    handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    timer = threading.Timer(0.3, lambda: kernel.CloseHandle(handle))
    journal = []
    timer.start()
    try:
        guarded_publication(atomic_json, target, journal.append)(target, value)
    finally:
        timer.join()
    assert journal
    assert target.read_bytes() == reference.read_bytes()


def test_retried_publication_matches_continuous_bytes_without_repeating_optimizer(tmp_path):
    target = tmp_path / "UPDATE_LEDGER.json"
    reference = tmp_path / "REFERENCE.json"
    value = {"update": 3500, "pending": None, "nested": [1, 2, 3]}
    atomic_json(reference, value)
    calls, journal, delays = [], [], []

    def original(path, payload):
        calls.append(payload)
        if len(calls) <= 2:
            raise sharing_error()
        atomic_json(path, payload)

    guarded_publication(original, target, journal.append, pause=delays.append)(target, value)
    assert target.read_bytes() == reference.read_bytes()
    assert all(payload is value for payload in calls)
    assert delays == [0.1, 0.1]
    assert len(journal) == 2
    assert all(not record["optimizer_invoked_by_retry"] for record in journal)


@pytest.mark.parametrize(
    "other_path,windows,attempts", [(True, True, 1), (False, False, 1), (False, True, 101)]
)
def test_scope_unrelated_errors_and_retry_budget_fail_closed(
    tmp_path, other_path, windows, attempts
):
    ledger = tmp_path / "UPDATE_LEDGER.json"
    target = tmp_path / "other.json" if other_path else ledger
    calls = []

    def original(path, payload):
        calls.append(path)
        raise sharing_error() if windows else PermissionError("not a Windows sharing failure")

    publish = guarded_publication(original, ledger, lambda row: None, pause=lambda seconds: None)
    with pytest.raises(PermissionError):
        publish(target, {"update": 3500})
    assert len(calls) == attempts
