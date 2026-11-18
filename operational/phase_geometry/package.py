"""GarlTTC submission package format and label-free output-disagreement summaries (numpy)."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import DT, row_mid

FORMAT = "garlttc_prediction_v1"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
DISAGREEMENT_BANDS = (
    ("pred_c (0,3]", 0.0, 3.0),
    ("pred_s (3,6]", 3.0, 6.0),
    ("pred_l (6,10]", 6.0, 10.0),
    ("pred_far (10,inf)", 10.0, np.inf),
    ("pred_n (-10,0]", -10.0, 0.0),
    ("pred_far_neg (-inf,-10]", -np.inf, -10.0),
)


def build_submission(tokens: Sequence[str], ttc: NDArray[Any]) -> dict[str, Any]:
    """Official JSON payload ``{meta:{format}, results:{token:{ttc}}}`` in the given order."""
    values = np.asarray(ttc, dtype=np.float64)
    if len(tokens) != len(values) or len(set(tokens)) != len(tokens):
        raise ValueError("Tokens must be unique and aligned with predictions")
    if not np.isfinite(values).all() or (np.abs(values) < DT).any():
        raise ValueError("Every prediction must be finite with |T| >= 0.1 s")
    return {
        "meta": {"format": FORMAT},
        "results": {str(t): {"ttc": float(v)} for t, v in zip(tokens, values, strict=True)},
    }


def write_zip(path: Path, submission: dict[str, Any]) -> bytes:
    """Write a byte-reproducible ZIP holding only a root ``submission.json``; return its payload."""
    payload = json.dumps(submission, separators=(",", ":"), allow_nan=False).encode("utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)
    return payload


def read_zip(path: Path) -> dict[str, Any]:
    """Read and structurally validate a package ZIP."""
    with zipfile.ZipFile(path, "r") as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("Package must contain exactly one root submission.json")
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    if payload.get("meta", {}).get("format") != FORMAT or not isinstance(
        payload.get("results"), dict
    ):
        raise ValueError("Package meta.format or results differ from the official format")
    return payload


def verify_package(path: Path, tokens: Sequence[str]) -> NDArray[np.float64]:
    """Check exact token order/coverage and failure-free predictions; return the TTC vector."""
    results = read_zip(path)["results"]
    if list(results) != list(tokens):
        raise ValueError("Package tokens differ from the official order")
    values = np.asarray([float(results[t]["ttc"]) for t in tokens])
    if not np.isfinite(values).all() or (np.abs(values) < DT).any():
        raise ValueError("Package contains a failed prediction")
    return values


def disagreement(candidate: NDArray[Any], reference: NDArray[Any]) -> dict[str, Any]:
    """Output disagreement between two prediction sets (NOT accuracy).

    MiD is computed with ``reference`` in the target slot only as a symmetric-in-spirit
    distance between predictions; bands are those of the reference prediction.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        raise ValueError("Prediction sets are misaligned")
    distance = row_mid(ref, cand)
    out: dict[str, Any] = {
        "rows": len(cand),
        "coverage_candidate": float(np.isfinite(cand).mean()),
        "coverage_reference": float(np.isfinite(ref).mean()),
        "failures_candidate": int((~np.isfinite(cand) | (np.abs(cand) < DT)).sum()),
        "failures_reference": int((~np.isfinite(ref) | (np.abs(ref) < DT)).sum()),
        "sign_changes": int((np.sign(cand) != np.sign(ref)).sum()),
        "sign_change_rate": float((np.sign(cand) != np.sign(ref)).mean()),
        "mid_between_mean": float(np.nanmean(distance)),
        "mid_between_median": float(np.nanmedian(distance)),
        "mid_between_p90": float(np.nanquantile(distance, 0.9)),
        "mid_between_undefined": int((~np.isfinite(distance)).sum()),
        "by_reference_band": [],
    }
    for label, low, high in DISAGREEMENT_BANDS:
        mask = (ref > low) & (ref <= high)
        out["by_reference_band"].append(
            {
                "band": label,
                "rows": int(mask.sum()),
                "mid_between_mean": float(np.nanmean(distance[mask])) if mask.any() else None,
                "mid_between_median": float(np.nanmedian(distance[mask])) if mask.any() else None,
                "sign_changes": int((np.sign(cand[mask]) != np.sign(ref[mask])).sum()),
            }
        )
    return out


__all__ = ["build_submission", "disagreement", "read_zip", "verify_package", "write_zip"]
