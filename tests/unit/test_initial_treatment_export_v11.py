"""Numeric outer fold assembly checks use temporary synthetic arrays only."""

import copy

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.evaluation.initial_treatment_export_v11 import (
    assemble_fold_arrays,
    publish_export_payloads,
)
from e_jepa_ttc.evaluation.standalone_outer_v11 import _publish_arrays


def test_physical_metadata_uses_bound_event_clock_and_preserves_empty_roi(tmp_path):
    """A token suffix must never become time; zero-event rows retain their mass."""
    import json

    from e_jepa_ttc.artifacts.risk_geometry_v10 import AllowlistedSources, binding
    from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import DiagnosticExport
    from e_jepa_ttc.evaluation.initial_treatment_export_v11 import append_physical_metadata

    token = "sequence_999999999999999"
    windows = []
    for index in range(2):
        windows.append(
            {
                "sample_token": token,
                "window_id": index,
                "sequence_id": "seq",
                "window_start_us": 1000000 + index * 100000,
                "window_end_us": 1100000 + index * 100000,
                "frame_to_event_clock_offset_us": 7000000,
                "target_anchor_us": 8200000,
                "roi_x0": 0,
                "roi_y0": 0,
                "roi_x1": 32,
                "roi_y1": 32,
                "h5_file_sha256": "a" * 64,
            }
        )
    raw = tmp_path / "RAW_BINDINGS.csv"
    pd.DataFrame(windows).to_csv(raw, index=False)
    observations = tmp_path / "diagnostics_00000.json"
    observations.write_text(
        json.dumps([{"sample_token": token, "windows": [{"roi_events": 0}, {"roi_events": 0}]}])
    )
    source = binding(observations)
    source["path"] = observations.name
    manifest = tmp_path / "CACHE_MANIFEST.json"
    manifest.write_text(json.dumps({"shards": [{"components": {observations.name: source}}]}))
    lock = {
        "raw_deployment": {"bindings": binding(raw)},
        "count_caches": {"original": binding(manifest)},
    }
    receipt = {
        "identity": {
            "ordered_tokens": [token],
            "count_cache_sha256": "b" * 64,
            "job_sha256": "c" * 64,
            "weights": {"sha256": "d" * 64},
        }
    }
    rows = pd.DataFrame(
        {"sample_token": [token], "global_mass": [1.0], "mid_loss": [10.0], "finite": [True]}
    )
    export = DiagnosticExport(rows, {}, {})
    journal = tmp_path / "reads.jsonl"
    append_physical_metadata(
        export, lock, [receipt], AllowlistedSources([binding(raw), binding(manifest)], journal)
    )
    assert rows.label_anchor_event_us.iloc[0] == 1200000
    assert rows.label_anchor_annotation_us.iloc[0] == 8200000
    assert rows.observation_cutoff_event_us.iloc[0] == 1200000
    assert rows.event_density_per_pixel_second.iloc[0] == 0
    assert rows.global_mass.iloc[0] == 1
    np.testing.assert_allclose(
        export.arrays["context_relative_times_seconds"], [[-0.175, -0.125, -0.075, -0.025]]
    )
    assert export.summary["physical_cuts"]["roi_area_pixels"]["groups"][1]["conditional_MiD"] == 10
    damaged = copy.deepcopy(receipt)
    damaged["identity"]["ordered_tokens"] = ["wrong_token"]
    with pytest.raises(KeyError):
        append_physical_metadata(
            export, lock, [damaged], AllowlistedSources([binding(raw), binding(manifest)], journal)
        )


def fixture(root, *, damage=None):
    receipts, roots = [], []
    for fold in range(3):
        directory = root / str(fold)
        directory.mkdir()
        tokens = [f"row{fold}", f"row{fold + 3}"]
        logits = np.full((2, 108), fold, dtype=np.float32)
        arrays = {
            "logits": logits,
            "direct_logits": logits.copy(),
            "score_correction": logits.copy(),
            "phase": np.zeros(2, np.float32),
            "ttc": np.array([fold, fold + 3], dtype=np.float32),
            "evidence": np.zeros((2, 4, 108), np.float32),
            "observability": np.zeros((2, 4, 108), bool),
            "phase_edges": np.linspace(-1, 1, 109, dtype=np.float32),
        }
        if fold == 2:
            if damage == "edges":
                arrays["phase_edges"] += 0.01
            elif damage == "dtype":
                arrays["evidence"] = arrays["evidence"].astype(np.float64)
            elif damage == "missing":
                del arrays["observability"]
            elif damage == "nonfinite":
                arrays["ttc"][0] = np.nan
        identity = {"ordered_tokens": tokens, "targets_read": False, "optimizer_updates": 0}
        identity_sha = object_digest(identity)
        shard = {
            "identity_sha256": identity_sha,
            "first_row": 0,
            "tokens": tokens,
            "arrays": _publish_arrays(directory / "inference_00000.npz", arrays),
        }
        shard["artifact_sha256"] = object_digest(shard)
        receipt = {
            "kind": "FIXED_ENDPOINT_OUTER_INFERENCE",
            "identity": identity,
            "identity_sha256": identity_sha,
            "rows": 2,
            "shards": [shard],
        }
        receipt["artifact_sha256"] = object_digest(receipt)
        receipts.append(receipt)
        roots.append(directory)
    return receipts, roots


def test_original_row_order_restored_across_all_three_outer_folds(tmp_path):
    receipts, roots = fixture(tmp_path)
    arrays, edges = assemble_fold_arrays(receipts, roots, tuple(f"row{i}" for i in range(6)))
    np.testing.assert_array_equal(arrays["ttc"], np.arange(6))
    assert edges.shape == (109,)
    assert arrays["observability"].dtype == bool
    assert arrays["evidence"].shape == (6, 4, 108)
    with pytest.raises(ValueError, match="duplicate or omit"):
        assemble_fold_arrays(receipts, roots, tuple(f"row{i}" for i in range(5)))
    duplicated = copy.deepcopy(receipts)
    duplicated[2] = duplicated[1]
    with pytest.raises(ValueError, match="duplicate or omit"):
        assemble_fold_arrays(
            duplicated, [roots[0], roots[1], roots[1]], tuple(f"row{i}" for i in range(6))
        )


def test_nested_assembly_rejects_outer_oof_substitution(tmp_path):
    receipts, roots = fixture(tmp_path)
    tokens = tuple(f"row{i}" for i in range(6))
    with pytest.raises(ValueError, match="receipt identity"):
        assemble_fold_arrays(
            receipts, roots, tokens, inference_kind="FIXED_ENDPOINT_INNER_INFERENCE"
        )
    for receipt in receipts:
        receipt.pop("artifact_sha256")
        receipt["kind"] = "FIXED_ENDPOINT_INNER_INFERENCE"
        receipt["artifact_sha256"] = object_digest(receipt)
    arrays, _ = assemble_fold_arrays(
        receipts, roots, tokens, inference_kind="FIXED_ENDPOINT_INNER_INFERENCE"
    )
    np.testing.assert_array_equal(arrays["ttc"], np.arange(6))
    with pytest.raises(ValueError, match="receipt identity"):
        assemble_fold_arrays(receipts, roots, tokens)


@pytest.mark.parametrize("damage", ["edges", "dtype", "missing"])
def test_producer_schema_changes_are_not_silently_coerced(tmp_path, damage):
    receipts, roots = fixture(tmp_path, damage=damage)
    with pytest.raises(ValueError):
        assemble_fold_arrays(receipts, roots, tuple(f"row{i}" for i in range(6)))


def test_failed_row_is_retained_for_complete_universe_diagnostics(tmp_path):
    receipts, roots = fixture(tmp_path, damage="nonfinite")
    arrays, _ = assemble_fold_arrays(receipts, roots, tuple(f"row{i}" for i in range(6)))
    assert len(arrays["ttc"]) == 6
    assert np.isnan(arrays["ttc"][2])


def test_interrupted_export_reuses_identical_payloads_without_overwriting(tmp_path):
    rows = pd.DataFrame({"sample_token": ["r0", "r1"], "prediction": [2.0, np.nan]})
    arrays = {"posterior": np.array([[0.1, 0.9], [np.nan, np.nan]])}
    first = publish_export_payloads(tmp_path, rows, arrays)
    mtimes = [
        (tmp_path / name).stat().st_mtime_ns for name in ("ROWS.csv", "DIAGNOSTIC_ARRAYS.npz")
    ]
    assert publish_export_payloads(tmp_path, rows, arrays) == first
    assert mtimes == [
        (tmp_path / name).stat().st_mtime_ns for name in ("ROWS.csv", "DIAGNOSTIC_ARRAYS.npz")
    ]
    with pytest.raises(ValueError, match="export rows differ"):
        publish_export_payloads(tmp_path, rows.iloc[::-1], arrays)
    with pytest.raises(ValueError, match="export arrays differ"):
        publish_export_payloads(tmp_path, rows, {"posterior": np.ones((2, 2))})
