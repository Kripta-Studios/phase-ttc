"""Known paired phase losses exercise the real fixed-draw statistical implementation."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.architecture_comparison_v11 import (
    aligned_loss_vectors,
    compare_standalone_rows,
    paired_contrasts,
)
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, scientific_ttc


@pytest.fixture
def fixture():
    targets = np.tile([2.0, 4.0, 8.0, -2.0], 9)
    metadata = pd.DataFrame(
        {
            "sample_token": [f"row{i}" for i in range(36)],
            "sequence_id": np.repeat([f"seq{i}" for i in range(9)], 4),
            "track_id": ["whole_track"] * 36,
            "outer_fold": np.repeat(np.arange(9) % 3, 4),
            "target_ttc_s": targets,
        }
    )
    rows = metadata.assign(prediction_ttc_s=targets, finite=True, failure=False, mid_loss=0.0)
    risk = scientific_ttc(benchmark_phase(targets) + 0.001)
    old = np.column_stack([risk, risk, risk])
    return metadata, rows, risk, old


def test_known_improvement_uses_shared_real_bootstrap_and_retains_failed_candidate(
    fixture, tmp_path
):
    metadata, rows, risk, old = fixture
    failed = rows.copy()
    failed.loc[0, ["finite", "failure"]] = [False, True]
    result = compare_standalone_rows(
        metadata,
        {"primary": rows, "failed": failed},
        risk,
        old,
        output=tmp_path / "comparison",
        check=lambda: None,
    )
    primary = result["comparisons"]["primary"]
    assert primary["score"] == 0
    assert primary["evidence"]["delta_vs_risk17"] == pytest.approx(-10, abs=1e-10)
    assert primary["evidence"]["crucial_delta"] == pytest.approx(-10, abs=1e-10)
    assert primary["evidence"]["new_oracle_gain"] == pytest.approx(10, abs=1e-10)
    assert primary["evidence"]["sequence_wins"] == 9
    assert primary["hierarchical_ci95"] == pytest.approx([-10, -10], abs=1e-10)
    assert primary["decision"]["replicate_new_architecture"]
    assert primary["hierarchical_fraction_negative"] == 1
    assert not result["comparisons"]["failed"]["technically_valid"]
    assert result["comparisons"]["failed"]["evidence"] is None
    assert result["bootstrap"]["valid_draws"] == result["bootstrap"]["attempts"] == 8192
    assert not result["independent_confirmation"]
    assert not result["historical_stage67_gate_changed"]
    assert result["joint_oracle_model_names"] == ["primary"]


def test_comparison_recomputes_losses_and_rejects_row_or_target_substitution(fixture):
    metadata, rows, _, _ = fixture
    with pytest.raises(ValueError, match="canonical target"):
        aligned_loss_vectors(metadata, rows.iloc[::-1])
    wrong_target = rows.copy()
    wrong_target.loc[0, "target_ttc_s"] = 2.1
    with pytest.raises(ValueError, match="canonical target"):
        aligned_loss_vectors(metadata, wrong_target)
    wrong_loss = rows.copy()
    wrong_loss.loc[0, "mid_loss"] = -5
    with pytest.raises(ValueError, match="recomputation"):
        aligned_loss_vectors(metadata, wrong_loss)


def test_existing_comparison_cannot_be_silently_replaced(fixture, tmp_path):
    metadata, rows, risk, old = fixture
    with pytest.raises(ValueError, match="immutable"):
        compare_standalone_rows(
            metadata, {"primary": rows}, risk, old, output=tmp_path, check=lambda: None
        )


def test_mechanism_contrasts_preserve_shared_draw_pairing_and_missing_arms(fixture):
    metadata, _, _, _ = fixture
    losses = np.tile([3.0, 9.0], (len(metadata), 1))
    shared_noise = np.arange(8, dtype=np.float64)
    draws = np.column_stack((shared_noise + 3, shared_noise + 9))
    result = paired_contrasts(
        metadata,
        losses,
        draws,
        ["A", "EXTRA"],
        (("A", "EXTRA"), ("AD", "D")),
    )
    paired = result["A_minus_EXTRA"]
    assert paired["delta"] == pytest.approx(-6)
    assert paired["hierarchical_ci95"] == [-6, -6]
    assert paired["exact_sequence"]["ci95"] == pytest.approx([-6, -6])
    assert not paired["selects_candidate_identity"]
    assert result["AD_minus_D"]["status"] == "UNAVAILABLE_OR_INVALID_ENDPOINT"


def test_mechanism_contrasts_reject_misaligned_draw_columns(fixture):
    metadata, _, _, _ = fixture
    with pytest.raises(ValueError, match="shared paired"):
        paired_contrasts(metadata, np.ones((36, 2)), np.ones((8, 1)), ["A", "D"], (("A", "D"),))
