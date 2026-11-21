"""TSTTC metric fidelity, safe annotation reading and sequence building on synthetic data."""

from __future__ import annotations

import math
import os
import pickle

import numpy as np
import pytest

from operational.second_benchmark import tsttc


def _official_mid(gt: float, pred: float) -> float:
    """Literal transcription of tusen-ai/TSTTC data/dataset_api.py::evalRes (scalar)."""

    def ttc_to_scale_ratio(ttc: float, fps: int = 10) -> float:
        return 1 / ((1 / (ttc * fps) + 1) + 1e-6)

    pred = max(min(pred, 20), -20)
    return float(abs(math.log(ttc_to_scale_ratio(gt)) - math.log(ttc_to_scale_ratio(pred))) * 10**4)


def _official_rte(gt: float, pred: float) -> float:
    pred = max(min(pred, 20), -20)
    return abs((pred - gt) / gt) * 100


@pytest.mark.parametrize(
    ("gt", "pred"), [(2.0, 2.5), (0.8, 0.5), (-4.0, -7.0), (15.0, 60.0), (5.0, -3.0), (19.0, 25.0)]
)
def test_mid_and_rte_equal_official_definition(gt: float, pred: float) -> None:
    mid, rte = tsttc.row_metrics(np.array([gt]), np.array([pred]))
    assert mid[0] == pytest.approx(_official_mid(gt, pred), rel=1e-12, abs=1e-9)
    assert rte[0] == pytest.approx(_official_rte(gt, pred), rel=1e-12)


def test_mid_uses_plus_convention_not_garl_minus() -> None:
    # TSTTC: log(T/(T+0.1)); GarlTTC: log(1-0.1/T).  They differ measurably at small T.
    gt, pred = 1.0, 1.5
    mid, _ = tsttc.row_metrics(np.array([gt]), np.array([pred]))
    plus = 1e4 * abs(math.log1p(0.1 / gt) - math.log1p(0.1 / pred))
    minus = 1e4 * abs(math.log1p(-0.1 / gt) - math.log1p(-0.1 / pred))
    assert mid[0] == pytest.approx(plus, rel=1e-4)
    assert abs(mid[0] - minus) > 10.0


def test_prediction_clip_and_zero_rejection() -> None:
    mid_far, _ = tsttc.row_metrics(np.array([10.0]), np.array([1e6]))
    mid_20, _ = tsttc.row_metrics(np.array([10.0]), np.array([20.0]))
    assert mid_far[0] == pytest.approx(mid_20[0])
    with pytest.raises(ValueError, match="cannot be zero"):
        tsttc.row_metrics(np.array([1.0]), np.array([0.0]))


def test_summary_ranges_are_inclusive_like_evalres() -> None:
    gt = np.array([-5.0, 0.5, 3.0, 4.0, 6.0, 12.0])
    summary = tsttc.summarize(gt, gt * 1.1)
    assert summary["n crucial_ttc(0~3)"] == 2  # 0.5 and 3.0
    assert summary["n small_ttc(3~6)"] == 3  # 3.0, 4.0 and 6.0 (inclusive on both ends)
    assert summary["n large_ttc(6~20)"] == 2
    assert summary["n negative_ttc(-20~0)"] == 1
    assert summary["RTE"] == pytest.approx(10.0)


class Box2DRecord:
    """Stand-in for TSTTC's unpublished annotation class (pickled under this test module)."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _approach(
    n: int, z0: float = 30.0, v: float = 5.0, start_ns: int = 10**15
) -> list[Box2DRecord]:
    records = []
    for i in range(n):
        z = z0 - v * 0.1 * i
        half = 20.0 / z  # normalized half-size proportional to 1/Z
        records.append(
            Box2DRecord(
                bag_stamp="/data/bag_x",
                box2d=[0.5 - half / 2, 0.5 - half / 2, 0.5 + half / 2, 0.5 + half / 2],
                cam_id=1,
                id=42,
                occ_ratio=0.0,
                same_lane=True,
                ts=start_ns + i * tsttc.FRAME_STEP_NS,
                ttc_imu=z / v,
            )
        )
    return records


def test_safe_unpickler_reads_custom_objects_without_their_class() -> None:
    records = _approach(6)
    payload = pickle.dumps([{r.ts: [r] for r in records}], protocol=4)
    cameras = tsttc.load_annotations(payload)
    obj = next(iter(cameras[0].values()))[0]
    assert isinstance(obj, tsttc.AnnotationObject)
    assert obj.id == 42 and obj.box2d == records[0].box2d


class _Exploit:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, ("echo pwned",))


def test_safe_unpickler_refuses_code_execution_globals() -> None:
    with pytest.raises(pickle.UnpicklingError):
        tsttc.load_annotations(pickle.dumps([_Exploit()]))


def test_build_sequences_and_detection_baseline_on_constant_velocity() -> None:
    records = _approach(7)
    camera = {r.ts: [r] for r in records}
    camera[records[-1].ts + 5 * tsttc.FRAME_STEP_NS] = [_approach(1, start_ns=0)[0]]
    sequences = tsttc.build_sequences([camera])
    assert len(sequences) == 1
    seq = sequences[0]
    assert len(seq.ts_ns) == 6 and seq.object_id == 42
    assert np.all(np.diff(seq.ts_ns) == tsttc.FRAME_STEP_NS)
    # Detection baseline with the official scale-ratio conversion recovers T at the target frame.
    assert tsttc.detection_baseline(seq) == pytest.approx(seq.ttc_gt, rel=1e-4)
    raw = tsttc.geometry_raw(seq)
    assert raw["n_frames"] == 6 and raw["span_s"] == pytest.approx(0.5)
    assert raw["k_area"] > 0


def test_target_free_prediction_bytes_do_not_depend_on_ttc_imu() -> None:
    records = _approach(6)
    camera = {r.ts: [r] for r in records}
    original = tsttc.build_prediction_sequences([camera])

    for record in records:
        record.ttc_imu = -1234.5
    changed = tsttc.build_prediction_sequences([{r.ts: [r] for r in records}])
    for record in records:
        del record.ttc_imu
    censored = tsttc.build_prediction_sequences([{r.ts: [r] for r in records}])

    assert tsttc.prediction_bytes(original) == tsttc.prediction_bytes(changed)
    assert tsttc.prediction_bytes(original) == tsttc.prediction_bytes(censored)


def test_compatibility_sequence_builder_accepts_one_shot_iterable() -> None:
    records = _approach(6)
    sequences = tsttc.build_sequences(iter([{r.ts: [r] for r in records}]))
    assert len(sequences) == 1
    assert sequences[0].ttc_gt == pytest.approx(records[-1].ttc_imu)


def test_root_requires_environment_variable() -> None:
    with pytest.raises(RuntimeError, match="PHASE_TTC_BENCH2_ROOT"):
        tsttc.bench2_root({})
