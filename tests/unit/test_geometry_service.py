"""Service replay contracts for the frozen supplied-box geometry head."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from operational.geometry_service import pack, replay, rgb_run
from operational.geometry_service import run as service_run
from operational.phase_geometry.geometry import TrackPool


def _pool() -> TrackPool:
    return TrackPool(
        frame_us=np.asarray([800_000, 900_000, 1_000_000], dtype=np.int64),
        available_us=np.asarray([800_000, 900_000, 1_000_000], dtype=np.int64),
        boxes=np.asarray(
            [[0.0, 0.0, 9.0, 9.0], [0.0, 0.0, 19.0, 19.0], [0.0, 0.0, 39.0, 39.0]],
            dtype=np.float64,
        ),
        disagreements=0,
    )


def test_box_delay_moves_only_availability_and_excludes_too_new_boxes() -> None:
    delayed = replay.delay_pool(_pool(), 50_000)
    assert delayed.frame_us.tolist() == [800_000, 900_000, 1_000_000]
    assert delayed.available_us.tolist() == [850_000, 950_000, 1_050_000]
    assert delayed.boxes.tolist() == _pool().boxes.tolist()
    assert replay.available_box_count(delayed, 1_000_000) == 2
    assert replay.available_box_count(replay.delay_pool(_pool(), 0), 1_000_000) == 3


def test_offer_groups_are_deterministic_and_require_common_physical_instants() -> None:
    rows = [
        {"row": 0, "sequence": "A", "query_us": 10, "track": "x"},
        {"row": 1, "sequence": "A", "query_us": 10, "track": "y"},
        {"row": 2, "sequence": "A", "query_us": 20, "track": "x"},
        {"row": 3, "sequence": "A", "query_us": 20, "track": "y"},
        {"row": 4, "sequence": "B", "query_us": 10, "track": "x"},
    ]
    first = replay.select_offer_rows(rows, updates=2, objects=2, seed=17)
    second = replay.select_offer_rows(rows, updates=2, objects=2, seed=17)
    assert first == second
    assert {tuple(group) for group in first} == {(0, 1), (2, 3)}
    try:
        replay.select_offer_rows(rows, updates=3, objects=2, seed=17)
    except ValueError as exc:
        assert "physical instants" in str(exc)
    else:  # pragma: no cover - explicit mutation guard
        raise AssertionError("insufficient common instants must be rejected")


def test_long_replay_cycles_a_fixed_unique_population_without_changing_offers() -> None:
    unique = [[1], [2], [3]]
    assert replay.cycle_offers(unique, 8) == [[1], [2], [3], [1], [2], [3], [1], [2]]
    with pytest.raises(ValueError, match="nonempty"):
        replay.cycle_offers([], 3)


def test_paced_cell_reports_offer_drops_deadlines_and_per_object_parity() -> None:
    offers = [
        [replay.ServiceObject("a", 1), replay.ServiceObject("b", 2)],
        [replay.ServiceObject("a", 3), replay.ServiceObject("b", 4)],
        [replay.ServiceObject("a", 5), replay.ServiceObject("b", 6)],
    ]

    def predict(item: replay.ServiceObject) -> replay.Prediction:
        return replay.Prediction(value=float(item.payload), reference=float(item.payload))

    result = replay.run_paced_cell(
        offers,
        predict,
        queue_capacity=3,
        rate_hz=200.0,
        parity_atol=0.0,
        startup_lead_s=0.0,
    )
    assert result["offer_accounting"] == {
        "offered": 3,
        "enqueued": 3,
        "dropped": 0,
        "source_errors": 0,
        "completed": 3,
        "valid": 3,
        "delivered_strictly_under_20ms": 3,
    }
    assert result["parity"] == {
        "count": 6,
        "max_abs": 0.0,
        "atol": 0.0,
        "passed": True,
    }
    assert result["latency_ms"]["count"] == 3
    assert result["latency_ms"]["max_ms"] < 20.0


def test_overloaded_cell_keeps_all_offers_and_exposes_queue_drops() -> None:
    offers = [[replay.ServiceObject("a", index)] for index in range(8)]

    def slow(item: replay.ServiceObject) -> replay.Prediction:
        time.sleep(0.012)
        return replay.Prediction(value=float(item.payload), reference=float(item.payload))

    result = replay.run_paced_cell(
        offers,
        slow,
        queue_capacity=1,
        rate_hz=500.0,
        parity_atol=0.0,
        startup_lead_s=0.0,
    )
    assert result["offer_accounting"]["offered"] == 8
    assert result["offer_accounting"]["dropped"] > 0
    assert result["offer_accounting"]["completed"] < 8
    assert result["parity"]["passed"]


def test_paced_cell_labels_the_actual_prediction_component() -> None:
    offers = [[replay.ServiceObject("a", 1)]]

    def predict(item: replay.ServiceObject) -> replay.Prediction:
        return replay.Prediction(value=float(item.payload), reference=float(item.payload))

    result = replay.run_paced_cell(
        offers,
        predict,
        queue_capacity=1,
        rate_hz=50.0,
        parity_atol=0.0,
        prediction_component="RGB_GEO_BAND",
    )

    assert result["prediction_components"] == ["RGB_GEO_BAND"]
    assert result["object_records"][0]["result"]["predictions"] == {"RGB_GEO_BAND": 1.0}


def test_real_runner_defaults_cover_the_preregistered_service_matrix() -> None:
    args = service_run.parser().parse_args(["--out", "fresh"])
    assert args.rate_hz == 50.0
    assert args.queue_capacities == [2, 64]
    assert args.object_counts == [1, 4]
    assert args.box_delays_ms == [0, 50]
    assert args.parity_atol == 1e-6
    assert args.device == "cpu"
    assert args.unique_offers is None


def test_query_prediction_uses_delayed_supplied_boxes_and_frozen_head() -> None:
    class Head:
        def ttc(self, x, anchor, width):
            del anchor, width
            return float(x[0, 4])  # normalized number of available frames

    immediate = service_run.predict_query(_pool(), 1_000_000, Head(), delay_us=0)
    delayed = service_run.predict_query(_pool(), 1_000_000, Head(), delay_us=50_000)
    assert immediate == pytest.approx(3 / 7)
    assert delayed == pytest.approx(2 / 7)


def test_nested_parquet_boxes_are_stacked_like_the_canonical_geometry_reader() -> None:
    parquet_value = np.empty(2, dtype=object)
    parquet_value[0] = np.asarray([1, 2, 3, 4])
    parquet_value[1] = np.asarray([5, 6, 7, 8])
    assert service_run.stack_boxes(parquet_value).tolist() == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]


def test_report_expresses_on_time_delivery_as_population_percentage() -> None:
    cell = {
        "cell_id": "cell",
        "latency_ms": {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0, "max_ms": 4.0},
        "offer_accounting": {
            "offered": 4,
            "valid": 3,
            "dropped": 1,
            "delivered_strictly_under_20ms": 2,
        },
        "cold_call_ms": 5.0,
        "parity": {"passed": True},
    }
    report = service_run._render_report(
        {
            "cells": [cell],
            "protocol": {"offered_duration_s": 2.0, "unique_offers_per_object_count": 4},
        }
    )
    assert "50.00%" in report


def test_source_receipt_binds_uncommitted_runner_and_critical_dependencies() -> None:
    receipt = service_run.source_receipt(Path.cwd())
    expected = {
        "operational/geometry_service/run.py",
        "operational/geometry_service/replay.py",
        "operational/train40_runtime/paced.py",
        "operational/phase_geometry/geometry.py",
        "operational/variant_latency/run.py",
    }
    assert expected <= set(receipt["sha256"])
    assert all(len(value) == 64 for value in receipt["sha256"].values())
    assert isinstance(receipt["git_dirty"], bool)
    assert isinstance(receipt["git_status_porcelain"], list)


def test_rgb_service_defaults_are_long_paced_and_gpu_only() -> None:
    args = rgb_run.parser().parse_args(["--out", "fresh"])
    assert args.device == "cuda"
    assert args.rate_hz == 50.0 and args.updates == 3000
    assert args.unique_offers == 2
    assert args.queue_capacities == [2, 64]
    assert args.object_counts == [1, 4]
    assert args.box_delays_ms == [0, 50]
    assert args.population_from is None
    assert rgb_run.cell_id(50, 4, 64) == "rgb_geo__supplied_050ms__objects_04__queue_64"


def test_rgb_delay_filters_unavailable_observations_and_shifts_all_box_clocks() -> None:
    from operational.variant_latency.run import Query

    query = Query(
        row=1,
        token="token",
        sequence="sequence",
        track="track",
        query_us=1_000_000,
        pool=_pool(),
        event_end_us=0,
        region=None,  # type: ignore[arg-type]
        centers={},
        observations=[{"name": "old"}, {"name": "new"}],
        obs_anchor_us=[850_000, 950_000],
        obs_available_us=[900_000, 970_000],
        events_path=Path("unused"),
    )

    delayed = rgb_run.causal_box_delay(query, 50_000)

    assert delayed.pool.available_us.tolist() == [850_000, 950_000, 1_050_000]
    assert delayed.observations == [{"name": "old"}]
    assert delayed.obs_anchor_us == [850_000]
    assert delayed.obs_available_us == [950_000]
    assert query.obs_available_us == [900_000, 970_000]
    with pytest.raises(ValueError, match="no RGB observation"):
        rgb_run.causal_box_delay(query, 151_000)


def test_windows_power_request_is_process_lifetime_and_resettable() -> None:
    calls: list[int] = []

    class Kernel:
        def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802
            calls.append(flags)
            return 1

    receipt, reset = rgb_run.windows_power_request(kernel32=Kernel(), platform_name="Windows")
    assert receipt["active"] is True
    assert receipt["scope"] == "current_process_main_thread"
    reset()
    assert calls[-1] == rgb_run.ES_CONTINUOUS


def test_gpu_process_receipt_parses_other_compute_users() -> None:
    rows = rgb_run.parse_gpu_processes("2684, python.exe, 6767 MiB\n42, app.exe, [N/A]\n")
    assert rows == [
        {"pid": 2684, "process_name": "python.exe", "used_memory": "6767 MiB"},
        {"pid": 42, "process_name": "app.exe", "used_memory": "[N/A]"},
    ]


def test_nvidia_smi_bytes_use_windows_ansi_encoding_under_python_utf8_mode() -> None:
    payload = b"2684, C:\\Users\\\xc1lvaro\\python.exe, 100 MiB\n"
    text = rgb_run.decode_nvidia_smi_output(payload, encoding="cp1252")
    assert text == "2684, C:\\Users\\Álvaro\\python.exe, 100 MiB\n"


def test_gpu_workload_audit_distinguishes_runner_from_foreign_python() -> None:
    rows = [
        {"pid": 10, "process_name": "python.exe", "used_memory": "100 MiB"},
        {"pid": 20, "process_name": "python.exe", "used_memory": "[N/A]"},
        {"pid": 30, "process_name": "explorer.exe", "used_memory": "[N/A]"},
    ]
    audit = rgb_run.gpu_workload_audit(rows, runner_pid=10)
    assert audit["runner_visible"] is True
    assert audit["foreign_python_pids"] == [20]
    assert audit["dedicated_model_workload"] is False


def test_campaign_validity_requires_no_host_pause_and_dedicated_model_workload() -> None:
    validity = rgb_run.campaign_validity(
        {"campaign_valid": True},
        {"dedicated_model_workload": False, "runner_visible": True},
    )
    assert validity == {
        "decision": "INVALIDATED",
        "host_pause_valid": True,
        "dedicated_model_workload": False,
        "runner_visible": True,
    }


def test_gpu_process_snapshots_union_every_cell_boundary() -> None:
    snapshots = [
        [{"pid": 10, "process_name": "runner", "used_memory": "10 MiB"}],
        None,
        [{"pid": 20, "process_name": "foreign", "used_memory": "20 MiB"}],
    ]
    assert rgb_run.merge_gpu_process_snapshots(snapshots) == [
        {"pid": 10, "process_name": "runner", "used_memory": "10 MiB"},
        {"pid": 20, "process_name": "foreign", "used_memory": "20 MiB"},
    ]


def test_trace_pack_accepts_rgb_service_cell_names(tmp_path: Path) -> None:
    (tmp_path / "rgb_geo__supplied_000ms__objects_01__queue_02.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "RESULTS.json").write_text("{}", encoding="utf-8")
    (tmp_path / "POPULATION.json").write_text("{}", encoding="utf-8")
    assert [path.name for path in pack.trace_files(tmp_path)] == [
        "POPULATION.json",
        "RESULTS.json",
        "rgb_geo__supplied_000ms__objects_01__queue_02.json",
    ]


def test_media_receipt_binds_exact_decoded_frame_bytes() -> None:
    class Frames:
        frames = {("seq/rgb_shards/0.tar", "frame.png"): np.asarray([[[1, 2, 3]]], np.uint8)}

    class Q:
        token = "token"

    receipt = rgb_run.decoded_media_receipts([Q()], [Frames()])
    assert receipt == [
        {
            "sample_token": "token",
            "shard": "seq/rgb_shards/0.tar",
            "member": "frame.png",
            "dtype": "uint8",
            "shape": [1, 1, 3],
            "decoded_bytes_sha256": (
                "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
            ),
        }
    ]


def test_host_pause_audit_invalidates_massive_scheduler_jump() -> None:
    cells = [
        {"cell_id": "ok", "scheduler_lateness_max_ms": 4.0},
        {"cell_id": "suspended", "scheduler_lateness_max_ms": 120_000.0},
    ]
    assert rgb_run.host_pause_audit(cells, threshold_ms=1_000.0) == {
        "threshold_ms": 1_000.0,
        "max_scheduler_lateness_ms": 120_000.0,
        "cells_over_threshold": ["suspended"],
        "campaign_valid": False,
    }
