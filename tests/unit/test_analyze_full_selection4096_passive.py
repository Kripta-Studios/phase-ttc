from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "analyze_full_selection4096_passive.py"
    spec = importlib.util.spec_from_file_location("full_selection_profile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load profile script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_categories_are_exclusive_and_unresolved_is_retained(tmp_path: Path) -> None:
    folded = tmp_path / "raw.txt"
    folded.write_text(
        "run (train_detection_campaign.py:768);tarfile.py:1;PngImagePlugin:2 3\n"
        "run (train_detection_campaign.py:768);decode_eap (x.py:1) 2\n"
        "run (train_detection_campaign.py:768);unknown (x.py:1) 1\n",
        encoding="utf-8",
    )
    result = cast(Any, _module()).classify_folded(folded)
    assert result["total_samples"] == 6
    categories = result["exclusive_categories"]
    assert categories["raw_tar"]["samples"] == 3
    assert categories["png_decode"]["samples"] == 0
    assert categories["decode"]["samples"] == 2
    assert categories["unresolved"]["samples"] == 1
    assert sum(value["samples"] for value in categories.values()) == 6


def test_render_labels_samples_not_wall_time() -> None:
    categories = {
        name: {"samples": 1, "sample_fraction": 0.125}
        for name in (
            "forward",
            "png_decode",
            "prepare",
            "decode",
            "finite_checks",
            "ground_truth",
            "raw_tar",
            "unresolved",
        )
    }
    analysis = {
        "pyspy": {
            "exclusive_categories": categories,
            "run_line_768_samples": 8,
            "total_samples": 8,
            "run_line_768_fraction": 1.0,
        },
        "passive": {
            "trainer_cpu_seconds": 10.0,
            "elapsed_seconds": 5.0,
            "mean_logical_core_equivalents": 2.0,
            "trainer_read_mib_per_second": 3.0,
            "system_read_mib_per_second": 4.0,
            "gpu_utilization_percent_mean": 13.0,
            "gpu_utilization_percent_peak": 42.0,
        },
    }
    rendered = cast(Any, _module()).render_section(analysis)
    assert "no fracciones de tiempo de pared" in rendered
    assert "no demuestra starvation" in rendered
