from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from operational.matched_outer.io_probe import call_row_materializer, measure_operation


def test_measure_operation_records_time_memory_and_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"

    def operation() -> int:
        path.write_bytes(b"x" * 1024)
        return 4

    result, measurement = measure_operation(operation, bytes_path=path)
    assert result == 4
    assert measurement["wall_seconds"] >= 0.0
    assert measurement["peak_rss_bytes"] > 0
    assert measurement["bytes"] == 1024


def test_row_adapter_passes_explicit_mask_reader() -> None:
    seen: dict[str, object] = {}

    def materializer(row: Mapping[str, object], **kwargs: object) -> Mapping[str, object]:
        seen.update(kwargs)
        return row

    row = {"sample_token": "q"}
    assert call_row_materializer(materializer, row, sentinel=1) == row
    assert seen["mask_reader"] is None
    assert seen["sentinel"] == 1
