"""Label-free released Garl-TTC inference (full RGB+event and event-only) through the release code.

``python -m operational.matched_comparator.garl_runner --population {train40,test12} --out <dir>``
builds the upstream ``TTCEstimationDataset`` in its HF-parquet *inference* mode (no labels
parquet is passed, so ``has_labels`` is False and no TTC/3D field is read), runs the released
``paper_ours_full.pth`` exactly as ``garl_ttc.engine.inference.run_inference`` does
(``model(data)`` then ``dT / (1 - h0/h1)``) and, on the same tensors, the released
event-only checkpoint on channels ``6:``.  Results are written in resumable fragments of
``--chunk`` rows with both visible-height pairs.  Rows the upstream loader rejects are kept as
explicit failures (no prediction), exactly as the release runner would omit them.

The OpenCV runtime is not part of the integration environment; put an ``opencv-python-headless``
target on ``PYTHONPATH`` (documented in the REPORT).  No optimizer step, no upload.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import Dataset

from .sources import ENV_EAP_ROOT, ENV_GARLTTC_BENCHMARK, location_root, sha256_file, verified

TEST12_TOKENS = 6762


def _prepare_test_parquet(out: Path) -> Path:
    """Write the official public ``test_inputs.parquet`` (label-free, SHA-pinned) to ``out``."""
    import pandas as pd

    from operational.phase_geometry.data import FORBIDDEN_TEST_COLUMNS, SOURCES, read_member

    path = out / "test_inputs.parquet"
    data = read_member("test_inputs")
    frame = pd.read_parquet(io.BytesIO(data))
    if FORBIDDEN_TEST_COLUMNS.intersection(frame.columns) or len(frame) != TEST12_TOKENS:
        raise ValueError("Public test inputs contain a label column or differ in size")
    if not path.exists() or sha256_file(path) != SOURCES["test_inputs"]["sha256"]:
        path.write_bytes(data)
    return path


def build_config(population: str, out: Path, workers: int, batch: int) -> dict[str, Any]:
    """Release config (README default) pointed at the label-free population parquet."""
    from garl_ttc.config import load_config

    eap = location_root(ENV_EAP_ROOT)
    cfg = load_config(verified("garl_full_config"), data_root=str(eap))
    cfg["dataset"]["annotation_format"] = "parquet"
    if population == "test12":
        data_parquet = _prepare_test_parquet(out)
    elif population == "train40":
        data_parquet = verified("garl_train_inputs")
    else:
        raise ValueError(f"Unknown population {population}")
    # Inference split: only the data parquet; no labels parquet and no asset filter.
    cfg["dataset"]["test"] = {"data_parquet": str(data_parquet)}
    cfg["testing_settings"].update(batch_size=batch, num_threads=workers, shuffle=False)
    for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
        cfg["model"].pop(key, None)
    return cfg


def load_models(cfg: dict[str, Any], device: str) -> tuple[Any, Any]:
    """Strictly load the full and the event-only released checkpoints."""
    import torch
    import yaml
    from garl_ttc.models import TTCNetwork

    full = TTCNetwork(cfg, is_train=False)
    full.load_state_dict(
        torch.load(verified("garl_full_checkpoint"), map_location="cpu", weights_only=True),
        strict=True,
    )
    event_cfg = yaml.safe_load(verified("garl_event_config").read_text(encoding="utf-8"))
    for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
        event_cfg["model"].pop(key, None)
    event = TTCNetwork(event_cfg, is_train=False)
    event.load_state_dict(
        torch.load(verified("garl_event_checkpoint"), map_location="cpu", weights_only=True),
        strict=True,
    )
    if float(full.dT) != 0.1 or float(event.dT) != 0.1:
        raise ValueError("Released Garl dT changed")
    return (
        full.float().eval().requires_grad_(False).to(device),
        event.float().eval().requires_grad_(False).to(device),
    )


def release_ttc(heights: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Release conversion ``dT / (1 - h0/h1)`` without clipping (``prediction_to_ttc``)."""
    h = np.asarray(heights, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return delta / (1.0 - h[:, 0] / h[:, 1])


class IndexedRows:
    """Map-style wrapper returning ``(absolute row index, upstream sample or None)``."""

    def __init__(self, base: Dataset[Any], offset: int) -> None:
        self.base = base
        self.offset = offset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[int, Any]:
        return self.offset + index, self.base[index]


def indexed_collate(batch: list[tuple[int, Any]]) -> dict[str, Any]:
    """Keep every row index (rejected rows included) and collate the accepted samples."""
    from garl_ttc.datasets.ttc_dataset import inference_collate_fn

    accepted = [(i, s) for i, s in batch if s is not None]
    return {
        "index": [i for i, _ in batch],
        "ok_index": [i for i, _ in accepted],
        "batch": inference_collate_fn([s for _, s in accepted]) if accepted else None,
    }


def _empty_buffer() -> dict[str, list[Any]]:
    return {"seen": [], "token": [], "full": [], "event": []}


def _flush(
    out: Path,
    population: str,
    rows: list[dict[str, Any]],
    start: int,
    stop: int,
    buf: dict[str, list[Any]],
) -> None:
    path = out / f"{population}_{start:06d}.npz"
    if path.exists():
        return
    expected = [str(r["sample_token"]) for r in rows[start:stop]]
    found = set(buf["token"])
    temporary = path.with_name(path.stem + ".pending.npz")
    np.savez_compressed(
        temporary,
        sample_token=np.asarray(buf["token"]),
        full_heights=np.asarray(buf["full"], np.float32).reshape(-1, 2),
        event_heights=np.asarray(buf["event"], np.float32).reshape(-1, 2),
        failed_token=np.asarray([t for t in expected if t not in found]),
    )
    temporary.replace(path)


def run(
    population: str,
    out: Path,
    chunk: int,
    workers: int,
    batch: int,
    limit: int | None,
    tokens: list[str] | None = None,
    tag: str = "probe",
) -> None:
    """Run (or resume) every fragment and write a sealed manifest when complete."""
    code_root = location_root(ENV_GARLTTC_BENCHMARK)
    for name in ("garl_ttc_network", "garl_ttc_dataset", "garl_ttc_inference"):
        verified(name)
    sys.path.insert(0, str(code_root))
    # Spawned DataLoader workers must import garl_ttc too (they inherit the environment).
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(code_root), os.environ.get("PYTHONPATH", "")) if p
    )
    import torch
    from garl_ttc.datasets import TTCEstimationDataset
    from torch.utils.data import DataLoader

    out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(population, out, workers, batch)
    dataset = TTCEstimationDataset(
        cfg, "test", seed=0, db_mode="sequence", seq_ts=5, test_ttc_range=(-10, 10)
    )
    if dataset.has_labels:
        raise ValueError("Inference dataset unexpectedly carries labels")
    rows = list(dataset.db)
    prefix = population
    if tokens is not None:
        keep = set(tokens)
        rows = [r for r in rows if str(r["sample_token"]) in keep]
        prefix = f"{population}{tag}"
    total = len(rows) if limit is None else min(limit, len(rows))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    full, event = load_models(cfg, device)
    binding = {
        "population": population,
        "rows": total,
        "pins": {
            n: sha256_file(verified(n))
            for n in (
                "garl_full_checkpoint",
                "garl_full_config",
                "garl_event_checkpoint",
                "garl_event_config",
                "garl_ttc_network",
                "garl_ttc_dataset",
                "garl_ttc_inference",
            )
        },
        "device": device,
        "batch": batch,
        "workers": workers,
        "labels_read": False,
        "optimizer_updates": 0,
    }
    binding_path = out / f"{prefix}_BINDING.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Existing fragments were produced under a different binding")
    binding_path.write_text(json.dumps(binding, indent=1), encoding="utf-8")
    started = time.perf_counter()
    starts = [s for s in range(0, total, chunk) if not (out / f"{prefix}_{s:06d}.npz").exists()]
    if starts:
        first = starts[0]
        dataset.db = rows[first:total]
        dataset.indices = torch.arange(total - first)
        loader = DataLoader(
            IndexedRows(dataset, first),
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            collate_fn=indexed_collate,
        )
        buffers: dict[int, dict[str, list[Any]]] = {}
        with torch.inference_mode():
            for item in loader:
                for index in item["index"]:
                    buffers.setdefault(index // chunk, _empty_buffer())["seen"].append(index)
                if item["batch"] is not None:
                    data = item["batch"]["data"].to(device, non_blocking=True)
                    full_out, _ = full(data)
                    event_out, _ = event(data[:, 6:])
                    full_np = full_out.float().cpu().numpy().reshape(len(data), -1)
                    event_np = event_out.float().cpu().numpy().reshape(len(data), -1)
                    for k, index in enumerate(item["ok_index"]):
                        buf = buffers[index // chunk]
                        buf["token"].append(str(item["batch"]["sample_token"][k]))
                        buf["full"].append(full_np[k])
                        buf["event"].append(event_np[k])
                for key in sorted(buffers):
                    start, stop = key * chunk, min(key * chunk + chunk, total)
                    if len(buffers[key]["seen"]) == stop - start:
                        _flush(out, prefix, rows, start, stop, buffers.pop(key))
                        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
                        print(
                            json.dumps(
                                {
                                    "population": population,
                                    "done": stop,
                                    "total": total,
                                    "seconds": round(time.perf_counter() - started, 1),
                                    "peak_vram_gib": round(peak, 3),
                                }
                            ),
                            flush=True,
                        )
        if buffers:
            raise RuntimeError("Loader ended with incomplete fragments")
    fragments = [p for p in sorted(out.glob(f"{prefix}_0*.npz")) if "pending" not in p.name]
    (out / f"{prefix}_MANIFEST.json").write_text(
        json.dumps(
            {
                "binding_sha256": sha256_file(binding_path),
                "fragments": {p.name: sha256_file(p) for p in fragments},
                "labels_read": False,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def load_fragments(out: Path, population: str) -> dict[str, Any]:
    """Concatenate fragments: tokens, both height pairs, release TTCs and failed tokens."""
    tokens, full, event, failed = [], [], [], []
    paths = [p for p in sorted(out.glob(f"{population}_0*.npz")) if "pending" not in p.name]
    for path in paths:
        with np.load(path, allow_pickle=False) as part:
            tokens.append(part["sample_token"].astype(str))
            full.append(part["full_heights"])
            event.append(part["event_heights"])
            failed.append(part["failed_token"].astype(str))
    full_h = np.concatenate(full).astype(np.float64)
    event_h = np.concatenate(event).astype(np.float64)
    return {
        "sample_token": np.concatenate(tokens),
        "full_heights": full_h,
        "event_heights": event_h,
        "full_ttc": release_ttc(full_h),
        "event_ttc": release_ttc(event_h),
        "failed_token": np.concatenate(failed),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("train40", "test12"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="probe", help="file prefix suffix for a token subset")
    parser.add_argument(
        "--tokens-csv", type=Path, default=None, help="probe only the sample_token column of a CSV"
    )
    args = parser.parse_args()
    tokens = None
    if args.tokens_csv is not None:
        import pandas as pd

        tokens = pd.read_csv(args.tokens_csv)["sample_token"].astype(str).tolist()
    run(
        args.population,
        args.out.resolve(),
        args.chunk,
        args.workers,
        args.batch,
        args.limit,
        tokens,
        args.tag,
    )


if __name__ == "__main__":
    main()
