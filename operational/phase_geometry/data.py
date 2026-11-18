"""Label-safe input assembly for the geometry study (archives are read in memory only).

``python -m operational.phase_geometry.data --work <dir>`` writes two compact NPZ work files
(TRAIN40 OOF rows with public train targets; Test12 physical queries without labels) and a
receipt binding every source by SHA-256.  Private Test12 labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import geometry
from .metrics import json_safe, ttc_to_phase

FINAL = "workspace/ejepa_final_runs/final/"
CANDIDATE = r"D:\E_JEPA_POD2_RETURN_20261027\FINAL_CANDIDATE"
V16 = "workspace/ejepa_final_inputs/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/"
SOURCES: dict[str, dict[str, str]] = {
    "oof_phase20": {
        "archive": r"D:\E_JEPA_POD2_RETURN_20261027\final_generated_fd84565b0df06d79.tar",
        "member": FINAL + "oof/PHASE20_CONTEXT.npz",
        "sha256": "48207e3c5366057bfeba4e91ec40c145b344e8e7eb6fabc8966c82386d6c51cd",
    },
    "train_cache": {
        "archive": r"D:\E_JEPA_POD2_RETURN_20261027\final_generated_fd84565b0df06d79.tar",
        "member": FINAL + "cache/PHASE_ONLY_EVENT_S7_TRAIN40.npz",
        "sha256": "5a814615630f7e9ae372acf2123b2c9b2da880e9a950b402179adde44ba1293b",
    },
    "test_inputs": {
        "archive": r"D:\E_JEPA_POD2_RETURN_20261027\final_inputs_63e2720817ba6521.tar",
        "member": V16
        + "research_independent_20261012T165833Z/DEPENDENCIES/OFFICIAL_TEST12/test_inputs.parquet",
        "sha256": "aedb9da5484ceedb642ef1cbc7cf6115dcf7f368e4444b55de947989fc19a362",
    },
    "test_inventory": {
        "archive": r"D:\E_JEPA_POD2_RETURN_20261027\final_inputs_63e2720817ba6521.tar",
        "member": V16 + "multimodal_research_20261012/test12_phase_only/INPUTS.json",
        "sha256": "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526",
    },
    "test_diagnostics": {
        "archive": r"D:\E_JEPA_POD2_RETURN_20261027\final_delivery_bf471efaacbee542.tar",
        "member": FINAL + "test12/staged_sequences/diagnostics/",
        "sha256": "",
    },
}
FILES: dict[str, dict[str, str]] = {
    "train_data": {
        "path": r"E:\GarlTTC_dataset\data\train.parquet",
        "sha256": "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6",
    },
    "train_annotations": {
        "path": r"E:\GarlTTC_dataset\annotations\train.parquet",
        "sha256": "64e9a81232643ecce1d2f66e906c7cff6ea1e9e6958f81588b95e112693b11ad",
    },
    "released_checkpoint": {
        "path": CANDIDATE + r"\checkpoints\PHASE_ONLY_EVENT_S7_TRAIN40.pt",
        "sha256": "a24bfc72ca7f53a93d53db806b6866c6a5a4b6abde074b5dfec50436b7005408",
    },
    "submitted_zip": {
        "path": CANDIDATE + r"\PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip",
        "sha256": "5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc",
    },
    "official_scorer": {
        "path": r"E:\Garl-TTC\garl_ttc_benchmark\codabench\scoring_program\score.py",
        "sha256": "f3bbf089ba6e47edfce1de522c92ddc969db2260fac7544e82bc017fedf82dce",
    },
    "blocks": {
        "path": "docs/final_refit_20261017/NESTED_ROTATION_REVIEW.json",
        "sha256": "12465d3b63dba27573461c098f2d19046e89c64c3d32444dfbf4b533fa5e8ec1",
    },
}
BOX_COLUMNS = [
    "sequence_id",
    "sample_token",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "boxes_xyxy",
]
FORBIDDEN_TEST_COLUMNS = {"ttc", "frame_ttc", "box3d_h", "box3d_Fcam", "target", "label"}


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_file(name: str, root: Path) -> Path:
    """Resolve a pinned file (relative paths against the repository root) and check its hash."""
    spec = FILES[name]
    path = Path(spec["path"])
    path = path if path.is_absolute() else root / path
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(f"{name} differs from its pinned SHA-256: {path}")
    return path


def read_member(name: str) -> bytes:
    """Read one pinned archive member fully into memory and verify its SHA-256."""
    spec = SOURCES[name]
    with tarfile.open(spec["archive"]) as archive:
        handle = archive.extractfile(spec["member"])
        if handle is None:
            raise FileNotFoundError(spec["member"])
        data = handle.read()
    if spec["sha256"] and sha256_bytes(data) != spec["sha256"]:
        raise ValueError(f"Archive member {name} differs from RESTORE_INDEX SHA-256")
    return data


def load_npz_bytes(data: bytes) -> dict[str, NDArray[Any]]:
    """Load a non-pickled NPZ payload from memory."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def iter_diagnostics() -> Iterator[tuple[str, bytes, dict[str, Any]]]:
    """Yield (npz member, npz bytes, metadata JSON) for every staged Test12 diagnostic."""
    spec = SOURCES["test_diagnostics"]
    pending: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    with tarfile.open(spec["archive"]) as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(spec["member"]):
                continue
            stem, suffix = member.name.rsplit(".", 1)
            handle = archive.extractfile(member)
            if handle is None:
                continue
            data = handle.read()
            if suffix == "json":
                pending[stem] = json.loads(data)
            elif suffix == "npz":
                payloads[stem] = data
            for key in [k for k in pending if k in payloads]:
                yield key, payloads.pop(key), pending.pop(key)
    if pending or payloads:
        raise ValueError("Unpaired Test12 diagnostic members")


def fold_of(groups: NDArray[Any], root: Path) -> tuple[NDArray[np.int64], dict[str, list[str]]]:
    """Map each row's sequence to its fixed block index B0..B4."""
    review = json.loads(verified_file("blocks", root).read_text(encoding="utf-8"))
    blocks: dict[str, list[str]] = review["group_blocks"]
    lookup = {seq: int(name[1:]) for name, members in blocks.items() for seq in members}
    names = np.asarray(groups).astype(str)
    if set(names) != set(lookup):
        raise ValueError("Block partition does not cover exactly the TRAIN40 sequences")
    return np.asarray([lookup[s] for s in names], dtype=np.int64), blocks


def _geometry(frame: pd.DataFrame) -> tuple[NDArray[np.float64], int]:
    return geometry.compute_table(
        frame["sequence_id"].astype(str).tolist(),
        frame["track_id"].astype(str).tolist(),
        frame["timestamp_us"].to_numpy(np.int64),
        [np.asarray(v, dtype=np.int64) for v in frame["frame_timestamps_us"]],
        [np.stack(v).astype(np.float64) for v in frame["boxes_xyxy"]],
    )


def prepare_train(root: Path) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Assemble OOF head inputs, public targets, folds and causal geometry for TRAIN40."""
    oof = load_npz_bytes(read_member("oof_phase20"))
    cache = load_npz_bytes(read_member("train_cache"))
    tokens = oof["sample_token"].astype(str)
    if not np.array_equal(tokens, cache["sample_token"].astype(str)):
        raise ValueError("OOF and training-cache token order differ")
    selected = np.concatenate((cache["features"][:, 160:170], cache["features"][:, 330:340]), 1)
    if not np.array_equal(selected, oof["features"]) or not np.array_equal(
        cache["rgb_width_logits"], oof["rgb_width_logits"]
    ):
        raise ValueError("PHASE20 OOF fields differ from the released training cache")
    data = pd.read_parquet(verified_file("train_data", root), columns=BOX_COLUMNS)
    annotations = pd.read_parquet(
        verified_file("train_annotations", root), columns=["sample_token", "ttc", "sequence_id"]
    )
    data = data.set_index("sample_token").loc[tokens].reset_index()
    target = annotations.set_index("sample_token").loc[tokens, "ttc"].to_numpy(np.float64)
    groups = oof["group_id"].astype(str)
    if not np.array_equal(data["sequence_id"].to_numpy(dtype=str), groups):
        raise ValueError("Sequence identity differs between OOF rows and public train data")
    phase_gap = float(np.abs(ttc_to_phase(target) - cache["target_phase"].astype(np.float64)).max())
    if phase_gap > 1e-6:
        raise ValueError("Public train TTC differs from the released training target phase")
    raw, disagreements = _geometry(data)
    folds, blocks = fold_of(groups, root)
    arrays = {
        "sample_token": tokens,
        "group_id": groups,
        "track_id": data["track_id"].to_numpy(dtype=str),
        "timestamp_us": data["timestamp_us"].to_numpy(np.int64),
        "fold": folds,
        "features": oof["features"].astype(np.float32),
        "rgb_width_logits": oof["rgb_width_logits"].astype(np.float32),
        "geometry_raw": raw,
        "geometry_g": geometry.learned_inputs(raw),
        "target_ttc": target,
    }
    receipt = {
        "rows": len(tokens),
        "groups": len(set(groups)),
        "blocks": blocks,
        "box_disagreements": disagreements,
        "max_public_target_phase_gap": phase_gap,
        "targets_used_for_geometry": False,
    }
    return arrays, receipt


def prepare_test() -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Assemble label-free Test12 head inputs (released diagnostics) and causal geometry."""
    test = pd.read_parquet(io.BytesIO(read_member("test_inputs")))
    if FORBIDDEN_TEST_COLUMNS.intersection(test.columns):
        raise ValueError("Public test inputs unexpectedly contain a label column")
    inventory = json.loads(read_member("test_inventory"))
    aliases: dict[str, str] = inventory["physical_aliases"]
    official = test["sample_token"].to_numpy(dtype=str)
    if len(official) != 6762 or set(aliases) != set(official):
        raise ValueError("Official Test12 population or alias map differs")
    raw, disagreements = _geometry(test)
    position = {token: i for i, token in enumerate(official)}
    canonical = sorted(set(aliases.values()), key=position.__getitem__)
    rows: dict[str, dict[str, Any]] = {}
    for _, data, meta in iter_diagnostics():
        token = str(meta["sample_token"])
        if meta.get("private_targets_read") is not False:
            raise ValueError("Diagnostic does not certify label-free inference")
        arrays = load_npz_bytes(data)
        states = arrays["PHASE_ONLY_EVENT_S7_TRAIN40.input_0"]
        rows[token] = {
            "features": np.concatenate((states[0, 160:170], states[0, 330:340])),
            "width": arrays["PHASE_ONLY_EVENT_S7_TRAIN40.input_1"][0],
            "released_ttc": float(arrays["PHASE_ONLY_EVENT_S7_TRAIN40.output_ttc"][0]),
            "event_valid": int(arrays["E_CTX_MATCHED.input_2"].sum()),
            "rgb_valid": int(arrays["R_CTX.input_2"].sum()),
            "prediction": float(meta["prediction"]),
        }
    if set(rows) != set(canonical):
        raise ValueError("Diagnostics do not cover exactly the canonical physical queries")
    index = np.asarray([position[t] for t in canonical], dtype=np.int64)
    arrays_out = {
        "official_token": official,
        "official_canonical": np.asarray([aliases[t] for t in official]),
        "sample_token": np.asarray(canonical),
        "group_id": test["sequence_id"].to_numpy(dtype=str)[index],
        "features": np.stack([rows[t]["features"] for t in canonical]).astype(np.float32),
        "rgb_width_logits": np.stack([rows[t]["width"] for t in canonical]).astype(np.float32),
        "released_ttc": np.asarray([rows[t]["released_ttc"] for t in canonical]),
        "fragment_prediction": np.asarray([rows[t]["prediction"] for t in canonical]),
        "event_valid": np.asarray([rows[t]["event_valid"] for t in canonical]),
        "rgb_valid": np.asarray([rows[t]["rgb_valid"] for t in canonical]),
        "geometry_raw": raw[index],
        "geometry_g": geometry.learned_inputs(raw[index]),
        "official_geometry_raw": raw,
    }
    receipt = {
        "official_tokens": len(official),
        "physical_queries": len(canonical),
        "aliases": int(len(official) - len(canonical)),
        "box_disagreements": disagreements,
        "private_labels_read": False,
    }
    return arrays_out, receipt


def main() -> None:
    """Write the two work NPZ files and the data receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train, train_receipt = prepare_train(root)
    test, test_receipt = prepare_test()
    for name, value in (*train.items(), *test.items()):
        if np.asarray(value).dtype == object:
            raise TypeError(f"Work array {name} must not be an object array")
    np.savez_compressed(work / "train_inputs.npz", **train)
    np.savez_compressed(work / "test_inputs.npz", **test)
    receipt = {
        "schema": "phase_geometry_data_receipt_v1",
        "sources": SOURCES,
        "files": {k: {**v, "verified": True} for k, v in FILES.items() if k != "submitted_zip"},
        "train": train_receipt,
        "test": test_receipt,
        "outputs": {
            name: sha256_file(work / name) for name in ("train_inputs.npz", "test_inputs.npz")
        },
        "geometry_source_sha256": sha256_file(Path(geometry.__file__)),
        "data_source_sha256": sha256_file(Path(__file__)),
        "seconds": time.time() - started,
    }
    (work / "DATA_RECEIPT.json").write_text(
        json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({"train": train_receipt, "test": test_receipt}), indent=1)[:3000])


if __name__ == "__main__":
    main()
