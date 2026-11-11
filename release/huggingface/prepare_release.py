"""Build independent, checksum-bound Hugging Face staging directories."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "output/huggingface"
MODEL = ROOT / "output/phase_submission/FINAL_MODEL"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure() -> set[Path]:
    """Include transitive local imports, including package initializers."""
    pending = [
        "operational.phase_robust.bundle",
        "operational.phase_submission.runtime",
        "scripts.train_phase_submission",
        "operational.phase_startup.runtime",
        "operational.evttc_competition.prepare",
        "operational.evttc_competition.readers",
        "operational.evttc_competition.geometry",
    ]
    seen, paths = set(), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        base = ROOT / ("src" if name.startswith("e_jepa_ttc") else "")
        rel = Path(*name.split("."))
        path = base / rel.with_suffix(".py")
        package = name.rsplit(".", 1)[0] if "." in name else ""
        if not path.exists():
            path = base / rel / "__init__.py"
            package = name
        if not path.is_file():
            continue
        paths.add(path)
        parts = name.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
                    module = ".".join([*prefix, module]).strip(".")
                pending.append(module)
                pending.extend(module + "." + a.name for a in node.names if a.name != "*")
    return paths


def synthetic_example(path: Path) -> None:
    """A deterministic expanding textured shape for execution checks, not accuracy."""
    yy, xx = np.mgrid[:128, :128]
    observations, arrays = [], {}
    for modality in ("event", "rgb"):
        for i, anchor in enumerate((350000, 450000, 550000, 650000)):
            clocks = [anchor - 200000, anchor - 100000, anchor]
            frames = []
            for clock in clocks:
                radius = 18 + 15 * clock / 1e6
                inside = (abs(xx - 64) < radius) & (abs(yy - 64) < radius)
                if modality == "rgb":
                    texture = ((xx // 5 + yy // 5) % 2) * 0.45 + 0.35
                    plane = np.where(inside, texture, 0.1).astype(np.float32)
                    frame = np.stack([plane, plane * 0.9, plane * 0.8])
                else:
                    edge = (abs(np.maximum(abs(xx - 64), abs(yy - 64)) - radius) < 2).astype(
                        np.float32
                    )
                    frame = np.stack(
                        [edge * (j + 1) / 5 for j in range(5)]
                        + [edge * 0.1] * 5
                        + [np.full_like(edge, 5), np.full_like(edge, 7)]
                    )
                frames.append(frame)
            key = f"{modality}_{i}"
            arrays[key] = np.stack(frames).astype(np.float32)
            observations.append(
                {
                    "modality": modality,
                    "pixels": key,
                    "timestamps_us": clocks,
                    "first_sensor_ts_us": clocks[0] - (100000 if modality == "event" else 0),
                    "anchor_us": anchor,
                    "available_us": anchor,
                }
            )
    box_clocks = list(range(50000, 650001, 50000))
    boxes = [
        [
            64 - (18 + 15 * t / 1e6),
            64 - (18 + 15 * t / 1e6),
            64 + (18 + 15 * t / 1e6),
            64 + (18 + 15 * t / 1e6),
        ]
        for t in box_clocks
    ]
    metadata = {
        "schema": "phase_ttc_prepared_input_v1",
        "query_us": 650000,
        "observations": observations,
        "boxes_xyxy": boxes,
        "box_timestamps_us": box_clocks,
    }
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)


def manifest(root: Path) -> None:
    """Seal distributable files, excluding validation outputs and local caches."""
    values = {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.name != "MANIFEST.json"
        and "__pycache__" not in p.parts
        and ".cache" not in p.parts
        and not p.name.endswith(".pyc")
    }
    (root / "MANIFEST.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    paths = source_closure()
    paths.add(ROOT / "configs/campaign/efficient_context_v1.json")
    model_meta = json.loads((MODEL / "MODEL.json").read_text())
    for name in model_meta["implementation"]:
        paths.add(ROOT / name)
    bindings = {}
    # Recover the exact historically bound source bytes in release copies only.
    for name, expected in model_meta["implementation"].items():
        raw = (ROOT / name).read_bytes()
        options = [
            raw,
            raw.replace(b"\r\n", b"\n"),
            raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"),
        ]
        match = next((x for x in options if hashlib.sha256(x).hexdigest() == expected), None)
        if match is None:
            raise ValueError(f"Historical implementation unavailable: {name}")
        bindings[Path(name).as_posix()] = match
    for name, variant in [("PHASE-TTC", "original"), ("PHASE-TTC-Adapt", "adapted")]:
        target = STAGE / name
        if target.exists():
            if target.resolve().parent != STAGE.resolve() or target.is_symlink():
                raise ValueError("Staging target escapes its declared build root")
            shutil.rmtree(target)
        target.mkdir(exist_ok=True)
        (target / "model").mkdir(exist_ok=True)
        if variant == "original":
            shutil.copytree(MODEL / "upstream", target / "model/upstream", dirs_exist_ok=True)
        else:
            shutil.copytree(MODEL, target / "model", dirs_exist_ok=True)
        for path in paths:
            rel = path.relative_to(ROOT)
            dest = target / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(bindings.get(rel.as_posix(), path.read_bytes()))
        for fname in ["predict.py", "extract_example.py", "train_router.py"]:
            shutil.copy2(ROOT / "release/huggingface" / fname, target / fname)
        for fname in ["README.md", "INPUT_FORMAT.md", "REPRODUCTION.md", "evaluate_saved.py"]:
            template = ROOT / "release/huggingface" / fname
            if template.exists():
                content = template.read_text(encoding="utf-8").replace("{{MODEL_NAME}}", name)
                if fname == "README.md":
                    content = content.replace(
                        "{{VARIANT}}",
                        (ROOT / "release/huggingface" / f"{variant}.md").read_text(
                            encoding="utf-8"
                        ),
                    )
                (target / fname).write_text(content, encoding="utf-8")
        evaluation = target / "evaluation"
        evaluation.mkdir(exist_ok=True)
        shutil.copy2(
            ROOT / "paper/data/official_benchmark.json", evaluation / "official_benchmark.json"
        )
        for fname in [
            "corrected_transfer_candidates.csv",
            "corrected_transfer_statistics.json",
            "transfer_reassociation.json",
        ]:
            shutil.copy2(
                ROOT / "paper/from_motion_to_time_20261102/audit_20261104" / fname,
                evaluation / fname,
            )
        if variant == "adapted":
            inputs = target / "reproduction/inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            for fname in ["POPULATION.npz", "exclusions.json"]:
                shutil.copy2(
                    ROOT / "output/phase_submission/extracted/inputs" / fname, inputs / fname
                )
        for fname in ["LICENSE", "CITATION.cff"]:
            shutil.copy2(ROOT / fname, target / fname)
        (target / "examples").mkdir(exist_ok=True)
        synthetic_example(target / "examples/synthetic_looming.npz")
        (target / "RELEASE.json").write_text(
            json.dumps(
                {
                    "schema": "phase_ttc_hf_release_v1",
                    "variant": variant,
                    "updated": "2026-11-11",
                    "code_base_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "source_snapshot": (
                        "Bundled source/ bytes are authoritative; see MANIFEST.json. "
                        "Includes release tooling added after the base commit."
                    ),
                    "model_identity": "PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF"
                    if variant == "original"
                    else "PHASE-TTC-Adapt-20261031",
                    "input_adapter": (
                        "evttc_competition_causal_adapter_v2; prepared observations, "
                        "supplied object regions"
                    ),
                    "historical_source_line_endings_preserved": True,
                    "synthetic_example": (
                        "Execution-only fixture; not benchmark data or an accuracy claim."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        (target / "requirements.txt").write_text(
            "torch==2.10.0\ntorchvision==0.25.0\nnumpy>=1.26,<3\nscipy>=1.14,<2\nPillow>=10,<13\nh5py>=3.11,<4\nhdf5plugin>=5,<8\npandas>=2.2,<4\npyarrow>=17\nPyYAML>=6\npsutil>=7\npydantic>=2.7,<3\nomegaconf>=2.3,<3\nhuggingface_hub>=1.0,<3\njsonschema>=4.26\ngdown>=5\nscikit-learn>=1.5,<2\n",
            encoding="utf-8",
        )
        manifest(target)
        print(
            json.dumps(
                {
                    "model": name,
                    "source_files": len(paths),
                    "size_mib": round(
                        sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 2**20, 2
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
