"""Attach measured references and a bounded research source snapshot to the release."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from release.huggingface.prepare_release import ROOT, STAGE, digest, manifest


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    record = next(
        json.loads(line)
        for line in (ROOT / "output/phase_submission/replay/all_rows.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line)["identity"]["sequence_id"] == "CCRm-low"
        and json.loads(line)["identity"]["row_index"] == 10
    )
    fields = ["original_ttc_seconds", "rgb_ttc_seconds", "event_ttc_seconds", "ttc_seconds"]
    archived = dict(
        zip(fields, [*record["native_ttc_seconds"], record["prediction_ttc_seconds"]], strict=True)
    )
    real = {
        device: json.loads((STAGE / f"real_{device}.json").read_text())
        for device in ["cpu", "cuda"]
    }
    synth = {
        device: json.loads((STAGE / f"adapted_{device}.json").read_text())
        for device in ["cpu", "cuda"]
    }
    for device in real:
        for key in fields:
            np.testing.assert_allclose(real[device][key], archived[key], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(synth[device][key], synth["cpu"][key], rtol=1e-5, atol=1e-5)
    snapshot = STAGE / "research_source.zip"
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [
        ROOT / name
        for name in tracked
        if name
        and (
            name.startswith(
                ("src/", "operational/", "scripts/", "configs/", "data/manifests/", "data/splits/")
            )
            and not name.startswith("configs/local/")
            and Path(name).suffix
            in {".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".md", ".ps1", ".sh"}
            or name in {"pyproject.toml", "uv.lock", "LICENSE", "CITATION.cff"}
        )
    ]
    paths += [p for p in (ROOT / "release/huggingface").rglob("*") if p.suffix in {".py", ".md"}]
    paths.append(ROOT / "docs/detection_local_20261029/evidence/deformable_detr_LICENSE.txt")
    forbidden = re.compile(
        rb"hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|"
        rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"
    )
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            if path.is_file():
                raw = path.read_bytes()
                if forbidden.search(raw):
                    raise ValueError(f"Credential-like data found in {path.relative_to(ROOT)}")
                z.writestr(path.relative_to(ROOT).as_posix(), raw)
    lock = subprocess.check_output(
        [
            "uv",
            "pip",
            "freeze",
            "--python",
            str(ROOT / "output/huggingface-check/Scripts/python.exe"),
        ],
        text=True,
    )
    for name in ["PHASE-TTC", "PHASE-TTC-Adapt"]:
        target = STAGE / name
        adapted = name.endswith("Adapt")
        (target / "reproduction").mkdir(exist_ok=True)
        shutil.copy2(snapshot, target / "reproduction/research_source.zip")
        (target / "requirements-cpu-tested.txt").write_text(
            "--extra-index-url https://download.pytorch.org/whl/cpu\n" + lock, encoding="utf-8"
        )
        for filename, values in [
            ("synthetic_expected.json", {k: synth["cpu"][k] for k in fields}),
            ("real_CCRm_low_row10_expected.json", archived.copy()),
        ]:
            if not adapted:
                values["ttc_seconds"] = values["original_ttc_seconds"]
            write_json(
                target / "examples" / filename, {"outputs": values, "rtol": 1e-5, "atol": 1e-5}
            )
        archived_variant = archived.copy()
        real_variant = {device: {k: real[device][k] for k in fields} for device in real}
        if not adapted:
            archived_variant["ttc_seconds"] = archived_variant["original_ttc_seconds"]
            for values in real_variant.values():
                values["ttc_seconds"] = values["original_ttc_seconds"]
        validation = {
            "date": "2026-11-11",
            "cpu_environment": (
                "Python 3.12.13, torch 2.10.0+cpu, Windows; requirements-cpu-tested.txt"
            ),
            "cuda_environment": "torch 2.10.0+cu128, Windows",
            "synthetic_cpu_cuda_agreement": "passed rtol=1e-5 atol=1e-5",
            "real_query": record["identity"],
            "archived_reference": archived_variant,
            "real_cpu": real_variant["cpu"],
            "real_cuda": real_variant["cuda"],
            "real_cpu_max_abs_difference_s": max(
                abs(real_variant["cpu"][k] - archived_variant[k]) for k in fields
            ),
            "raw_example_distribution": "not included; extract from your own benchmark copy",
            "source_archive_sha256": digest(snapshot),
            "full_original_retraining": "not rerun",
        }
        write_json(target / "VALIDATION.json", validation)
        if adapted:
            shutil.copy2(
                ROOT / "docs/research_log_20261031_sign_safe/evidence/evaluator_audit.json",
                target / "evaluation/adaptation_evaluator_audit.json",
            )
            import torch

            refit = STAGE / "refit-validated/training"
            if (refit / "REPORT.json").exists():
                errors = {}
                for seed in ["7", "13", "21"]:
                    rebuilt = torch.load(
                        refit / "final" / seed / "checkpoint.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["state_dict"]
                    original = torch.load(
                        target / "model/router/final" / seed / "checkpoint.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["state_dict"]
                    errors[seed] = max(
                        float((value - original[key]).abs().max()) for key, value in rebuilt.items()
                    )
                    if errors[seed] != 0:
                        raise ValueError("Reproduced router weights differ")
                validation["router_refit"] = {
                    "state_dict_max_absolute_difference": errors,
                    "fit_rows": 2608,
                    "groups": 25,
                    "updates_per_seed": 300,
                    "scripted_export_parity_rows": 2608,
                    "purpose": "Reproduction check; distributed original checkpoints unchanged",
                }
                write_json(target / "VALIDATION.json", validation)
        for p in target.rglob("*"):
            if (
                p.is_file()
                and p.suffix in {".py", ".json", ".yaml", ".yml", ".txt", ".md"}
                and "__pycache__" not in p.parts
            ):
                if forbidden.search(p.read_bytes()):
                    raise ValueError(f"Credential-like content in {p}")
        manifest(target)
        print(
            json.dumps(
                {
                    "model": name,
                    "files": len(json.loads((target / "MANIFEST.json").read_text())),
                    "manifest_sha256": digest(target / "MANIFEST.json"),
                    "source_zip_mib": snapshot.stat().st_size / 2**20,
                }
            )
        )


if __name__ == "__main__":
    main()
