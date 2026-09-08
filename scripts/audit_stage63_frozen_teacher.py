"""Audit the existing frozen teacher locally, with no download or fitting."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, verify  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings  # noqa: E402
from e_jepa_ttc.data.capacity_sources import _signed  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    CompleteDinoTeacherCache,
    capacity_teacher_read_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--teacher-manifest", required=True, type=Path)
    parser.add_argument("--canonical-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--capacity-preparation",
        type=Path,
        help="Explicit signed capacity cohort for audited incidental TRAIN reads",
    )
    parser.add_argument("--capacity-preparation-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("teacher provenance evidence must not overwrite a prior audit")
    if (args.capacity_preparation is None) != (args.capacity_preparation_sha256 is None):
        raise ValueError("capacity audit requires preparation path and exact SHA256 together")
    manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    if not verify_artifact_hash(manifest):
        raise ValueError("teacher manifest identity invalid")
    model = AutoModel.from_pretrained(
        str(args.model_root), local_files_only=True, trust_remote_code=False
    ).eval()
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    if digest.hexdigest() != manifest["teacher"]["weights_sha256"]:
        raise ValueError("local pretrained teacher weights differ from recorded teacher")
    code_commit = manifest["code_identity"]["git_commit"]
    source = subprocess.check_output(
        ["git", "show", f"{code_commit}:scripts/materialize_dinov3_relational_teacher.py"],
        cwd=args.reference_root,
    )
    tree = ast.parse(source)
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if any(
        isinstance(call, ast.Attribute) and call.attr in {"step", "backward", "fit", "train"}
        for call in calls
    ):
        raise ValueError(
            "teacher materializer contains a fitting operation requiring further audit"
        )
    if not any(isinstance(call, ast.Attribute) and call.attr == "eval" for call in calls):
        raise ValueError("teacher materializer does not enter evaluation mode")
    if not any(isinstance(call, ast.Attribute) and call.attr == "no_grad" for call in calls):
        raise ValueError("teacher materializer lacks gradient-disabled inference")
    read_contract = None
    if args.capacity_preparation is not None:
        preparation = binding(args.capacity_preparation)
        if preparation["sha256"] != args.capacity_preparation_sha256:
            raise ValueError("capacity preparation hash differs from explicit audit authority")
        read_contract = capacity_teacher_read_contract(
            manifest,
            manifest_sha256=compute_file_hash(str(args.teacher_manifest)),
            preparation=preparation,
            materializer_source=source,
        )
        selected = _signed(preparation)["selected_metadata"]
        if verify(selected) != args.canonical_metadata.resolve():
            raise ValueError("capacity audit metadata differs from signed selected TRAIN metadata")
    canonical = pd.read_csv(
        args.canonical_metadata,
        dtype={
            "sample_token": str,
            "track_id": str,
            "sequence_id": str,
        },
    )
    if canonical.sample_token.duplicated().any():
        raise ValueError("canonical teacher audit metadata contains duplicate tokens")
    teacher = CompleteDinoTeacherCache.open_verified(
        args.teacher_manifest,
        expected_artifact_sha256=manifest["artifact_sha256"],
        expected_manifest_sha256=compute_file_hash(str(args.teacher_manifest)),
        allowed_sample_tokens=set(canonical.sample_token),
        capacity_read_contract=read_contract,
    )
    if teacher.source_teacher_row_count != len(canonical):
        raise ValueError("teacher audit must cover the complete canonical token universe")
    for token, track_id, sequence_id in canonical[
        ["sample_token", "track_id", "sequence_id"]
    ].itertuples(index=False, name=None):
        track, sequence, *_ = teacher[str(token)]
        if track != str(track_id) or sequence != str(sequence_id):
            raise ValueError("teacher row identities differ from canonical train universe")
    result = sign_stage63_65_artifact(
        {
            "artifact_type": "stage63_frozen_representation_teacher_provenance_v1",
            "status": "passed",
            "training_executed": False,
            "download_executed": False,
            "preexisting_frozen_teacher": True,
            "fold_fitted_teacher": False,
            "teacher_artifact_sha256": manifest["artifact_sha256"],
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "teacher_manifest_sha256": compute_file_hash(str(args.teacher_manifest)),
            "teacher_weights_sha256": digest.hexdigest(),
            "local_model_root": str(args.model_root.resolve()),
            "materializer_commit": code_commit,
            "materializer_source_sha256": hashlib.sha256(source).hexdigest(),
            "materializer_ast_no_fit_calls": True,
            "verified_tokens": len(teacher.tokens()),
            "verified_tokens_sha256": hash_sorted_token_strings(list(teacher.tokens())),
            "capacity_read_contract": read_contract,
            "ttc_labels_read_during_materialization": manifest["scope"]["ttc_labels_read"],
            "scope": "frozen external teacher; original web-pretraining overlap is not audited",
        },
        evidence_type="frozen_teacher_dependency",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "verified_tokens": result["verified_tokens"]}))


if __name__ == "__main__":
    main()
