"""Hash-bound source materializer for the isolated B1 static-attention candidate."""

# ruff: noqa: E501 -- exact frozen-source anchors intentionally retain upstream line lengths.

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

R0_RUNTIME_PREREGISTRATION_SHA256 = (
    "1964580644eaa08a998881663691d1c351532d7b9d6dbbcaee2f4b621dc0135c"
)
R0_CODE_MANIFEST_SHA256 = "36c5bd45d548e7a3315d41e95be4192511e8225434d4229e8909cdf6b2f575b7"
R0_PORTABLE_MANIFEST_SHA256 = "0a7d44d6ec24e3a247a6d347460b59041740983f0a1d3384217e6279cd77a57f"
R0_PORTABLE_ATTENTION_SHA256 = "e3a5ca88ca989e7bd10fe1f8381f7da32b83eea3063ac89b67113511b045f7fa"
R0_DET2D_SHA256 = "4706f607808b2c5fd0e2033a4f5d40d803bfc02271d08adb05f9968805304ff2"
STATIC_CANDIDATE_SHA256 = "6fa1f2dd3bb9ee5d6fd053b8365a4621125e72e25994cd606beae0ee055665eb"
FINITE_VALIDATION_SHA256 = "127caa523edd79a5abb8cc6f44d188a6eb5a9f6f50865ee6b7443933755c5bf8"
R0_SELECTED_CHECKPOINT_SHA256 = "5e42f17f4eaac802e6c83ee4265896054167a8b26f0b6e7432501355bd115996"

PORTABLE_ATTENTION = Path("src/e_jepa_ttc/detection/portable_attention.py")
DET2D = Path("portable_source/lib/models/monodgp/det2d_transformer.py")
STATIC_CANDIDATE = Path("src/e_jepa_ttc/detection/static_attention_candidate.py")
FINITE_VALIDATION = Path("src/e_jepa_ttc/detection/finite_validation.py")

_PORTABLE_IMPORT = "from torch.nn import functional\n"
_PORTABLE_BODY = """        del im2col_step  # A custom-kernel batching hint, irrelevant to grid_sample.
        sizes = spatial_shapes.prod(1)
        expected = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
        if not torch.equal(level_start_index, expected):
            raise ValueError("level_start_index must describe contiguous feature levels.")
        return deformable_attention_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
"""
_PORTABLE_REPLACEMENT = """        return static_attention_apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
"""
_DET2D_IMPORT = "from .ops.modules import MSDeformAttn, MSDeformAttn_cross, MultiheadAttention\n"
_DET2D_BODY = """        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
"""
_DET2D_REPLACEMENT = """        spatial_shapes, level_start_index = static_attention_layout_tensors(
            spatial_shapes, srcs[0].device)
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Runtime manifest path is unsafe: {value!r}")
    return path


@dataclass(frozen=True)
class StaticEvalSourceContract:
    runtime_preregistration_sha256: str = R0_RUNTIME_PREREGISTRATION_SHA256
    code_manifest_sha256: str = R0_CODE_MANIFEST_SHA256
    portable_manifest_sha256: str = R0_PORTABLE_MANIFEST_SHA256
    portable_attention_sha256: str = R0_PORTABLE_ATTENTION_SHA256
    det2d_sha256: str = R0_DET2D_SHA256
    static_candidate_sha256: str = STATIC_CANDIDATE_SHA256
    finite_validation_sha256: str = FINITE_VALIDATION_SHA256
    checkpoint_sha256: str = R0_SELECTED_CHECKPOINT_SHA256


def transform_portable_attention(source: str, expected_sha256: str) -> str:
    """Redirect only the exact drop-in apply body to the scoped static candidate."""
    if _sha256(source.encode()) != expected_sha256:
        raise ValueError("Portable-attention source SHA-256 differs")
    if source.count(_PORTABLE_IMPORT) != 1 or source.count(_PORTABLE_BODY) != 1:
        raise ValueError("Portable-attention patch markers are absent or ambiguous")
    transformed = source.replace(
        _PORTABLE_IMPORT,
        _PORTABLE_IMPORT
        + "from e_jepa_ttc.detection.static_attention_candidate import static_attention_apply\n",
        1,
    ).replace(_PORTABLE_BODY, _PORTABLE_REPLACEMENT, 1)
    compile(transformed, "portable_attention_static.py", "exec")
    return transformed


def transform_det2d(source: str, expected_sha256: str) -> str:
    """Replace per-forward index-tensor construction with host-validated static tensors."""
    if _sha256(source.encode()) != expected_sha256:
        raise ValueError("Det2D source SHA-256 differs")
    if source.count(_DET2D_IMPORT) != 1 or source.count(_DET2D_BODY) != 1:
        raise ValueError("Det2D patch markers are absent or ambiguous")
    transformed = source.replace(
        _DET2D_IMPORT,
        _DET2D_IMPORT
        + "from e_jepa_ttc.detection.static_attention_candidate import (\n"
        + "    static_attention_layout_tensors,\n"
        + ")\n",
        1,
    ).replace(_DET2D_BODY, _DET2D_REPLACEMENT, 1)
    compile(transformed, "det2d_static.py", "exec")
    return transformed


def _patch(name: Path, source: str, transformed: str) -> str:
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            transformed.splitlines(keepends=True),
            fromfile=f"a/{name.as_posix()}",
            tofile=f"b/{name.as_posix()}",
        )
    )


def materialize_static_eval_source(
    *,
    source_code_root: Path | str,
    destination: Path | str,
    checkpoint: Path | str,
    runtime_preregistration: Path | str,
    candidate_source: Path | str,
    finite_source: Path | str,
    patch_output: Path | str,
    manifest_output: Path | str,
    forbidden_destination: Path | str,
    contract: StaticEvalSourceContract | None = None,
) -> dict[str, object]:
    """Materialize a new eval-only source tree; never alter the admitted producer."""
    contract = StaticEvalSourceContract() if contract is None else contract
    source_code_root = Path(source_code_root).resolve()
    destination = Path(destination).resolve()
    checkpoint = Path(checkpoint).resolve()
    runtime_preregistration = Path(runtime_preregistration).resolve()
    candidate_source = Path(candidate_source).resolve()
    finite_source = Path(finite_source).resolve()
    patch_output = Path(patch_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    forbidden_destination = Path(forbidden_destination).resolve()
    named = {
        source_code_root,
        destination,
        patch_output,
        manifest_output,
        candidate_source,
        finite_source,
    }
    if len(named) != 6:
        raise ValueError("Static source/materialization paths must be distinct")
    if destination == forbidden_destination:
        raise ValueError("Static destination aliases the admitted producer")
    if destination.is_relative_to(source_code_root) or source_code_root.is_relative_to(destination):
        raise ValueError("Static source and destination must not contain one another")
    for artifact in (patch_output, manifest_output):
        if artifact.is_relative_to(source_code_root) or artifact.is_relative_to(destination):
            raise ValueError("Static patch/manifest must stay outside source and destination")
    if destination.exists() or patch_output.exists() or manifest_output.exists():
        raise FileExistsError("Use new static destination, patch and manifest paths")
    if not source_code_root.is_dir():
        raise FileNotFoundError(f"Static source root is absent: {source_code_root}")
    for path in (checkpoint, runtime_preregistration, candidate_source, finite_source):
        if not path.is_file():
            raise FileNotFoundError(f"Static materialization input is absent: {path}")
    if _file_sha256(runtime_preregistration) != contract.runtime_preregistration_sha256:
        raise ValueError("Runtime preregistration SHA-256 differs")
    if _file_sha256(candidate_source) != contract.static_candidate_sha256:
        raise ValueError("Static candidate source SHA-256 differs")
    if _file_sha256(finite_source) != contract.finite_validation_sha256:
        raise ValueError("Finite-validation source SHA-256 differs")
    if _file_sha256(checkpoint) != contract.checkpoint_sha256:
        raise ValueError("Selected checkpoint SHA-256 differs")

    preregistration = json.loads(runtime_preregistration.read_text(encoding="utf-8"))
    if preregistration.get("schema") != "eap_detection_campaign_runtime_preregistration_v1":
        raise ValueError("Runtime preregistration schema differs")
    declared_code = preregistration.get("code_sha256")
    declared_portable = preregistration.get("portable_source_sha256")
    if not isinstance(declared_code, dict) or not isinstance(declared_portable, dict):
        raise ValueError("Runtime source manifests are malformed")
    if (
        _canonical_sha256(declared_code) != contract.code_manifest_sha256
        or preregistration.get("code_sha256_manifest") != contract.code_manifest_sha256
        or _canonical_sha256(declared_portable) != contract.portable_manifest_sha256
        or preregistration.get("portable_source_sha256_manifest")
        != contract.portable_manifest_sha256
    ):
        raise ValueError("Runtime source manifest identity differs")
    for relative, expected in declared_code.items():
        path = source_code_root / _relative(relative)
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"Declared code source differs: {relative}")
    portable_root = source_code_root / "portable_source"
    if _tree_hashes(portable_root) != declared_portable:
        raise ValueError("Portable source tree differs from runtime preregistration")

    portable_path = source_code_root / PORTABLE_ATTENTION
    det2d_path = source_code_root / DET2D
    portable_source = portable_path.read_text(encoding="utf-8")
    det2d_source = det2d_path.read_text(encoding="utf-8")
    portable_transformed = transform_portable_attention(
        portable_source, contract.portable_attention_sha256
    )
    det2d_transformed = transform_det2d(det2d_source, contract.det2d_sha256)

    destination.mkdir(parents=True)
    for relative in declared_code:
        relative_path = _relative(relative)
        source = source_code_root / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(
        portable_root,
        destination / "portable_source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (destination / PORTABLE_ATTENTION).write_text(
        portable_transformed, encoding="utf-8", newline=""
    )
    (destination / DET2D).write_text(det2d_transformed, encoding="utf-8", newline="")
    shutil.copy2(candidate_source, destination / STATIC_CANDIDATE)
    shutil.copy2(finite_source, destination / FINITE_VALIDATION)

    patch = _patch(PORTABLE_ATTENTION, portable_source, portable_transformed) + _patch(
        DET2D, det2d_source, det2d_transformed
    )
    if not patch:
        raise ValueError("Static source patch unexpectedly empty")
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with patch_output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(patch)
    output_tree = _tree_hashes(destination)
    manifest: dict[str, object] = {
        "schema": "eap_static_attention_eval_source_v1",
        "variant": "monodgp_portable_static_attention_eval_b1_v1",
        "source_code_root": str(source_code_root),
        "destination": str(destination),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.checkpoint_sha256,
        "runtime_preregistration_sha256": _file_sha256(runtime_preregistration),
        "input_code_manifest_sha256": contract.code_manifest_sha256,
        "input_portable_manifest_sha256": contract.portable_manifest_sha256,
        "input_portable_attention_sha256": contract.portable_attention_sha256,
        "input_det2d_sha256": contract.det2d_sha256,
        "static_candidate_sha256": contract.static_candidate_sha256,
        "finite_validation_sha256": contract.finite_validation_sha256,
        "output_portable_attention_sha256": _file_sha256(destination / PORTABLE_ATTENTION),
        "output_det2d_sha256": _file_sha256(destination / DET2D),
        "output_tree_sha256": _canonical_sha256(output_tree),
        "output_tree_file_count": len(output_tree),
        "patch_sha256": _file_sha256(patch_output),
        "state_dict_module_names_changed": False,
        "scope": "EVAL_B1_ONLY",
        "full_model_gpu_parity": "PENDING",
        "cuda_graph_profile": "PENDING",
        "status": "MATERIALIZED_NOT_EXECUTED",
    }
    manifest["manifest_canonical_sha256"] = _canonical_sha256(manifest)
    with manifest_output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest
