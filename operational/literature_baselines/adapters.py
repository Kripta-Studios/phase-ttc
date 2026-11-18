"""Load each pinned third-party model from ``PHASE_TTC_BASELINES_ROOT`` and run one pair.

Third-party source stays outside the repository; this module only inserts the pinned clone
into ``sys.path`` (after its code was reviewed), verifies weight SHA-256 values and loads
weights with ``torch.load(weights_only=True)``.  Every predictor maps two uint8 RGB crops
(image 1 = query frame t, image 2 = frame t - delta) to either a dense ``rho`` map
(``log(s(t)/s(t-delta))`` in image-1 coordinates) or a flow field image 1 -> image 2.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .common import CROP_SIZE

ENV_BASELINES = "PHASE_TTC_BASELINES_ROOT"

WEIGHTS: dict[str, dict[str, str]] = {
    "OEXP": {
        "path": "weights/expansion/exp-kitti-trainval.pth",
        "sha256": "56ca4d00fa43ddc12d045f3af130bcb345424f7ee4f7065c79ee50e98ca052a2",
    },
    "OEXP_ROBUST": {
        "path": "weights/expansion/robust.pth",
        "sha256": "3d062d2615f587c5647d8c177f8dc1aed01008c97b728e37a7590fdcad4bac11",
    },
    "SFPP": {
        "path": "weights/scaleflowpp/ResScale_KITTI160FT.pth",
        "sha256": "33351346c7937ae8746ff745de0e165f72d41b545f680615e6ece1dfd666d848",
    },
    "RAFT_AFF": {
        "path": "torch_home/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
        "sha256": "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322",
    },
    "RESNET18_INIT": {
        "path": "torch_home/hub/checkpoints/resnet18-f37072fd.pth",
        "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    },
}
CLONES = {
    "OEXP": ("src/expansion", "dc1bc3f4a5ea8db32d182c2647dd6054df852382"),
    "SFPP": ("src/CSCV", "8406de13947478dc0fca0ca346d842b15455efc1"),
}
# Buffers rebuilt from the input size (not learned) and therefore absent from checkpoints.
OEXP_SIZE_BUFFERS = ("flowx", "flowy", "grid")

Array = NDArray[Any]


def baselines_root() -> Path:
    """Third-party root named by ``PHASE_TTC_BASELINES_ROOT``."""
    value = os.environ.get(ENV_BASELINES, "").strip()
    if not value or not Path(value).is_dir():
        raise RuntimeError(
            f"Set {ENV_BASELINES} to the directory holding src/, weights/ and torch_home/ "
            "(see docs/sota_20261118/literature_baselines/REPORT.md)."
        )
    return Path(value)


def verified_weight(name: str) -> Path:
    """Pinned weight file, refusing any byte change."""
    spec = WEIGHTS[name]
    path = baselines_root() / spec["path"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"{name} weights differ from the pinned SHA-256: {path}")
    return path


def clone_path(name: str) -> Path:
    """Pinned third-party clone, refusing any other commit."""
    relative, commit = CLONES[name]
    path = baselines_root() / relative
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != commit:
        raise ValueError(f"{path} is at {head}, expected {commit}")
    return path


@dataclass
class Predictor:
    """A loaded model: ``run(img1, img2)`` returns ``{"rho": [H,W]}`` or ``{"flow": [2,H,W]}``."""

    name: str
    run: Callable[[Array, Array], dict[str, Array]]
    provenance: dict[str, Any] = field(default_factory=dict)


def _strip_module(state: dict[str, Any]) -> dict[str, Any]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def _numpy_safe_globals() -> list[Any]:
    import numpy.core.multiarray as legacy  # numpy>=2 shim with the pickled name

    return [
        (legacy._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]


def load_oexp(
    device: torch.device,
    size: int = CROP_SIZE,
    *,
    weights: str = "OEXP",
    md: tuple[int, ...] = (4, 4, 4, 4, 4),
    fac: float = 1.0,
    height: int | None = None,
) -> Predictor:
    """VCN+expansion (Yang & Ramanan 2020); ``rho = dchange2 = log(Z2/Z1)``."""
    clone = clone_path("OEXP")
    sys.path.insert(0, str(clone))
    from models.VCN_exp import VCN  # type: ignore[import-not-found]

    width = size
    rows = height if height is not None else size
    if width % 64 or rows % 64:
        raise ValueError("VCN needs input sides divisible by 64")
    model = VCN([1, width, rows], md=list(md), fac=fac, exp_unc=(weights == "OEXP_ROBUST"))
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        checkpoint = torch.load(verified_weight(weights), map_location="cpu", weights_only=True)
    state = _strip_module(checkpoint["state_dict"])
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected OEXP keys: {result.unexpected_keys[:5]}")
    loaded = {id(p) for k, p in model.state_dict(keep_vars=True).items() if k in state}
    unloaded = sorted(k for k, p in model.state_dict(keep_vars=True).items() if id(p) not in loaded)
    if any(k.rsplit(".", 1)[-1] not in OEXP_SIZE_BUFFERS for k in unloaded):
        raise ValueError(f"OEXP parameters missing from the checkpoint: {unloaded[:5]}")
    mean = np.asarray(checkpoint["mean_L"]).mean(0).astype(np.float64)
    model = model.to(device).eval()

    def run(image1: Array, image2: Array) -> dict[str, Array]:
        batch = []
        for image in (image1, image2):
            bgr = image[:, :, ::-1].astype(np.float64) / 255.0 - mean[None, None, :]
            batch.append(np.transpose(bgr, (2, 0, 1)))
        tensor = torch.from_numpy(np.stack(batch)).float().to(device)
        with torch.no_grad():
            _, _, logmid, _ = model(tensor)
        return {"rho": logmid.float().cpu().numpy()}

    return Predictor(
        "OEXP" if weights == "OEXP" else weights,
        run,
        {
            "clone": str(clone.name),
            "commit": CLONES["OEXP"][1],
            "weights": WEIGHTS[weights],
            "md": list(md),
            "fac": fac,
            "unloaded_size_buffers": len(unloaded),
        },
    )


def load_sfpp(device: torch.device, *, iters: int = 6) -> Predictor:
    """ScaleFlow++ ``ResScale_AB``; ``rho = log(dchange)`` with ``dchange = Z2/Z1``."""
    clone = clone_path("SFPP")
    os.environ.setdefault("TORCH_HOME", str(baselines_root() / "torch_home"))
    verified_weight("RESNET18_INIT")
    sys.path.insert(0, str(clone / "core"))
    sys.path.insert(0, str(clone))
    from core.scale_flowpp import ResScale_AB  # type: ignore[import-not-found]

    args = types.SimpleNamespace(
        mixed_precision=False,
        small=False,
        alternate_corr=False,
        num_heads=1,
        position_only=False,
        position_and_content=False,
        start=0,
    )
    model = ResScale_AB(args)
    state = _strip_module(
        torch.load(verified_weight("SFPP"), map_location="cpu", weights_only=True)
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    def run(image1: Array, image2: Array) -> dict[str, Array]:
        h, w = image1.shape[:2]
        if h % 32 or w % 32:
            raise ValueError("ScaleFlow++ adapter expects sides divisible by 32")
        t1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to(device)
        t2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].to(device)
        with torch.no_grad():
            _, _, dchange = model(t1, t2, iters=iters, test_mode=True)
        ratio = dchange[0, 0].float().cpu().numpy().astype(np.float64)
        return {"rho": np.log(ratio)}

    return Predictor(
        "SFPP",
        run,
        {
            "clone": clone.name,
            "commit": CLONES["SFPP"][1],
            "weights": WEIGHTS["SFPP"],
            "iters": iters,
            "mixed_precision": False,
        },
    )


def load_raft(device: torch.device, *, updates: int = 12) -> Predictor:
    """torchvision RAFT-large ``C_T_SKHT_V2``; returns the flow image 1 -> image 2."""
    os.environ.setdefault("TORCH_HOME", str(baselines_root() / "torch_home"))
    verified_weight("RAFT_AFF")
    import torchvision
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()

    def run(image1: Array, image2: Array) -> dict[str, Array]:
        tensors = [
            (torch.from_numpy(i).permute(2, 0, 1).float()[None] / 255.0 * 2.0 - 1.0).to(device)
            for i in (image1, image2)
        ]
        with torch.no_grad():
            flows = model(tensors[0], tensors[1], num_flow_updates=updates)
        return {"flow": flows[-1][0].float().cpu().numpy().astype(np.float64)}

    return Predictor(
        "RAFT_AFF",
        run,
        {
            "torchvision": torchvision.__version__,
            "weights": WEIGHTS["RAFT_AFF"],
            "num_flow_updates": updates,
        },
    )


def load(name: str, device: torch.device) -> Predictor:
    """Primary configuration of a method as pre-registered."""
    if name == "OEXP":
        return load_oexp(device)
    if name == "SFPP":
        return load_sfpp(device)
    if name == "RAFT_AFF":
        return load_raft(device)
    raise ValueError(f"Unknown method {name}")
