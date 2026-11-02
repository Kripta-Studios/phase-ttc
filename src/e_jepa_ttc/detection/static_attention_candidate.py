"""Isolated fixed-layout attention candidate for B1 evaluation profiling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.detection.finite_validation import require_finite_fused

HostDecision = Callable[[Tensor], bool]


def _default_host_decision(flag: Tensor) -> bool:
    return bool(flag.item())


@dataclass(frozen=True)
class StaticAttentionLayout:
    """Immutable Python/tensor layout for one fixed batch, dtype and device."""

    shapes: tuple[tuple[int, int], ...]
    sizes: tuple[int, ...]
    total_length: int
    batch_size: int
    device: torch.device
    value_dtype: torch.dtype
    index_dtype: torch.dtype
    spatial_shapes: Tensor
    level_start_index: Tensor

    @classmethod
    def create(
        cls,
        shapes: Sequence[tuple[int, int]],
        *,
        batch_size: int,
        device: torch.device | str,
        value_dtype: torch.dtype,
        index_dtype: torch.dtype = torch.long,
    ) -> StaticAttentionLayout:
        normalized = tuple((int(height), int(width)) for height, width in shapes)
        if not normalized or any(height <= 0 or width <= 0 for height, width in normalized):
            raise ValueError("Static attention shapes must be nonempty and positive")
        if batch_size < 1 or index_dtype not in {torch.int32, torch.int64}:
            raise ValueError("Static attention batch/index dtype is invalid")
        if not value_dtype.is_floating_point:
            raise ValueError("Static attention value dtype must be floating point")
        resolved_device = torch.device(device)
        sizes = tuple(height * width for height, width in normalized)
        starts_list = [0]
        for size in sizes[:-1]:
            starts_list.append(starts_list[-1] + size)
        starts = tuple(starts_list)
        spatial_shapes = torch.tensor(normalized, dtype=index_dtype, device=resolved_device)
        level_start_index = torch.tensor(starts, dtype=index_dtype, device=resolved_device)
        return cls(
            normalized,
            sizes,
            sum(sizes),
            batch_size,
            resolved_device,
            value_dtype,
            index_dtype,
            spatial_shapes,
            level_start_index,
        )


@dataclass(frozen=True)
class _AttentionCall:
    spatial_shapes: Tensor
    level_start_index: Tensor
    sampling_locations: Tensor


_ACTIVE_GUARD: ContextVar[StaticAttentionForwardGuard | None] = ContextVar(
    "static_attention_forward_guard", default=None
)


class StaticAttentionForwardGuard:
    """Accumulate layout/finite flags and make one host decision after a forward."""

    def __init__(
        self,
        layout: StaticAttentionLayout,
        *,
        host_decision: HostDecision = _default_host_decision,
    ) -> None:
        self.layout = layout
        self.host_decision = host_decision
        self._token: Token[StaticAttentionForwardGuard | None] | None = None
        self._calls: list[_AttentionCall] = []
        self._flags: list[Tensor] = []

    def __enter__(self) -> StaticAttentionForwardGuard:
        if self._token is not None:
            raise RuntimeError("Static attention guard cannot be nested or re-entered")
        self._calls.clear()
        self._flags.clear()
        self._token = _ACTIVE_GUARD.set(self)
        return self

    def __exit__(self, error_type: object, error: object, traceback: object) -> None:
        try:
            if error_type is None:
                self.finalize()
        finally:
            assert self._token is not None
            _ACTIVE_GUARD.reset(self._token)
            self._token = None

    def _validate_metadata(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> None:
        layout = self.layout
        if value.ndim != 4:
            raise ValueError("Expected value [B,S,H,D]")
        batch, length, heads, _ = value.shape
        if (
            batch != layout.batch_size
            or length != layout.total_length
            or value.device != layout.device
            or value.dtype != layout.value_dtype
        ):
            raise ValueError("Value tensor differs from the fixed attention layout key")
        if (
            spatial_shapes.shape != layout.spatial_shapes.shape
            or level_start_index.shape != layout.level_start_index.shape
            or spatial_shapes.device != layout.device
            or level_start_index.device != layout.device
            or spatial_shapes.dtype != layout.index_dtype
            or level_start_index.dtype != layout.index_dtype
        ):
            raise ValueError("Runtime attention index metadata differs from fixed layout metadata")
        if (
            sampling_locations.ndim != 6
            or sampling_locations.shape[0] != batch
            or sampling_locations.shape[2] != heads
            or sampling_locations.shape[3] != len(layout.shapes)
            or sampling_locations.shape[-1] != 2
            or sampling_locations.device != layout.device
            or sampling_locations.dtype != layout.value_dtype
            or attention_weights.shape != sampling_locations.shape[:-1]
            or attention_weights.device != layout.device
            or attention_weights.dtype != layout.value_dtype
        ):
            raise ValueError("Runtime attention sampling metadata differs from fixed layout key")

    def apply(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        if self._token is None:
            raise RuntimeError("Static attention call requires an active forward guard")
        self._validate_metadata(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        self._calls.append(_AttentionCall(spatial_shapes, level_start_index, sampling_locations))
        self._flags.extend(
            (
                (spatial_shapes == self.layout.spatial_shapes).all(),
                (level_start_index == self.layout.level_start_index).all(),
                torch.isfinite(sampling_locations).all(),
            )
        )
        return deformable_attention_static(
            value, self.layout, sampling_locations, attention_weights
        )

    def finalize(self) -> None:
        if not self._calls:
            raise RuntimeError("Static attention forward executed no attention calls")
        passed = self.host_decision(torch.stack(self._flags).all())
        if passed:
            return
        for call in self._calls:
            if not torch.equal(call.spatial_shapes, self.layout.spatial_shapes):
                raise ValueError("spatial_shapes differ from the fixed attention layout")
            if not torch.equal(call.level_start_index, self.layout.level_start_index):
                raise ValueError("level_start_index must describe contiguous feature levels")
            if not torch.isfinite(call.sampling_locations).all().item():
                raise ValueError("Sampling locations must be finite before grid_sample.")
        raise RuntimeError("Static attention fused decision failed but slow validation passed")


def deformable_attention_static(
    value: Tensor,
    layout: StaticAttentionLayout,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Run the original grid-sample math/order using host-hoisted split sizes."""
    batch, _, heads, channels = value.shape
    queries = sampling_locations.shape[1]
    points = sampling_locations.shape[4]
    grids = 2 * sampling_locations - 1
    output: Tensor | None = None
    for level, (level_value, (height, width)) in enumerate(
        zip(value.split(layout.sizes, dim=1), layout.shapes, strict=True)
    ):
        feature = (
            level_value.flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, height, width)
        )
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        samples = functional.grid_sample(
            feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        weights = (
            attention_weights[:, :, :, level]
            .transpose(1, 2)
            .reshape(batch * heads, 1, queries, points)
        )
        contribution = (samples * weights).sum(-1)
        output = contribution if output is None else output + contribution
    assert output is not None
    return output.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


def static_attention_apply(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    im2col_step: int,
) -> Tensor:
    """Drop-in apply target for a hash-bound copied portable-attention source."""
    del im2col_step
    guard = _ACTIVE_GUARD.get()
    if guard is None:
        raise RuntimeError("Static attention candidate requires StaticEvaluationCandidate.forward")
    return guard.apply(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights
    )


def static_attention_layout_tensors(
    shapes: Sequence[Sequence[int]], device: torch.device | str
) -> tuple[Tensor, Tensor]:
    """Return preregistered tensors after a host-only feature-shape assertion."""
    guard = _ACTIVE_GUARD.get()
    if guard is None:
        raise RuntimeError("Static layout tensors require StaticEvaluationCandidate.forward")
    normalized = tuple((int(shape[0]), int(shape[1])) for shape in shapes)
    if normalized != guard.layout.shapes or torch.device(device) != guard.layout.device:
        raise ValueError("Fresh feature shapes/device differ from the static attention layout")
    return guard.layout.spatial_shapes, guard.layout.level_start_index


class StaticEvaluationCandidate:
    """Evaluation-only wrapper: guarded attention then fused output finite validation."""

    def __init__(
        self,
        layout: StaticAttentionLayout,
        *,
        host_decision: HostDecision = _default_host_decision,
    ) -> None:
        self.guard = StaticAttentionForwardGuard(layout, host_decision=host_decision)

    def forward(
        self, producer: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        with self.guard:
            output = producer(*args, **kwargs)
        require_finite_fused(output)
        return output
