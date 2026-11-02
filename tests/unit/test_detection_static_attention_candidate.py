"""CPU fixtures for the evaluation-only static attention candidate."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from e_jepa_ttc.detection.portable_attention import deformable_attention_pytorch
from e_jepa_ttc.detection.static_attention_candidate import (
    StaticAttentionForwardGuard,
    StaticAttentionLayout,
    StaticEvaluationCandidate,
    static_attention_apply,
    static_attention_layout_tensors,
)


def inputs(
    *, batch: int = 1, dtype: torch.dtype = torch.float64
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(17)
    shapes = torch.tensor([[2, 2], [1, 2]], dtype=torch.long)
    starts = torch.tensor([0, 4], dtype=torch.long)
    value = torch.randn(batch, 6, 2, 3, generator=generator, dtype=dtype)
    locations = torch.rand(batch, 3, 2, 2, 2, 2, generator=generator, dtype=dtype)
    weights = torch.rand(batch, 3, 2, 2, 2, generator=generator, dtype=dtype)
    return value, shapes, starts, locations, weights


def layout(*, batch: int = 1, dtype: torch.dtype = torch.float64) -> StaticAttentionLayout:
    return StaticAttentionLayout.create(
        ((2, 2), (1, 2)), batch_size=batch, device="cpu", value_dtype=dtype
    )


def test_static_math_and_gradients_are_exactly_reference_order() -> None:
    value, shapes, starts, locations, weights = inputs()
    eager_value = value.clone().requires_grad_(True)
    eager_locations = locations.clone().requires_grad_(True)
    eager_weights = weights.clone().requires_grad_(True)
    expected = deformable_attention_pytorch(eager_value, shapes, eager_locations, eager_weights)
    expected.square().sum().backward()

    static_value = value.clone().requires_grad_(True)
    static_locations = locations.clone().requires_grad_(True)
    static_weights = weights.clone().requires_grad_(True)
    with StaticAttentionForwardGuard(layout()) as guard:
        actual = guard.apply(static_value, shapes, starts, static_locations, static_weights)
    actual.square().sum().backward()

    assert torch.equal(actual, expected)
    assert static_value.grad is not None and eager_value.grad is not None
    assert static_locations.grad is not None and eager_locations.grad is not None
    assert static_weights.grad is not None and eager_weights.grad is not None
    assert torch.equal(static_value.grad, eager_value.grad)
    assert torch.equal(static_locations.grad, eager_locations.grad)
    assert torch.equal(static_weights.grad, eager_weights.grad)


def test_many_attention_calls_make_one_host_decision() -> None:
    value, shapes, starts, locations, weights = inputs()
    decisions: list[bool] = []

    def decide(flag: Tensor) -> bool:
        result = bool(flag.item())
        decisions.append(result)
        return result

    with StaticAttentionForwardGuard(layout(), host_decision=decide) as guard:
        first = guard.apply(value, shapes, starts, locations, weights)
        second = guard.apply(value, shapes, starts, locations, weights)
    assert torch.equal(first, second)
    assert decisions == [True]


def test_nonfinite_sampling_uses_actionable_slow_path() -> None:
    value, shapes, starts, locations, weights = inputs()
    locations[0, 1, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Sampling locations must be finite"):
        with StaticAttentionForwardGuard(layout()) as guard:
            guard.apply(value, shapes, starts, locations, weights)


def test_wrong_level_start_fails_after_single_consolidated_decision() -> None:
    value, shapes, starts, locations, weights = inputs()
    wrong = starts.clone()
    wrong[1] = 3
    decisions = 0

    def decide(flag: Tensor) -> bool:
        nonlocal decisions
        decisions += 1
        return bool(flag.item())

    with pytest.raises(ValueError, match="level_start_index"):
        with StaticAttentionForwardGuard(layout(), host_decision=decide) as guard:
            guard.apply(value, shapes, wrong, locations, weights)
    assert decisions == 1


@pytest.mark.parametrize(
    ("batch", "dtype", "match"),
    [(2, torch.float64, "Value tensor"), (1, torch.float32, "Value tensor")],
)
def test_fixed_batch_and_dtype_keys_fail_closed(batch: int, dtype: torch.dtype, match: str) -> None:
    value, shapes, starts, locations, weights = inputs(batch=batch, dtype=dtype)
    with pytest.raises(ValueError, match=match):
        with StaticAttentionForwardGuard(layout()) as guard:
            guard.apply(value, shapes, starts, locations, weights)


def test_drop_in_requires_scope_and_context_is_not_sticky() -> None:
    value, shapes, starts, locations, weights = inputs()
    with pytest.raises(RuntimeError, match="requires StaticEvaluationCandidate"):
        static_attention_apply(value, shapes, starts, locations, weights, 64)
    fixed = layout()
    with StaticAttentionForwardGuard(fixed):
        cached_shapes, cached_starts = static_attention_layout_tensors(((2, 2), (1, 2)), "cpu")
        assert cached_shapes is fixed.spatial_shapes
        assert cached_starts is fixed.level_start_index
        output = static_attention_apply(value, shapes, starts, locations, weights, 64)
    assert output.shape == (1, 3, 6)
    with pytest.raises(RuntimeError, match="requires StaticEvaluationCandidate"):
        static_attention_apply(value, shapes, starts, locations, weights, 64)


def test_fresh_feature_shape_assertion_rejects_other_geometry() -> None:
    with pytest.raises(ValueError, match="Fresh feature shapes"):
        with StaticAttentionForwardGuard(layout()):
            static_attention_layout_tensors(((2, 2), (1, 3)), "cpu")


def test_evaluation_wrapper_validates_output_before_return() -> None:
    value, shapes, starts, locations, weights = inputs()
    candidate = StaticEvaluationCandidate(layout())

    def producer(*, bad: bool) -> dict[str, Tensor]:
        output = static_attention_apply(value, shapes, starts, locations, weights, 64)
        return {"attention": output, "later": torch.tensor([float("nan") if bad else 1.0])}

    good = candidate.forward(producer, bad=False)
    assert isinstance(good, dict) and torch.isfinite(good["attention"]).all()
    with pytest.raises(ValueError, match="Nonfinite tensor: output.later"):
        candidate.forward(producer, bad=True)
