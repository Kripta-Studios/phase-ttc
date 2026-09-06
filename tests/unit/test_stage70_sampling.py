"""Interpolation parity and deterministic CUDA backward before scientific fits."""

import pytest
import torch
from torch.nn import functional

from e_jepa_ttc.models.projective_phase_network import QuadrantAveragePool
from e_jepa_ttc.models.projective_phase_sweep import sample_correlations, sweep_scores, translations


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bilinear_values_and_both_gradients_match_reference(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    generator = torch.Generator(device=device).manual_seed(7)
    corr = torch.randn(3, 1, 7, 9, device=device, dtype=dtype, generator=generator)
    grid = torch.rand(3, 37, 2, 2, device=device, dtype=dtype, generator=generator) * 3 - 1.5
    upstream = torch.randn(3, 1, 37, 2, device=device, dtype=dtype, generator=generator)
    reference_input, reference_grid = corr.clone().requires_grad_(), grid.clone().requires_grad_()
    new_input, new_grid = corr.clone().requires_grad_(), grid.clone().requires_grad_()
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        reference = functional.grid_sample(reference_input, reference_grid, align_corners=False)
        expected = torch.autograd.grad(reference, (reference_input, reference_grid), upstream)
        torch.use_deterministic_algorithms(True)
        actual = sample_correlations(new_input, new_grid)
        gradients = torch.autograd.grad(actual, (new_input, new_grid), upstream)
        tolerance = 2e-5 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
        for value, target in zip(gradients, expected, strict=True):
            torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)
        repeated = []
        for _ in range(3):
            repeat = corr.clone().requires_grad_()
            out = sample_correlations(repeat, grid)
            repeated.append(torch.autograd.grad(out, repeat, upstream)[0])
        assert all(torch.equal(repeated[0], value) for value in repeated[1:])
    finally:
        torch.use_deterministic_algorithms(prior)


def test_boundary_centers_padding_and_outside():
    corr = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    grid = torch.tensor(
        [[[[-0.75, -2 / 3]], [[0.75, 2 / 3]], [[-1.0, -1.0]], [[1.0, 1.0]], [[2.0, 2.0]]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        sample_correlations(corr, grid),
        functional.grid_sample(corr, grid, align_corners=False),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpointed_chunks_preserve_all_values_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    generator = torch.Generator(device=device).manual_seed(7)
    inputs = [torch.randn(2, 3, 8, 8, device=device, generator=generator) for _ in range(2)]
    inputs.append(torch.rand(2, 8, 8, device=device, generator=generator) + 0.1)
    scale = torch.tensor([[0.8, 1.0, 1.2, 1.5]] * 2, device=device)
    results, gradients = [], []
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for checkpointed in (False, True):
            left, right, mass = [value.clone().requires_grad_() for value in inputs]
            scores, overlap = sweep_scores(
                left,
                right,
                scale,
                translations(radius=1, device=device),
                mass,
                chunk=3,
                checkpoint_chunks=checkpointed,
            )
            results.append((scores.detach(), overlap.detach()))
            gradients.append(
                torch.autograd.grad(scores.square().sum() + overlap.sum(), (left, right, mass))
            )
        for left, right in zip(results[0], results[1], strict=True):
            assert torch.equal(left, right)
        for left, right in zip(gradients[0], gradients[1], strict=True):
            assert torch.equal(left, right)
    finally:
        torch.use_deterministic_algorithms(prior)


@pytest.mark.parametrize("shape", [(16, 16), (3, 5), (2, 2)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quadrant_pool_preserves_adaptive_regions_values_and_gradients(shape, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(2, 5, *shape, device=device, generator=generator)
    expected_input, actual_input = x.clone().requires_grad_(), x.clone().requires_grad_()
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        expected = functional.adaptive_avg_pool2d(expected_input, (2, 2))
        expected_gradient = torch.autograd.grad(expected.square().sum(), expected_input)[0]
        torch.use_deterministic_algorithms(True)
        actual = QuadrantAveragePool()(actual_input)
        actual_gradient = torch.autograd.grad(actual.square().sum(), actual_input)[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6)
    finally:
        torch.use_deterministic_algorithms(previous)
