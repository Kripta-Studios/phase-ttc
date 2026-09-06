import math

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.stage70_provenance import assert_excludes
from e_jepa_ttc.evaluation.stage70_76 import Evidence, decide, missing_support_is_local_block
from e_jepa_ttc.models.phase_distribution_v11 import (
    benchmark_mass,
    crps,
    phase_edges,
    phase_to_finite_ttc,
    quantile,
    ttc_to_phase,
    validate_edges,
)
from e_jepa_ttc.models.projective_geometry_v11 import (
    pixel_centers,
    reanchor,
    relative_seconds,
    scale_between,
)
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork
from e_jepa_ttc.models.projective_phase_sweep import (
    marginalize_translation,
    sweep_scores,
)


@pytest.fixture(autouse=True, scope="module")
def reference_cpu_threads():
    """Scope reference-test execution settings without changing collection-time globals."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("ttc", [-60.0, -10.0, -1.0, -0.101, 0.101, 0.5, 1.0, 10.0, 60.0])
def test_signed_roundtrip(ttc):
    t = torch.tensor([ttc], dtype=torch.float64)
    torch.testing.assert_close(phase_to_finite_ttc(ttc_to_phase(t)), t, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, 0.05, 0.1, float("nan"), float("inf")])
def test_phase_rejects_invalid(bad):
    with pytest.raises(ValueError):
        ttc_to_phase(torch.tensor([bad]))


def test_zero_phase_has_documented_finite_limit():
    assert phase_to_finite_ttc(torch.tensor([0.0])).item() == 60


def test_clock_integer_subtraction():
    a = torch.tensor([19_317_100_000], dtype=torch.int64)
    t = a + torch.tensor([3])
    torch.testing.assert_close(relative_seconds(t, a), torch.tensor([3e-6], dtype=torch.float64))
    with pytest.raises(TypeError):
        relative_seconds(t.float(), a)


@pytest.mark.parametrize("q", [-2.0, -0.5, 0.0, 0.3, 1.8])
def test_projective_composition(q):
    q = torch.tensor(q, dtype=torch.float64)
    a, b, c = [torch.tensor(t, dtype=torch.float64) for t in [-0.04, 0.02, 0.12]]
    ab, v1 = scale_between(q, a, b)
    bc, v2 = scale_between(q, b, c)
    ac, v3 = scale_between(q, a, c)
    assert v1 and v2 and v3
    torch.testing.assert_close(ab * bc, ac, rtol=1e-13, atol=1e-13)
    ba, _ = scale_between(q, b, a)
    torch.testing.assert_close(ba * ab, torch.ones_like(ab))


def test_physical_forward_is_not_constant_divergence():
    q = torch.tensor([1.0], dtype=torch.float64)
    s, v = scale_between(q, torch.tensor([0.0]), torch.tensor([0.2], dtype=torch.float64))
    assert v.all()
    torch.testing.assert_close(s, torch.tensor([1.25], dtype=torch.float64))
    assert abs(s.item() - math.exp(0.2)) > 0.02


def test_reanchor_matches_geometry():
    q = torch.tensor([0.8], dtype=torch.float64)
    h = torch.tensor([0.07], dtype=torch.float64)
    q2, v = reanchor(q, h)
    a = torch.tensor([0.02], dtype=torch.float64)
    b = torch.tensor([0.11], dtype=torch.float64)
    before, _ = scale_between(q, a, b)
    after, _ = scale_between(q2, a - h, b - h)
    assert v.all()
    torch.testing.assert_close(before, after)


def test_singularity_is_masked():
    s, v = scale_between(torch.tensor([10.0]), torch.tensor([0.0]), torch.tensor([0.2]))
    assert not v.any() and torch.isfinite(s).all()


@pytest.mark.parametrize("level", [0.1, 0.5, 0.9])
def test_uniform_quantile(level):
    e = torch.tensor([-2.0, 1.0, 4.0], dtype=torch.float64)
    logits = torch.zeros((2, 2), dtype=torch.float64)
    torch.testing.assert_close(
        quantile(logits, e, level), torch.full((2,), -2 + 6 * level, dtype=torch.float64)
    )


@pytest.mark.parametrize("y", [-2.0, -1.0, -0.31, 0.0, 0.42, 1.0, 2.0])
def test_crps_against_numerical_integral(y):
    e = torch.tensor([-1.0, -0.4, 0.1, 1.0], dtype=torch.float64)
    logits = torch.tensor([[0.2, 1.0, -0.2]], dtype=torch.float64, requires_grad=True)
    exact = crps(logits, torch.tensor([y], dtype=torch.float64), e)
    x = np.linspace(min(-1.1, y - 0.1), max(1.1, y + 0.1), 100001)
    p = logits.detach().softmax(-1).numpy()[0]
    cdf = np.zeros_like(x)
    for j in range(3):
        cdf += p[j] * np.clip((x - e[j].item()) / (e[j + 1] - e[j]).item(), 0, 1)
    approximate = np.trapezoid((cdf - (x >= y)) ** 2, x)
    assert abs(exact.item() - approximate) < 3e-5
    exact.backward()
    assert torch.isfinite(logits.grad).all()


def test_crps_direction_and_nonnegative():
    e = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    for y in [-3.0, -0.5, 0.0, 0.5, 3.0]:
        logits = torch.randn(20, 2, dtype=torch.float64)
        assert bool((crps(logits, torch.full((20,), y, dtype=torch.float64), e) >= -1e-12).all())
    y = torch.tensor([0.5], dtype=torch.float64)
    good = crps(torch.tensor([[-10.0, 10.0]], dtype=torch.float64), y, e)
    bad = crps(torch.tensor([[10.0, -10.0]], dtype=torch.float64), y, e)
    assert good < bad


@pytest.mark.parametrize("edges", [[0.0, 0.0, 1.0], [1.0, 0.0], [0.0, float("nan")]])
def test_bad_edges(edges):
    with pytest.raises(ValueError):
        validate_edges(torch.tensor(edges))


def test_fixed_edges_cover_signed_domain():
    e = phase_edges(dtype=torch.float64)
    validate_edges(e)
    assert len(e) == 109
    assert e[0] > -math.log(2) and e[-1] == 8
    assert torch.isfinite(phase_to_finite_ttc((e[:-1] + e[1:]) / 2)).all()


def test_metric_mass():
    s = torch.tensor([0] * 8 + [1] * 8)
    b = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 2)
    w = benchmark_mass(s, b)
    assert abs(w.sum().item() - 1) < 1e-12
    for k, v in enumerate([0.5, 0.3, 0.1, 0.1]):
        assert abs(w[b == k].sum().item() - v) < 1e-12
    with pytest.raises(ValueError):
        benchmark_mass(torch.tensor([0]), torch.tensor([0]))


def test_grid_centers():
    p = pixel_centers(4, 4, dtype=torch.float64)
    torch.testing.assert_close(p[0, 0], torch.tensor([-0.75, -0.75], dtype=torch.float64))
    torch.testing.assert_close(p[-1, -1], torch.tensor([0.75, 0.75], dtype=torch.float64))


@pytest.mark.parametrize("chunk", [1, 2, 8])
def test_sweep_identity_peak_and_chunk_equivalence(chunk):
    torch.manual_seed(12)
    f = torch.randn(1, 8, 6, 6, dtype=torch.float64)
    scales = torch.tensor([[0.8, 1.0, 1.2]], dtype=torch.float64)
    d = torch.zeros(1, 2, dtype=torch.float64)
    s, o = sweep_scores(f, f, scales, d, chunk=chunk)
    assert s[0, :, 0].argmax().item() == 1
    torch.testing.assert_close(
        s[0, 1, 0], torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    assert o[0, 1, 0] == 1
    ref, ro = sweep_scores(f, f, scales, d, chunk=3)
    torch.testing.assert_close(s, ref)
    torch.testing.assert_close(o, ro)


def test_sweep_gradients():
    torch.manual_seed(1)
    f = torch.randn(1, 4, 5, 5, dtype=torch.float64, requires_grad=True)
    g = torch.randn_like(f, requires_grad=True)
    scales = torch.tensor([[0.94, 1.05]], dtype=torch.float64, requires_grad=True)
    s, o = sweep_scores(f, g, scales, torch.zeros(1, 2, dtype=torch.float64), chunk=1)
    s.sum().backward()
    for grad in (f.grad, g.grad, scales.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_invalid_translation_has_no_evidence():
    f = torch.randn(1, 4, 4, 4)
    s, o = sweep_scores(f, f, torch.ones(1, 2), torch.tensor([[4.0, 4.0]]))
    ev, valid = marginalize_translation(s, o)
    assert not valid.any() and torch.equal(ev, torch.zeros_like(ev))


def test_valid_count_marginalization():
    s = torch.full((1, 2, 3), 0.5)
    o = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]])
    e, v = marginalize_translation(s, o)
    torch.testing.assert_close(e, torch.tensor([[0.5, 0.5]]))


def test_weights_reject_zero():
    f = torch.randn(1, 3, 4, 4)
    with pytest.raises(ValueError):
        sweep_scores(f, f, torch.ones(1, 1), torch.zeros(1, 2), torch.zeros(1, 4, 4))


def test_model_initial_equivalence_and_backward():
    torch.manual_seed(7)
    m = ProjectivePhaseNetwork(radius=0, edges=torch.tensor([-0.1, -0.02, 0.0, 0.03, 0.1]))
    x = torch.randn(1, 4, 8, 16, 16)
    t = torch.tensor([[-0.15, -0.1, -0.05, 0.0]])
    a = m(x, t, arm="DIRECT")
    b = m(x, t, arm="PPS-TRAJ")
    torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
    loss = crps(b["logits"], torch.tensor([0.04]), m.edges).sum()
    loss.backward()
    assert m.encoder.net[0].weight.grad.abs().sum() > 0
    assert m.score_refiner[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(b["ttc"]).all()


@pytest.mark.parametrize("arm", ["DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"])
def test_model_arm_smoke(arm):
    m = ProjectivePhaseNetwork(radius=0, edges=torch.tensor([-0.1, 0.0, 0.1]))
    x = torch.randn(1, 4, 8, 16, 16)
    t = torch.tensor([[-0.15, -0.1, -0.05, 0.0]])
    result = m(x, t, arm=arm, permutation=torch.tensor([[1, 0, 3, 2]]))
    assert result["logits"].shape == (1, 2)
    assert torch.isfinite(result["phase"]).all()


def test_bad_temporal_order():
    m = ProjectivePhaseNetwork(radius=0, edges=torch.tensor([-0.1, 0.0, 0.1]))
    with pytest.raises(ValueError):
        m(torch.randn(1, 4, 8, 16, 16), torch.zeros(1, 4))


def test_exploration_is_not_significance_gate():
    e = Evidence(-7.0, 1.0, 0.5, 7, 9, 0.0, -3.0, 1.0)
    d = decide(e)
    assert d["replicate_new_architecture"] and not d["development_statistical_gate"]


def test_complementarity_can_authorize_router_without_standalone_win():
    e = Evidence(10.0, 15.0, 14.0, 3, 9, 0.0, 8.0, 1.0, 9.0)
    assert decide(e)["authorize_new_expert_routing"]


def test_classical_flow_not_required_for_architecture():
    assert missing_support_is_local_block(representation_ready=True, classical_flow_ready=False)[
        "train_standalone_architecture"
    ]


def test_bad_evidence_cannot_become_negative():
    with pytest.raises(ValueError):
        decide(Evidence(float("nan"), 0.0, 0.0, 5, 9, 0.0, 0.0, 1.0))


def test_ancestry_checks_transitive_selection_leak():
    nodes = {
        "new": dict(
            fit_sequences=["a"],
            selection_sequences=[],
            normalization_sequences=["a"],
            parents=["teacher"],
        ),
        "teacher": dict(
            fit_sequences=["a"],
            selection_sequences=["held"],
            normalization_sequences=[],
            parents=[],
        ),
    }
    with pytest.raises(ValueError):
        assert_excludes(nodes, "new", {"held"})
    nodes["teacher"]["selection_sequences"] = []
    assert assert_excludes(nodes, "new", {"held"}) == {"new", "teacher"}


def test_ancestry_rejects_unrecorded_field():
    with pytest.raises(ValueError):
        assert_excludes({"a": {}}, "a", {"held"})


def test_weighted_batch_expectation_not_batch_renormalized():
    from e_jepa_ttc.models.phase_distribution_v11 import weighted_training_loss

    e = torch.tensor([-0.12, 0.0, 0.12], dtype=torch.float64)
    logits = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -2.0]], dtype=torch.float64, requires_grad=True
    )
    y = torch.tensor([-0.03, 0.05, -0.02], dtype=torch.float64)
    w = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    all_loss = weighted_training_loss(logits, y, e, w, 3)
    one = torch.stack(
        [
            weighted_training_loss(logits[i : i + 1], y[i : i + 1], e, w[i : i + 1], 3)
            for i in range(3)
        ]
    ).mean()
    torch.testing.assert_close(all_loss, one)
    all_loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_autocast_boundary_keeps_physics_and_distribution_float32():
    model = ProjectivePhaseNetwork(radius=0, edges=torch.tensor([-0.1, 0.0, 0.1]))
    x = torch.randn(1, 4, 8, 16, 16)
    times = torch.tensor([[-0.15, -0.1, -0.05, 0.0]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(x, times, arm="PPS-TRAJ")
    assert output["logits"].dtype == torch.float32
    assert output["evidence"].dtype == torch.float32
    assert output["phase"].dtype == torch.float32
    assert torch.isfinite(output["ttc"]).all()
