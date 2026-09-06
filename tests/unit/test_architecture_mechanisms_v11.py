"""Counterfactuals preserve input identity and cannot silently enter supervised training."""

import subprocess
import types

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.event_context128 import ArchitectureInput, RateNormalization, four_contexts
from e_jepa_ttc.evaluation.architecture_mechanisms_v11 import mechanism_input, zero_event_contexts
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    rng = torch.get_rng_state()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)
    torch.set_rng_state(rng)


def test_zero_event_intervention_matches_real_count_adapter_exactly():
    support = np.ones((128, 128), bool)
    support[:, :3] = False
    norm = RateNormalization((2.0, 3.0), (4.0, 5.0), "a" * 64, 10)
    actual = four_contexts(
        np.zeros((2, 8, 2, 128, 128), np.uint16),
        np.ones((2, 8)),
        support,
        np.array([-350000, -250000, -150000, -50000], np.int64),
        norm,
    )
    assert torch.equal(zero_event_contexts(actual, norm), actual.contexts)
    assert not torch.equal(actual.contexts, torch.zeros_like(actual.contexts))


def test_fixed_translation_has_no_wrap_or_scale_and_does_not_mutate_source():
    context = torch.zeros((4, 8, 16, 16))
    context[-1, :, 8, 8] = 1
    sample = ArchitectureInput(
        context,
        torch.tensor([-0.175, -0.125, -0.075, -0.025]),
        torch.ones((16, 16), dtype=torch.bool),
    )
    norm = RateNormalization((0.0, 0.0), (1.0, 1.0), "a" * 64, 10)
    rng = torch.get_rng_state()
    translated, times, kwargs = mechanism_input(
        sample, norm, "translation_only", token="fixture", seed=7
    )
    for index, position in enumerate((5, 7, 9, 11)):
        assert translated[index, 0, 8, position] == 1
        assert translated[index, 0].sum() == 1
    assert torch.equal(
        context[-1, 0],
        torch.nn.functional.one_hot(torch.tensor(8 * 16 + 8), 256).reshape(16, 16).float(),
    )
    assert torch.equal(times, sample.times) and not kwargs
    assert torch.equal(rng, torch.get_rng_state())


def test_eval_only_geometry_interventions_preserve_direct_content():
    model = ProjectivePhaseNetwork(checkpoint_sweep=False)
    contexts = torch.randn((1, 4, 8, 16, 16))
    times = torch.tensor([[-0.175, -0.125, -0.075, -0.025]])
    with pytest.raises(ValueError, match="evaluation-only"):
        model(contexts, times, diagnostic_zero_geometry=True)
    model.eval()
    with torch.inference_mode():
        direct = model(contexts, times, arm="DIRECT")
        zero = model(contexts, times, diagnostic_zero_geometry=True)
        shuffled = model(contexts, times, diagnostic_time_permutation=torch.tensor([[2, 3, 0, 1]]))
    assert not zero["evidence"].any() and not zero["observability"].any()
    assert torch.equal(direct["logits"], zero["logits"])
    assert torch.equal(direct["direct_logits"], shuffled["direct_logits"])


def test_default_four_arm_forward_is_exactly_unchanged_from_pre_intervention_code():
    # Pinned local source only; no model fitting or external source download.
    source = subprocess.check_output(
        ["git", "show", "cc25ef6:src/e_jepa_ttc/models/projective_phase_network.py"],
        text=True,
        encoding="utf-8",
    )
    reference = types.ModuleType("e_jepa_ttc.models._pre_intervention_reference")
    exec(compile(source, "pinned_pre_intervention_reference.py", "exec"), reference.__dict__)
    torch.manual_seed(71)
    model = ProjectivePhaseNetwork(checkpoint_sweep=False).eval()
    old = reference.ProjectivePhaseNetwork(checkpoint_sweep=False).eval()
    old.load_state_dict(model.state_dict())
    contexts = torch.randn((1, 4, 8, 16, 16))
    times = torch.tensor([[-0.175, -0.125, -0.075, -0.025]])
    with torch.inference_mode():
        for arm in ("DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"):
            kwargs = {
                "arm": arm,
                "permutation": torch.tensor([[2, 3, 0, 1]]) if arm == "PPS-TIMEPERM" else None,
            }
            actual, expected = model(contexts, times, **kwargs), old(contexts, times, **kwargs)
            assert all(torch.equal(actual[key], expected[key]) for key in expected)
