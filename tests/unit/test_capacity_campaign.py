"""Zero-optimizer CPU checks for capacity, cohort isolation and combined budgets."""

import hashlib

import pytest
import torch

from e_jepa_ttc.data.capacity_cohort import CohortRecipe, build_cohort
from e_jepa_ttc.data.stage70_roles import SamplingRow
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork
from e_jepa_ttc.training.capacity_campaign import admit_capacity_budget, capacity_graph
from e_jepa_ttc.training.stage73_factorial import Availability


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def availability(diversity=True):
    return Availability(True, diversity, "a" * 64, "b" * 64, ("2M", "4M", "8M"), "4M")


@pytest.mark.parametrize("size,count", [("2M", 2037822), ("4M", 4059286), ("8M", 8043646)])
def test_exact_parameter_counts_and_cpu_direct_gradient(size, count):
    model = ProjectivePhaseNetwork(capacity=size, radius=0)
    assert sum(p.numel() for p in model.parameters()) == count
    output = model(
        torch.randn(1, 4, 8, 16, 16), torch.tensor([[-0.175, -0.125, -0.075, -0.025]]), arm="DIRECT"
    )
    logits = output["logits"]
    assert logits.shape == (1, 108) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert model.encoder.net[0].weight.grad is not None
    assert torch.isfinite(model.encoder.net[0].weight.grad).all()


def test_default_2m_preserves_original_seed7_parameter_and_buffer_bytes():
    torch.manual_seed(7)
    model = ProjectivePhaseNetwork()
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    assert digest.hexdigest() == "74893997a22afc502d892b48851413980320b17d0add088eecd8b73c03cb4cb7"


@pytest.mark.parametrize("size", ["2M", "4M", "8M"])
@pytest.mark.parametrize("arm", ["PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"])
def test_geometry_uses_scaled_encoder_and_retains_108_bin_float32_contract(size, arm):
    model = ProjectivePhaseNetwork(capacity=size, radius=0)
    output = model(
        torch.randn(1, 4, 8, 16, 16),
        torch.tensor([[-0.175, -0.125, -0.075, -0.025]]),
        arm=arm,
        permutation=torch.tensor([[1, 0, 3, 2]]) if arm == "PPS-TIMEPERM" else None,
    )
    assert output["evidence"].shape == (1, 4, 108)
    assert output["evidence"].dtype == torch.float32
    assert output["observability"].dtype == torch.bool
    assert torch.isfinite(output["ttc"]).all()
    (output["logits"].square().mean() + output["evidence"].square().mean()).backward()
    assert model.encoder.net[0].weight.grad is not None
    assert torch.isfinite(model.encoder.net[0].weight.grad).all()


@pytest.mark.parametrize("diversity,total,s73", [(True, 720000, 147000), (False, 639000, 66000)])
def test_matrix_and_conditional_budgets_are_prospective(diversity, total, s73):
    graph = capacity_graph(availability(diversity))
    assert len(graph["jobs"]["S72"]) == 36
    assert graph["optimizer_updates"]["S72"] == 288000
    assert graph["optimizer_updates"]["S73"] == s73
    assert graph["architecture_maximum_updates"] == total
    for stage in ("S73", "S74", "S75"):
        assert {job["capacity"] for job in graph["jobs"][stage]} == {"4M"}
    assert {job["seed"] for job in graph["jobs"]["S74"]} == {13, 23}
    assert not graph["S73_requires_positive_S72"]


def test_secondary_size_cannot_replace_registered_primary():
    with pytest.raises(ValueError, match="primary"):
        Availability(True, False, "a" * 64, "b" * 64, ("2M", "4M", "8M"), "8M")


def test_original_work_and_unknown_comparator_cost_block_admission():
    result = admit_capacity_budget(
        capacity_graph(availability(False)),
        predecessor_terminal=False,
        charged_updates=4257,
        outstanding_reserved_updates=6000,
        comparator_updates=None,
        confirmation_updates=0,
        technical_updates_reserved=400,
    )
    assert result["status"] == "PREREQUISITE_BLOCKED"
    assert "ORIGINAL_CAMPAIGN_NOT_TERMINAL" in result["reasons"]
    assert "COMPARATOR_BUDGET_UNRESOLVED" in result["reasons"]
    assert result["required_total_updates"] is None


@pytest.mark.parametrize(
    "consumed,expected", [(245257, "ADMITTED"), (530257, "PREREQUISITE_BLOCKED")]
)
def test_whole_graph_must_fit_shared_budget(consumed, expected):
    result = admit_capacity_budget(
        capacity_graph(availability(False)),
        predecessor_terminal=True,
        charged_updates=consumed,
        outstanding_reserved_updates=0,
        comparator_updates=10000,
        confirmation_updates=0,
        technical_updates_reserved=400,
    )
    assert result["status"] == expected


def cohort_input(count):
    mapping = {f"s{i:02}": f"g{i:02}" for i in range(count)}
    mapping.update({"confirmation": "held", "protected": "forbidden"})
    rows = [
        SamplingRow(f"{seq}_{i}", seq, "track", i * 20000) for seq in mapping for i in range(20)
    ]
    return mapping, rows


@pytest.mark.parametrize("count,expected", [(22, [7, 7, 8]), (27, [9, 9, 9])])
def test_available_sequence_count_and_shared_folds(count, expected):
    mapping, rows = cohort_input(count)
    result = build_cohort(
        rows,
        sequence_to_group=mapping,
        confirmation_sequences=["confirmation"],
        protected_sequences=["protected"],
        recipe=CohortRecipe(sample_limit=256),
    )
    assert result["actual_sequence_count"] == count
    assert sorted(len(fold["held_sequences"]) for fold in result["folds"]) == expected
    assert len(result["ordered_tokens"]) == 256
    for fold in result["folds"]:
        assert not set(fold["train_sequences"]) & set(fold["held_sequences"])
        assert not set(fold["train_tokens"]) & set(fold["held_tokens"])
        assert set(fold["inner_sequence_to_fold"]) == set(fold["train_sequences"])
    repeated = build_cohort(
        list(reversed(rows)),
        sequence_to_group=mapping,
        confirmation_sequences=["confirmation"],
        protected_sequences=["protected"],
        recipe=CohortRecipe(sample_limit=256),
    )
    assert repeated == result


def test_related_confirmation_alias_is_excluded_before_selection():
    mapping, rows = cohort_input(22)
    mapping["s00"] = "held"
    mapping["s01"] = mapping["s02"]
    result = build_cohort(
        rows,
        sequence_to_group=mapping,
        confirmation_sequences=["confirmation"],
        protected_sequences=["protected"],
        recipe=CohortRecipe(sample_limit=256),
    )
    assert "s00" not in result["selected_sequences"]
    assert result["sequence_to_outer_fold"]["s01"] == result["sequence_to_outer_fold"]["s02"]
