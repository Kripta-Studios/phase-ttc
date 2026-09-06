"""Utility versus every included standalone is separate from the permutation mechanism."""

from copy import deepcopy

import pytest

from e_jepa_ttc.evaluation.router_evidence_v11 import routing_gate


def contrasts():
    return {
        f"S75-FULLCOST_minus_{reference}": {
            "status": "COMPLETE",
            "delta": -3.0,
            "hierarchical_ci95": [-5.0, -0.1],
            "exact_sequence": {"ci95": [-5.0, -0.1]},
            "finite_fraction": 1.0,
            "sign_error_increase": 0.005,
            "crucial_delta": 3.0,
        }
        for reference in ("RISK17", "DIRECT", "PPS", "S75-NEWPERM")
    }


def test_permutation_failure_does_not_rewrite_practical_utility():
    evidence = contrasts()
    evidence["S75-FULLCOST_minus_S75-NEWPERM"]["delta"] = 1.0
    result = routing_gate(evidence, ("DIRECT", "PPS"))
    assert result["development_utility_pass"]
    assert not result["association_point_improvement"]


@pytest.mark.parametrize("reference", ["RISK17", "DIRECT", "PPS"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("delta", -2.99),
        ("finite_fraction", 0.99),
        ("sign_error_increase", 0.0051),
        ("crucial_delta", 3.01),
        ("hierarchical_ci95", [-5.0, 0.0]),
        ("exact_sequence", {"ci95": [-5.0, 0.0]}),
    ],
)
def test_every_reference_must_pass_all_fixed_guards(reference, field, value):
    evidence = deepcopy(contrasts())
    evidence[f"S75-FULLCOST_minus_{reference}"][field] = value
    assert not routing_gate(evidence, ("DIRECT", "PPS"))["development_utility_pass"]


def test_failed_endpoint_is_not_a_scientific_negative_or_a_substituted_control():
    evidence = contrasts()
    evidence["S75-FULLCOST_minus_DIRECT"] = {"status": "UNAVAILABLE_OR_INVALID_ENDPOINT"}
    result = routing_gate(evidence, ("DIRECT", "PPS"))
    assert not result["development_utility_pass"]
    assert result["evidence_status"] == "INCOMPLETE_OR_INVALID_ENDPOINTS"
