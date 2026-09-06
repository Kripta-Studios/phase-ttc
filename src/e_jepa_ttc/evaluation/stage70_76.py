"""Practical exploration and acceptance are deliberately separate decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    delta_vs_risk17: float
    hierarchical_high: float
    sequence_high: float
    sequence_wins: int
    sequence_count: int
    sign_error_increase: float
    crucial_delta: float
    finite_fraction: float
    new_oracle_gain: float = 0.0

    def validate(self) -> None:
        values = (
            self.delta_vs_risk17,
            self.hierarchical_high,
            self.sequence_high,
            self.sign_error_increase,
            self.crucial_delta,
            self.finite_fraction,
            self.new_oracle_gain,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("incomplete/nonfinite evidence is not a scientific negative")
        if self.sequence_count != 9 or not 0 <= self.sequence_wins <= 9:
            raise ValueError("this historical development gate requires exactly nine sequences")


def decide(e: Evidence) -> dict[str, bool | str]:
    e.validate()
    valid = e.finite_fraction == 1.0
    guarded = valid and e.sign_error_increase <= 0.005 and e.crucial_delta <= 3.0
    practical = guarded and e.delta_vs_risk17 <= -5.0 and e.sequence_wins >= 6
    complementary = valid and e.new_oracle_gain >= 8.0
    accepted_development = practical and e.hierarchical_high < 0 and e.sequence_high < 0
    return {
        "replicate_new_architecture": practical or complementary,
        "authorize_new_expert_routing": complementary,
        "development_statistical_gate": accepted_development,
        "independent_confirmation_needed": True,
        "claim_ceiling": "adaptive_development_not_sota",
        "historical_stage67_gate_changed": False,
    }


def missing_support_is_local_block(
    *, representation_ready: bool, classical_flow_ready: bool
) -> dict[str, bool]:
    return {
        "train_standalone_architecture": representation_ready,
        "reuse_classical_flow_features": representation_ready and classical_flow_ready,
    }
