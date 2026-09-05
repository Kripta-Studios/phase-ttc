"""Post-hoc, nonselectable diagnostics on recovered historical tables only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from recover_stage66_tables import write_json
from run_scientific_recovery_v9_stage65 import _load_fit, strict_macro_mass

from e_jepa_ttc.training.stage62_local_field import cross_track_derangement_indices


def main() -> None:
    """Materialize historical cost, routing, sign and frozen usage diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    out = root / "historical_posthoc"
    out.mkdir(exist_ok=False)
    collected: dict[str, list[pd.DataFrame]] = {}
    violations = []
    for fold in range(3):
        with np.load(root / f"tables/outer{fold}_outer_dev.npz", allow_pickle=False) as z:
            phase = z["expert_phase"]
            features = z["features17"]
            truth = z["true_costs"]
            expert = z["expert_ttc"]
        metadata = pd.read_csv(root / f"tables/outer{fold}_outer_dev.csv")
        donor = cross_track_derangement_indices(
            sequence_ids=metadata.sequence_id.astype(str).tolist(),
            track_ids=metadata.track_id.astype(str).tolist(),
            seed=70 + fold,
        )
        pd.DataFrame(
            {
                "sample_token": metadata.sample_token,
                "donor_token": metadata.sample_token.to_numpy()[donor],
                "donor_index": donor,
            }
        ).to_csv(out / f"outer{fold}_usage_donors.csv", index=False)
        for arm in ("S65-RISK8", "S65-RISK17", "S65-CE17-REPLAY", "RouterR"):
            frame = pd.read_csv(root / f"tables/outer{fold}_{arm}_replay.csv")
            selected = frame.selected_expert.to_numpy(int)
            frame["outer_fold"] = fold
            frame["oracle_index"] = truth.argmin(1)
            frame["oracle_loss"] = truth.min(1)
            frame["regret"] = frame.loss - frame.oracle_loss
            frame["wrong_sign"] = np.sign(frame.target_ttc_s) != np.sign(frame.prediction_ttc_s)
            frame["target_positive"] = frame.target_ttc_s > 0
            frame["prediction_positive"] = frame.prediction_ttc_s > 0
            frame["bucket"] = np.select(
                [frame.target_ttc_s <= 0, frame.target_ttc_s <= 3, frame.target_ttc_s <= 6],
                ["negative", "crucial", "small"],
                default="large",
            )
            if arm.startswith("S65-RISK"):
                fit = _load_fit(
                    root / f"frozen_audit/extracted_input/run/stage65/outer{fold}/{arm}.npz"
                )
                costs = fit.predict_regret(features[:, :8] if arm.endswith("8") else features)
                order = np.argsort(phase, axis=1, kind="stable")
                sorted_phase = np.take_along_axis(phase, order, axis=1)
                sorted_cost = np.take_along_axis(costs, order, axis=1)
                gaps = np.diff(sorted_phase, axis=1)
                slopes = np.divide(
                    np.diff(sorted_cost, axis=1),
                    100 * gaps,
                    out=np.zeros_like(gaps),
                    where=gaps > 0,
                )
                pair_excess = np.stack(
                    [
                        np.maximum(
                            np.abs(costs[:, a] - costs[:, b])
                            - 100 * np.abs(phase[:, a] - phase[:, b]),
                            0,
                        )
                        for a, b in ((0, 1), (0, 2), (1, 2))
                    ],
                    axis=1,
                )
                convex_excess = np.maximum(slopes[:, 0] - slopes[:, 1], 0)
                convex_valid = np.all(gaps > 0, axis=1)
                frame["lipschitz_excess_relative_cost"] = pair_excess.max(1)
                frame["convexity_excess_slope"] = np.where(convex_valid, convex_excess, 0)
                frame["duplicate_phase"] = ~convex_valid
                for label, values in (
                    ("lipschitz", pair_excess.max(1)),
                    ("convexity", np.where(convex_valid, convex_excess, 0)),
                ):
                    mask = values > 1e-10
                    violations.append(
                        dict(
                            arm=arm,
                            outer_fold=fold,
                            bound=label,
                            fraction=float(mask.mean()),
                            maximum=float(values.max()),
                            mean_excess=float(values.mean()),
                            wrong_choice_fraction_among_violations=float(
                                np.mean(selected[mask] != truth[mask].argmin(1))
                            )
                            if mask.any()
                            else None,
                        )
                    )
                changed = features.copy()
                changed[:, 10] = phase[donor, 2]
                changed[:, 11] = changed[:, 10] - changed[:, 8]
                changed[:, 12] = changed[:, 10] - changed[:, 9]
                changed[:, 14] = np.abs(changed[:, 11])
                changed[:, 15] = np.abs(changed[:, 12])
                changed_expert = expert.copy()
                changed_expert[:, 2] = expert[donor, 2]
                changed_cost = fit.predict_regret(changed[:, :8] if arm.endswith("8") else changed)
                changed_selection = changed_cost.argmin(1)
                changed_prediction = changed_expert[np.arange(len(expert)), changed_selection]
                usage = frame[["sample_token", "sequence_id", "track_id"]].copy()
                usage["donor_token"] = metadata.sample_token.to_numpy()[donor]
                usage["selected_expert"] = changed_selection
                usage["prediction_ttc_s"] = changed_prediction
                usage["selection_changed"] = selected != changed_selection
                usage.to_csv(out / f"outer{fold}_{arm}_frozen_pair_usage.csv", index=False)
            collected.setdefault(arm, []).append(frame)
    pd.DataFrame(violations).to_csv(out / "COST_BOUND_VIOLATIONS.csv", index=False)
    aggregate = {}
    for arm, parts in collected.items():
        frame = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        mass = strict_macro_mass(frame.target_ttc_s.to_numpy(), frame.sequence_id.to_numpy())
        frame["global_mass"] = mass
        frame["weighted_loss"] = mass * frame.loss
        frame["weighted_regret"] = mass * frame.regret
        frame.to_csv(out / f"{arm}_rows.csv", index=False)
        for keys, name in (
            (["sequence_id"], "sequence"),
            (["outer_fold"], "fold"),
            (["bucket"], "bucket"),
            (["sequence_id", "track_id"], "track"),
            (["sequence_id", "bucket", "selected_expert"], "selection"),
            (["target_positive", "prediction_positive"], "sign_confusion"),
            (["sequence_id", "oracle_index"], "expert_wins"),
        ):
            stats = frame.groupby(keys).agg(
                rows=("loss", "size"),
                mass=("global_mass", "sum"),
                weighted_loss=("weighted_loss", "sum"),
                weighted_regret=("weighted_regret", "sum"),
                mean_loss=("loss", "mean"),
                wrong_sign=("wrong_sign", "sum"),
            )
            stats = cast(pd.DataFrame, stats)
            stats["conditional_weighted_loss"] = stats.weighted_loss / stats.mass
            stats.to_csv(out / f"{arm}_{name}.csv")
        aggregate[arm] = dict(
            score=float(mass @ frame.loss),
            oracle_score=float(mass @ frame.oracle_loss),
            regret=float(mass @ frame.regret),
            weighted_sign_error=float(mass @ frame.wrong_sign),
            evidence_class="POSTHOC_NONSELECTABLE",
            oracle_deployable=False,
        )
        for reference in ("S65-CE17-REPLAY", "RouterR"):
            other = (
                pd.concat(collected[reference]).sort_values("sample_token").reset_index(drop=True)
            )
            comparison = frame[
                ["sample_token", "sequence_id", "track_id", "selected_expert"]
            ].copy()
            comparison = cast(pd.DataFrame, comparison)
            comparison["reference_selected_expert"] = other.selected_expert
            comparison["delta_loss"] = frame.loss - other.loss
            comparison["weighted_contribution"] = mass * comparison.delta_loss
            comparison.to_csv(out / f"{arm}_vs_{reference}_changes.csv", index=False)
            comparison.sort_values("weighted_contribution").head(50).to_csv(
                out / f"{arm}_vs_{reference}_largest_help.csv", index=False
            )
            comparison.sort_values("weighted_contribution").tail(50).to_csv(
                out / f"{arm}_vs_{reference}_largest_harm.csv", index=False
            )
    write_json(out / "SUMMARY.json", aggregate)
    write_json(
        out / "USAGE_CONTRACT.json",
        dict(
            evidence_class="POSTHOC_NONSELECTABLE",
            seed_rule="70 + outer_fold",
            maps_frozen_before_usage_inference=True,
            consistent_pair_action_phase_and_derived_features=True,
            fitted=False,
            replaces_retrained_control=False,
            deployable=False,
            caution=(
                "CE17/RouterR exported values are negative logits, not relative loss predictions."
            ),
        ),
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
