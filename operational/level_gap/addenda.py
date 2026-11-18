"""Post-protocol addenda (declared as such in REPORT.md): transfer-risk analyses.

A1 pairwise consistency: ratio test/train of the mean disagreement between estimator pairs.
A2 arbiter: does "closer to the event context estimate" track "closer to truth" on folds, and
   how often is the candidate closer to it on Test12.
A3 disagreement-matched additive projection: fold change of |error| per (band, disagreement
   bin) weighted by the Test12 disagreement distribution.
A4 difficulty-matched emulation: fold rows exponentially tilted on the released-recipe error
   so that its band means equal the official Test12 bands; label-free checks of the emulation.
A5 uninformative-move bound: candidate moves treated as noise independent of the error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry.metrics import BANDS

from . import stats as st_

if TYPE_CHECKING:
    from .analyze import Study

EDGES = (0.0, 10.0, 20.0, 40.0, 80.0)


def _phase(ttc: NDArray[Any]) -> NDArray[np.float64]:
    from operational.phase_geometry.metrics import ttc_to_phase

    return ttc_to_phase(np.asarray(ttc, dtype=np.float64))


def consistency(study: Study) -> list[dict[str, Any]]:
    """A1: mean pairwise disagreement (MiD units) on train (raw and reweighted) and test."""
    from .analyze import mid_phase

    d = study.d
    train = {
        "RCTX": d["train_features"][:, 0].astype(np.float64),
        "ECTX": d["train_features"][:, 10].astype(np.float64),
        "PAIR_R": d["train_rgb_experts_oof"][:, 2],
        "PAIR_E": d["train_event_experts_oof"][:, 2],
        "GEO": study.geo_train,
        "REL": _phase(study.rel[7]),
        "CAND": _phase(study.cand[7]),
        "TRUTH": study.phase_true,
    }
    test = {
        "RCTX": d["test_rctx_location"],
        "ECTX": d["test_ectx_location"],
        "PAIR_R": d["test_rgb_experts"][:, 2],
        "PAIR_E": d["test_event_experts"][:, 2],
        "GEO": study.geo_test,
        "REL": _phase(study.test_rel),
        "CAND": _phase(study.test_cand),
    }
    pairs = [
        ("RCTX", "GEO"),
        ("ECTX", "GEO"),
        ("PAIR_R", "GEO"),
        ("PAIR_E", "GEO"),
        ("RCTX", "ECTX"),
        ("PAIR_R", "PAIR_E"),
        ("REL", "RCTX"),
        ("REL", "GEO"),
        ("CAND", "REL"),
        ("TRUTH", "GEO"),
        ("TRUTH", "RCTX"),
        ("TRUTH", "REL"),
    ]
    rows = []
    for a, b in pairs:
        m_train = mid_phase(train[a], train[b])
        row: dict[str, Any] = {
            "pair": f"{a}-{b}",
            "train_mean": float(m_train.mean()),
            "train_median": float(np.median(m_train)),
        }
        if a in test and b in test:
            m_test = mid_phase(test[a], test[b])
            row["test_mean"] = st_.describe(m_test, study.mass)["mean"]
            row["test_median"] = st_.describe(m_test, study.mass)["q50"]
            row["ratio_mean_test_over_train"] = row["test_mean"] / row["train_mean"]
        rows.append(row)
    return rows


def arbiter(study: Study) -> dict[str, Any]:
    """A2: arbiter-implied win rates (candidate closer to an event estimate) by bin."""
    from .analyze import mid_phase

    d = study.d
    out: dict[str, Any] = {}
    sides = {
        "train": (
            _phase(study.cand[7]),
            _phase(study.rel[7]),
            {
                "ECTX": d["train_features"][:, 10].astype(np.float64),
                "PAIR_E": d["train_event_experts_oof"][:, 2],
            },
            np.ones(len(study.truth)),
        ),
        "test": (
            _phase(study.test_cand),
            _phase(study.test_rel),
            {"ECTX": d["test_ectx_location"], "PAIR_E": d["test_event_experts"][:, 2]},
            study.mass,
        ),
    }
    true_win = np.abs(_phase(study.cand[7]) - study.phase_true) < np.abs(
        _phase(study.rel[7]) - study.phase_true
    )
    for side, (cand, rel, arbiters, weights) in sides.items():
        dis = mid_phase(cand, rel)
        bins = np.digitize(dis, EDGES[1:])
        for name, value in arbiters.items():
            closer = np.abs(cand - value) < np.abs(rel - value)
            rows = []
            for i, low in enumerate(EDGES):
                keep = bins == i
                row = {
                    "low": low,
                    "share": float(weights[keep].sum() / weights.sum()),
                    "arbiter_win_rate": float(weights[keep & closer].sum() / weights[keep].sum()),
                }
                if side == "train":
                    row["true_win_rate"] = float(true_win[keep].mean())
                    row["true_win_given_arbiter_win"] = float(true_win[keep & closer].mean())
                    row["true_win_given_arbiter_loss"] = float(true_win[keep & ~closer].mean())
                rows.append(row)
            move = cand - rel
            rng = np.random.default_rng(20261118)
            random_rates = []
            for _ in range(20):
                shuffled = rel + np.abs(move) * rng.choice((-1.0, 1.0), len(move))
                random_rates.append(
                    np.average(np.abs(shuffled - value) < np.abs(rel - value), weights=weights)
                )
            entry: dict[str, Any] = {
                "overall_arbiter_win_rate": float(weights[closer].sum() / weights.sum()),
                "random_sign_arbiter_win_rate": float(np.mean(random_rates)),
                "move_sign_agrees_with_arbiter_minus_REL": float(
                    np.average(np.sign(move) == np.sign(value - rel), weights=weights)
                ),
                "by_bin": rows,
            }
            if side == "train":
                entry["move_sign_agrees_with_truth_minus_REL"] = float(
                    np.mean(np.sign(move) == np.sign(study.phase_true - rel))
                )
            out[f"{side}::{name}"] = entry
    return out


def _cell_change(
    study: Study, new: NDArray[Any], old: NDArray[Any], condition: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    from .analyze import group4, mid_phase

    e_new = np.abs(_phase(new) - study.phase_true) * 1e4
    e_old = np.abs(_phase(old) - study.phase_true) * 1e4
    strata = study.bands if condition == "target" else group4(st_.predicted_band(old))
    cell = strata * len(EDGES) + np.digitize(mid_phase(_phase(new), _phase(old)), EDGES[1:])
    return e_new - e_old, cell


def _projected(
    cell_sum: NDArray[Any], cell_mass: NDArray[Any], p_test: NDArray[Any], official: NDArray[Any]
) -> NDArray[np.float64]:
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.nan_to_num(cell_sum / cell_mass).reshape(
            *cell_sum.shape[:-1], len(BANDS), len(EDGES)
        )
    delta = (means * p_test).sum(-1)
    return st_.overall_of(np.maximum(official + delta, 0.0))


def additive_projection(
    study: Study,
    folds: dict[str, NDArray[Any]],
    tests: dict[str, NDArray[Any]],
    condition: str = "target",
) -> list[dict[str, Any]]:
    """A3: official_b + sum_bin P_test(bin | predicted group b) * fold change(stratum b, bin).

    ``condition`` selects the fold stratum: the target band (``target``) or the released-recipe
    predicted group (``predicted``, the same label-free stratum used on Test12).
    """
    from .analyze import OFFICIAL_REL, group4, mid_phase

    rows = []
    n_cells = len(BANDS) * len(EDGES)
    groups_test = group4(st_.predicted_band(study.test_rel))
    names, inverse = np.unique(study.groups, return_inverse=True)
    rng = np.random.default_rng(20261118)
    draws = rng.integers(0, len(names), (2000, len(names)))
    counts = np.zeros((2000, len(names)))
    np.add.at(counts, (np.repeat(np.arange(2000), len(names)), draws.ravel()), 1.0)
    for name, fold_pred in folds.items():
        if name == "REL":
            continue
        change, cell = _cell_change(study, fold_pred, study.rel[7], condition)
        sums = np.zeros((len(names), n_cells))
        mass = np.zeros((len(names), n_cells))
        np.add.at(sums, (inverse, cell), change)
        np.add.at(mass, (inverse, cell), 1.0)
        dis_test = mid_phase(_phase(tests[name]), _phase(study.test_rel))
        bins_test = np.digitize(dis_test, EDGES[1:])
        p_test = np.zeros((len(BANDS), len(EDGES)))
        for g in range(len(BANDS)):
            keep = groups_test == g
            for i in range(len(EDGES)):
                p_test[g, i] = study.mass[keep & (bins_test == i)].sum() / study.mass[keep].sum()
        official = np.asarray([OFFICIAL_REL[b] for b in BANDS])

        point = float(_projected(sums.sum(0), mass.sum(0), p_test, official))
        reps = _projected(counts @ sums, counts @ mass, p_test, official)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_means = np.nan_to_num(sums.sum(0) / mass.sum(0)).reshape(len(BANDS), len(EDGES))
        rows.append(
            {
                "variant": name,
                "condition": condition,
                "projected_overall": point,
                "ci95": st_.ci(reps),
                "p_worse_than_submitted": float(np.mean(reps > OFFICIAL_REL["overall"])),
                "band_delta": {
                    b: float((cell_means[i] * p_test[i]).sum()) for i, b in enumerate(BANDS)
                },
                "test_bin_share_by_group": {b: p_test[i].tolist() for i, b in enumerate(BANDS)},
            }
        )
    return rows


def tilt_weights(
    error_mid: NDArray[Any], bands: NDArray[Any], targets: dict[str, float]
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Per-band exponential tilt w = exp(-lambda_b |e|) matching each target mean |e|."""
    e = np.abs(np.asarray(error_mid, dtype=np.float64))
    w = np.ones_like(e)
    lambdas: dict[str, float] = {}
    for i, b in enumerate(BANDS):
        keep = (np.asarray(bands) == i) & np.isfinite(e)
        values = e[keep]
        target = targets[b]
        if values.mean() <= target:
            lambdas[b] = 0.0
            continue
        low, high = 0.0, 1.0
        while np.average(values, weights=np.exp(-high * (values - values.min()))) > target:
            high *= 2.0
        for _ in range(100):
            mid = 0.5 * (low + high)
            if np.average(values, weights=np.exp(-mid * (values - values.min()))) > target:
                low = mid
            else:
                high = mid
        lambdas[b] = 0.5 * (low + high)
        w[keep] = np.exp(-lambdas[b] * (values - values.min()))
    for i, _ in enumerate(BANDS):
        keep = np.asarray(bands) == i
        w[keep] = w[keep] / w[keep].mean()
    return w, lambdas


def emulation(study: Study, folds: dict[str, NDArray[Any]]) -> dict[str, Any]:
    """A4: difficulty-matched fold emulation with label-free plausibility checks."""
    from operational.phase_geometry.metrics import row_mid

    from .analyze import OFFICIAL_REL, mid_phase

    rel_mid = row_mid(study.truth, study.rel[7])
    targets = {b: OFFICIAL_REL[b] for b in BANDS}
    weights, lambdas = tilt_weights(rel_mid, study.bands, targets)
    mids = {name: row_mid(study.truth, pred) for name, pred in folds.items()}
    reps = st_.bootstrap_band_means(mids, weights, study.bands, study.groups, replicates=10_000)
    overall_rel = st_.overall_of(reps["REL"])
    variants = {}
    for name, m in mids.items():
        point = st_.weighted_overall(m, study.bands, weights)
        over = st_.overall_of(reps[name])
        variants[name] = {
            **point,
            "ci95": st_.ci(over),
            "delta_vs_REL_ci95": st_.ci(over - overall_rel),
            "p_worse_than_REL": float(np.mean((over - overall_rel)[np.isfinite(over)] > 0)),
            "delta_vs_RGB_GEO_BAND_ci95": st_.ci(over - st_.overall_of(reps["RGB_GEO_BAND"])),
            "p_better_than_RGB_GEO_BAND": float(
                np.mean((over - st_.overall_of(reps["RGB_GEO_BAND"])) < 0)
            ),
        }
    dis = mid_phase(_phase(study.cand[7]), _phase(study.rel[7]))
    checks = {
        "ess": st_.effective_sample_size(weights),
        "lambda_per_band": lambdas,
        "cand_rel_disagreement_mean": {
            "emulated_folds": float(np.average(dis, weights=weights)),
            "raw_folds": float(dis.mean()),
            "test": st_.describe(
                mid_phase(_phase(study.test_cand), _phase(study.test_rel)), study.mass
            )["mean"],
        },
        "rel_geo_disagreement_mean": {
            "emulated_folds": float(
                np.average(mid_phase(_phase(study.rel[7]), study.geo_train), weights=weights)
            ),
            "raw_folds": float(mid_phase(_phase(study.rel[7]), study.geo_train).mean()),
            "test": st_.describe(mid_phase(_phase(study.test_rel), study.geo_test), study.mass)[
                "mean"
            ],
        },
        "rel_pred_band_mix": {
            "emulated_folds": st_.shares(
                st_.predicted_band(study.rel[7]), st_.PRED_BAND_NAMES, weights
            ),
            "test": st_.shares(st_.predicted_band(study.test_rel), st_.PRED_BAND_NAMES, study.mass),
        },
        "geo_error_mean_by_band_emulated": {
            b: float(
                np.average(
                    mid_phase(study.geo_train, study.phase_true)[study.bands == i],
                    weights=weights[study.bands == i],
                )
            )
            for i, b in enumerate(BANDS)
        },
    }
    return {"variants": variants, "checks": checks}


def noise_bound(study: Study, tests: dict[str, NDArray[Any]]) -> list[dict[str, Any]]:
    """A5: E|s_b e + d| with fold errors rescaled to the official bands and test moves d."""
    from operational.phase_geometry.metrics import row_mid

    from .analyze import OFFICIAL_REL, group4

    rng = np.random.default_rng(20261118)
    e = (_phase(study.rel[7]) - study.phase_true) * 1e4
    fold_band = {
        b: float(np.nanmean(row_mid(study.truth, study.rel[7])[study.bands == i]))
        for i, b in enumerate(BANDS)
    }
    groups_test = group4(st_.predicted_band(study.test_rel))
    rows = []
    for name, pred in tests.items():
        if name == "REL":
            continue
        move = (_phase(pred) - _phase(study.test_rel)) * 1e4
        bands = {}
        for i, b in enumerate(BANDS):
            errors = e[study.bands == i] * OFFICIAL_REL[b] / fold_band[b]
            pool = np.repeat(move[groups_test == i], study.mass[groups_test == i].astype(int))
            sample_e = rng.choice(errors, 200_000)
            sample_d = rng.choice(pool, 200_000) * rng.choice((-1.0, 1.0), 200_000)
            bands[b] = float(np.abs(sample_e + sample_d).mean())
        rows.append(
            {
                "variant": name,
                "band_MiD": bands,
                "overall": float(
                    sum(w * bands[b] for b, w in zip(BANDS, (0.5, 0.3, 0.1, 0.1), strict=True))
                ),
                "assumption": "moves independent of the error, random sign",
            }
        )
    return rows


__all__ = [
    "additive_projection",
    "arbiter",
    "consistency",
    "emulation",
    "noise_bound",
    "tilt_weights",
]
