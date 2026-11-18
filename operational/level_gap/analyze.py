"""Compute every pre-registered level-gap measurement and write RESULTS.json plus CSV tables.

``python -m operational.level_gap.analyze --work <artifacts dir> --docs <docs dir>``

Inputs: ``level_gap_inputs.npz`` (``operational.level_gap.extract``), the pinned official scorer
and the public 64-query eAP comparison table.  Test12 is used only through label-free inputs,
predictions and the official aggregate scores of the single submission.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.phase_geometry.geometry import RAW_FIELDS
from operational.phase_geometry.metrics import (
    BAND_WEIGHTS,
    BANDS,
    band_index,
    json_safe,
    phase_to_ttc,
    row_mid,
    summarize,
    ttc_to_phase,
)

from . import stats as st_

OFFICIAL_REL = {  # paper/from_motion_to_time_20261102/data/official_scoring_scores.json
    "c": 43.5189935425749,
    "s": 28.25675232393477,
    "l": 25.71789700925724,
    "n": 29.931394114302854,
    "overall": 35.80145158082389,
    "mean_MiD": 29.298890906303914,
    "num_samples": 6762,
}
GARL_TEST = {"c": 53.13, "s": 37.58, "l": 40.61, "n": 31.29}  # Garl-TTC README / paper


def _required_dir(variable: str, purpose: str) -> Path:
    """Return a directory named by an environment variable, with an actionable error."""
    value = os.environ.get(variable)
    if not value or not Path(value).is_dir():
        raise RuntimeError(f"Set {variable} to {purpose}; it is unset or not a directory.")
    return Path(value)


def scorer_path() -> Path:
    """Locate the official benchmark scorer inside the Garl-TTC checkout."""
    root = _required_dir("PHASE_TTC_GARLTTC_BENCHMARK", "the Garl-TTC checkout holding score.py")
    return root / "garl_ttc_benchmark" / "codabench" / "scoring_program" / "score.py"


GARL_64 = Path("docs/post_v18_transfer_20261017/EAP_64_QUERIES.csv")
SEEDS = (7, 13, 21)
CLIP = 0.1
CLIP_SENSITIVITY = 0.03
DIS_EDGES = (0.0, 10.0, 20.0, 40.0, 80.0)
BLENDS = (0.25, 0.5, 0.75)
GROUPS4 = ("c", "s", "l", "n")  # predicted-band groups: pc | ps | pl+pfar | pn+pfarneg

Arrays = dict[str, NDArray[Any]]


def raw(geometry: NDArray[Any], name: str) -> NDArray[np.float64]:
    """One raw geometry column."""
    return np.asarray(geometry, dtype=np.float64)[:, RAW_FIELDS.index(name)]


def phase(ttc: NDArray[Any]) -> NDArray[np.float64]:
    """Benchmark phase of TTC predictions."""
    return ttc_to_phase(np.asarray(ttc, dtype=np.float64))


def group4(pred_band: NDArray[Any]) -> NDArray[np.int64]:
    """Map six predicted strata to four groups aligned with the target bands."""
    return np.asarray([0, 1, 2, 2, 3, 3], dtype=np.int64)[np.asarray(pred_band)]


def mid_phase(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
    """Absolute phase difference in MiD units."""
    return 1e4 * np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))


def load(work: Path) -> Arrays:
    with np.load(work / "level_gap_inputs.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def track_lengths(sequences: NDArray[Any], tracks: NDArray[Any]) -> NDArray[np.int64]:
    """Number of queries of each row's (sequence, track)."""
    keys = np.char.add(np.char.add(sequences.astype(str), "|"), tracks.astype(str))
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse]


class Study:
    """All arrays and derived quantities shared by the measurements."""

    def __init__(self, data: Arrays) -> None:
        self.d = data
        self.truth = data["train_target_ttc"]
        self.bands = band_index(self.truth)
        self.phase_true = phase(self.truth)
        self.groups = data["train_group_id"].astype(str)
        self.tracks = np.char.add(np.char.add(self.groups, "|"), data["train_track_id"].astype(str))
        self.g_train = data["train_geometry_raw"]
        self.g_test = data["test_geometry_raw"]
        self.mass = data["test_multiplicity"].astype(np.float64)
        self.rel = {s: data[f"train_oof_REL_S{s}"] for s in SEEDS}
        self.cand = {s: data[f"train_oof_RGB_GEO_BAND_S{s}"] for s in SEEDS}
        self.test_rel = data["test_submitted_ttc"]
        self.test_cand = data["test_candidate_ttc"]
        self.geo_train = np.clip(raw(self.g_train, "phi_trunc_aware"), -0.69, 8.0)
        self.geo_test = np.clip(raw(self.g_test, "phi_trunc_aware"), -0.69, 8.0)

    # ---------------------------------------------------------------- variants on folds
    def fold_variants(self) -> dict[str, dict[int, NDArray[np.float64]]]:
        """TTC predictions per variant and seed (alternatives built in phase space)."""
        out: dict[str, dict[int, NDArray[np.float64]]] = {
            "REL": dict(self.rel),
            "RGB_GEO_BAND": dict(self.cand),
            "REL_GEO_BAND": {s: self.d[f"train_oof_REL_GEO_BAND_S{s}"] for s in SEEDS},
            "REL_BAND": {s: self.d[f"train_oof_REL_BAND_S{s}"] for s in SEEDS},
        }
        gate = self.gate_mask(self.g_train)
        for w in BLENDS:
            out[f"BLEND{int(w * 100)}"] = {
                s: phase_to_ttc(st_.blend_phase(phase(self.cand[s]), phase(self.rel[s]), w))
                for s in SEEDS
            }
        out["GATE5"] = {s: np.where(gate, self.cand[s], self.rel[s]) for s in SEEDS}
        return out

    def test_variants(self) -> dict[str, NDArray[np.float64]]:
        """Test12 predictions of the same alternatives (seed 7 candidate, submitted REL)."""
        out = {"REL": self.test_rel, "RGB_GEO_BAND": self.test_cand}
        for w in BLENDS:
            out[f"BLEND{int(w * 100)}"] = phase_to_ttc(
                st_.blend_phase(phase(self.test_cand), phase(self.test_rel), w)
            )
        out["GATE5"] = np.where(self.gate_mask(self.g_test), self.test_cand, self.test_rel)
        return out

    @staticmethod
    def gate_mask(geometry: NDArray[Any]) -> NDArray[np.bool_]:
        """Geometry trusted: current box untruncated and at least five causal frames."""
        untruncated = (raw(geometry, "trunc_x_current") == 0) & (
            raw(geometry, "trunc_y_current") == 0
        )
        return untruncated & (raw(geometry, "n_frames") >= 5)

    def seed_mean_mid(self, by_seed: dict[int, NDArray[Any]]) -> NDArray[np.float64]:
        return np.mean([row_mid(self.truth, by_seed[s]) for s in SEEDS], axis=0)


# -------------------------------------------------------------------- H1 producer strength
def h1(st: Study) -> dict[str, Any]:
    d = st.d
    t = st.truth
    estimators_train = {
        "PAIR_R_OOF_rotation24": d["train_rgb_experts_oof"][:, 2],
        "PAIR_R_final40_in_sample": d["train_pair_in_sample"],
        "R_A5_OOF_rotation24": d["train_rgb_experts_oof"][:, 0],
        "R_C2F_OOF_rotation24": d["train_rgb_experts_oof"][:, 1],
        "R_CTX_OOF_rotation24": d["train_features"][:, 0].astype(np.float64),
        "R_CTX_final40_on_OOF_inputs": d["train_rctx_in_sample"],
        "E_CTX_OOF_rotation24": d["train_features"][:, 10].astype(np.float64),
        "PAIR_E_OOF_rotation24": d["train_event_experts_oof"][:, 2],
        "HEAD_REL_OOF_S7": phase(st.rel[7]),
        "HEAD_RELEASED_in_sample": phase(d["train_oof_RELEASED_IN_SAMPLE"]),
    }
    accuracy = {k: summarize(t, phase_to_ttc(v)) for k, v in estimators_train.items()}
    reliable_train = reliable(st.g_train)
    reliable_test = reliable(st.g_test)
    estimators_test = {
        "PAIR_R_final40_deployed": d["test_rgb_experts"][:, 2],
        "R_A5_final40_deployed": d["test_rgb_experts"][:, 0],
        "R_CTX_final40_deployed": d["test_rctx_location"],
        "E_CTX_final40_deployed": d["test_ectx_location"],
        "PAIR_E_final40_deployed": d["test_event_experts"][:, 2],
        "HEAD_REL_submitted": phase(st.test_rel),
        "RGB_GEO_BAND_candidate": phase(st.test_cand),
    }
    agreement: dict[str, Any] = {}
    for name, value in estimators_train.items():
        agreement[f"train::{name}"] = agree(value, st.geo_train, reliable_train, None)
    agreement["train::GEO_TRUNC_AWARE_vs_truth"] = agree(
        st.geo_train, st.phase_true, reliable_train, None
    )
    agreement["train::RGB_GEO_BAND_OOF_S7"] = agree(
        phase(st.cand[7]), st.geo_train, reliable_train, None
    )
    for name, value in estimators_test.items():
        agreement[f"test::{name}"] = agree(value, st.geo_test, reliable_test, st.mass)
    return {
        "train_accuracy": accuracy,
        "geometry_agreement_MiD": agreement,
        "reliable_geometry_definition": "no truncation in window (x,y) and n_frames >= 5",
        "reliable_share": {
            "train": float(reliable_train.mean()),
            "test_physical": float(reliable_test.mean()),
        },
    }


def reliable(geometry: NDArray[Any]) -> NDArray[np.bool_]:
    return (
        (raw(geometry, "trunc_x_any") == 0)
        & (raw(geometry, "trunc_y_any") == 0)
        & (raw(geometry, "n_frames") >= 5)
    )


def agree(
    value: NDArray[Any], reference: NDArray[Any], subset: NDArray[Any], weights: NDArray[Any] | None
) -> dict[str, float]:
    m = mid_phase(value, reference)
    w = np.ones(len(m)) if weights is None else weights
    return {
        "all_mean": st_.describe(m, w)["mean"],
        "all_median": st_.describe(m, w)["q50"],
        "reliable_mean": st_.describe(m[subset], w[subset])["mean"],
        "reliable_median": st_.describe(m[subset], w[subset])["q50"],
    }


# -------------------------------------------------------------------- H2 descriptors
def descriptors(st: Study) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = st.d
    train_len = track_lengths(d["train_group_id"], d["train_track_id"])
    test_len = track_lengths(d["test_group_id"], d["test_track_id"])
    columns: dict[str, tuple[NDArray[Any], NDArray[Any]]] = {
        "log_size": (raw(st.g_train, "log_size"), raw(st.g_test, "log_size")),
        "n_frames": (raw(st.g_train, "n_frames"), raw(st.g_test, "n_frames")),
        "trunc_current": (trunc_current(st.g_train), trunc_current(st.g_test)),
        "trunc_any": (trunc_any(st.g_train), trunc_any(st.g_test)),
        "track_queries": (train_len.astype(float), test_len.astype(float)),
        "phi_area_MiDunits": (1e4 * raw(st.g_train, "phi_area"), 1e4 * raw(st.g_test, "phi_area")),
        "abs_phi_trunc_aware_MiDunits": (1e4 * np.abs(st.geo_train), 1e4 * np.abs(st.geo_test)),
        "rms_residual": (raw(st.g_train, "rms_residual"), raw(st.g_test, "rms_residual")),
        "rgb_event_disagreement_MiD": (
            mid_phase(d["train_features"][:, 0], d["train_features"][:, 10]),
            mid_phase(d["test_features"][:, 0], d["test_features"][:, 10]),
        ),
        "rel_geometry_disagreement_MiD": (
            mid_phase(phase(st.rel[7]), st.geo_train),
            mid_phase(phase(st.test_rel), st.geo_test),
        ),
        "rgb_ctx_geometry_disagreement_MiD": (
            mid_phase(d["train_features"][:, 0], st.geo_train),
            mid_phase(d["test_features"][:, 0], st.geo_test),
        ),
        "candidate_rel_disagreement_MiD": (
            mid_phase(phase(st.cand[7]), phase(st.rel[7])),
            mid_phase(phase(st.test_cand), phase(st.test_rel)),
        ),
        "rel_prediction_ttc": (st.rel[7], st.test_rel),
        "rgb_width_logit_mean": (
            d["train_rgb_width_logits"].mean(1),
            d["test_rgb_width_logits"].mean(1),
        ),
    }
    rows = []
    for name, (a, b) in columns.items():
        rows.append({"descriptor": name, "population": "train_oof", **st_.describe(a)})
        rows.append({"descriptor": name, "population": "test_physical", **st_.describe(b)})
        rows.append({"descriptor": name, "population": "test_official", **st_.describe(b, st.mass)})
    mixes = {
        "train_REL_S7_pred": st_.shares(st_.predicted_band(st.rel[7]), st_.PRED_BAND_NAMES),
        "train_CAND_S7_pred": st_.shares(st_.predicted_band(st.cand[7]), st_.PRED_BAND_NAMES),
        "train_target": st_.shares(st_.predicted_band(st.truth), st_.PRED_BAND_NAMES),
        "test_REL_pred_physical": st_.shares(st_.predicted_band(st.test_rel), st_.PRED_BAND_NAMES),
        "test_REL_pred_official": st_.shares(
            st_.predicted_band(st.test_rel), st_.PRED_BAND_NAMES, st.mass
        ),
        "test_CAND_pred_official": st_.shares(
            st_.predicted_band(st.test_cand), st_.PRED_BAND_NAMES, st.mass
        ),
    }
    per_sequence = []
    for population, groups, prediction, weights in (
        ("train", st.groups, st.rel[7], np.ones(len(st.groups))),
        ("test", d["test_group_id"].astype(str), st.test_rel, st.mass),
    ):
        bands = st_.predicted_band(prediction)
        for seq in sorted(set(groups.tolist())):
            keep = groups == seq
            per_sequence.append(
                {
                    "population": population,
                    "sequence": seq,
                    "rows": float(weights[keep].sum()),
                    **st_.shares(bands[keep], st_.PRED_BAND_NAMES, weights[keep]),
                }
            )
    track_summary = {
        "train_tracks": int(len(np.unique(st.tracks))),
        "test_tracks": int(
            len(np.unique(np.char.add(d["test_group_id"].astype(str), d["test_track_id"])))
        ),
        "train_sequences": int(len(set(st.groups.tolist()))),
        "test_sequences": int(len(set(d["test_group_id"].astype(str).tolist()))),
    }
    return rows, {"band_mix": mixes, "per_sequence": per_sequence, "tracks": track_summary}


def trunc_current(geometry: NDArray[Any]) -> NDArray[np.float64]:
    return np.maximum(raw(geometry, "trunc_x_current"), raw(geometry, "trunc_y_current"))


def trunc_any(geometry: NDArray[Any]) -> NDArray[np.float64]:
    return np.maximum(raw(geometry, "trunc_x_any"), raw(geometry, "trunc_y_any"))


# -------------------------------------------------------------------- H2 reweighting
def weightings(st: Study) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    d = st.d
    edges = np.quantile(raw(st.g_train, "log_size"), [1 / 3, 2 / 3])

    def cells(prediction: NDArray[Any], geometry: NDArray[Any]) -> NDArray[np.int64]:
        size = np.digitize(raw(geometry, "log_size"), edges)
        return st_.cell_keys(
            [
                st_.predicted_band(prediction),
                size,
                trunc_current(geometry).astype(int),
                (raw(geometry, "n_frames") >= 5).astype(int),
            ],
            [6, 3, 2, 2],
        )

    primary, info_primary = st_.stratified_weights(
        cells(st.rel[7], st.g_train), cells(st.test_rel, st.g_test), st.mass
    )
    band_only, info_band = st_.stratified_weights(
        st_.predicted_band(st.rel[7]), st_.predicted_band(st.test_rel), st.mass
    )

    def covariates(
        rel: NDArray[Any],
        features: NDArray[Any],
        width: NDArray[Any],
        geometry: NDArray[Any],
        geo: NDArray[Any],
        lengths: NDArray[Any],
        event: bool,
    ) -> NDArray[np.float64]:
        p = np.clip(phase(rel), -0.5, 0.5)
        cols = [
            p,
            np.clip(features[:, 0], -0.5, 0.5),
            width[:, 0],
            width[:, 1],
            np.clip(raw(geometry, "phi_area"), -0.5, 0.5),
            np.clip(geo, -0.5, 0.5),
            raw(geometry, "n_frames"),
            raw(geometry, "span_s"),
            raw(geometry, "trunc_x_current"),
            raw(geometry, "trunc_y_current"),
            raw(geometry, "trunc_x_any"),
            raw(geometry, "trunc_y_any"),
            raw(geometry, "log_size"),
            np.minimum(raw(geometry, "rms_residual"), 0.5),
            np.log1p(np.minimum(mid_phase(p, geo), 5000.0)),
            np.log(lengths),
        ]
        if event:
            cols += [
                np.clip(features[:, 10], -0.5, 0.5),
                np.log1p(np.minimum(mid_phase(features[:, 0], features[:, 10]), 5000.0)),
            ]
        return np.stack(cols, 1).astype(np.float64)

    train_len = track_lengths(d["train_group_id"], d["train_track_id"]).astype(float)
    test_len = track_lengths(d["test_group_id"], d["test_track_id"]).astype(float)
    out = {"NONE": np.ones(len(st.truth)), "STRAT_PRIMARY": primary, "PRED_BAND_ONLY": band_only}
    info: dict[str, Any] = {"STRAT_PRIMARY": info_primary, "PRED_BAND_ONLY": info_band}
    for name, event in (("LOGIT_NO_EVENT", False), ("LOGIT_WITH_EVENT", True)):
        xtr = covariates(
            st.rel[7],
            d["train_features"],
            d["train_rgb_width_logits"],
            st.g_train,
            st.geo_train,
            train_len,
            event,
        )
        xte = covariates(
            st.test_rel,
            d["test_features"],
            d["test_rgb_width_logits"],
            st.g_test,
            st.geo_test,
            test_len,
            event,
        )
        out[name], info[name] = st_.logistic_weights(xtr, xte, st.mass)
    info["size_tercile_edges_log_size"] = edges.tolist()
    return out, info


def reweighted(
    st: Study,
    weights: dict[str, NDArray[np.float64]],
    variants: dict[str, dict[int, NDArray[np.float64]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    mids = {name: st.seed_mean_mid(v) for name, v in variants.items()}
    mids7 = {name: row_mid(st.truth, v[7]) for name, v in variants.items()}
    extra = {
        "RGB_EXPERT": row_mid(st.truth, phase_to_ttc(st.d["train_features"][:, 0])),
        "GEO_TRUNC_AWARE": row_mid(st.truth, phase_to_ttc(st.geo_train)),
        "RELEASED_IN_SAMPLE": row_mid(st.truth, st.d["train_oof_RELEASED_IN_SAMPLE"]),
    }
    boots: dict[str, Any] = {}
    for wname, w in weights.items():
        for name in mids:
            for label, m in (("seedmean", mids[name]), ("S7", mids7[name])):
                rows.append(
                    {"weighting": wname, "variant": name, "seeds": label}
                    | st_.weighted_overall(m, st.bands, w)
                )
        for name, m in extra.items():
            rows.append(
                {"weighting": wname, "variant": name, "seeds": "none"}
                | st_.weighted_overall(m, st.bands, w)
            )
        reps = st_.bootstrap_band_means(
            {"REL": mids7["REL"], **{k: mids7[k] for k in mids7 if k != "REL"}},
            w,
            st.bands,
            st.groups,
        )
        boots[wname] = reps
    return rows, boots


# -------------------------------------------------------------------- H2c three-cornered hat
def hat(st: Study) -> list[dict[str, Any]]:
    d = st.d
    rows = []
    train_group = group4(st_.predicted_band(st.rel[7]))
    test_group = group4(st_.predicted_band(st.test_rel))
    sides = {
        "train": {
            "REL": phase(st.rel[7]),
            "CAND": phase(st.cand[7]),
            "RCTX": d["train_features"][:, 0].astype(np.float64),
            "PAIR": d["train_rgb_experts_oof"][:, 2],
            "GEO": st.geo_train,
            "EVT": d["train_features"][:, 10].astype(np.float64),
            "group": train_group,
            "weights": np.ones(len(train_group)),
        },
        "test": {
            "REL": phase(st.test_rel),
            "CAND": phase(st.test_cand),
            "RCTX": d["test_features"][:, 0].astype(np.float64),
            "PAIR": d["test_rgb_experts"][:, 2],
            "GEO": st.geo_test,
            "EVT": d["test_features"][:, 10].astype(np.float64),
            "group": test_group,
            "weights": st.mass,
        },
    }
    for clip in (CLIP, CLIP_SENSITIVITY):
        for side, v in sides.items():
            for g in (-1, 0, 1, 2, 3):
                keep = np.ones(len(v["group"]), bool) if g < 0 else v["group"] == g
                expanded = np.repeat(np.flatnonzero(keep), v["weights"][keep].astype(int))
                for target in ("REL", "CAND", "RCTX", "PAIR"):
                    est = st_.three_cornered_hat(
                        v[target][expanded], v["GEO"][expanded], v["EVT"][expanded], clip=clip
                    )
                    row = {
                        "clip": clip,
                        "side": side,
                        "pred_group": "all" if g < 0 else GROUPS4[g],
                        "rows": float(len(expanded)),
                        "target": target,
                        "sd_target_hat_MiD": 1e4 * est["sd_a"],
                        "sd_geo_hat_MiD": 1e4 * est["sd_b"],
                        "sd_evt_hat_MiD": 1e4 * est["sd_c"],
                    }
                    if side == "train":
                        truth = st.phase_true[expanded]
                        row["sd_target_true_MiD"] = 1e4 * st_.winsorized_sd(
                            v[target][expanded] - truth, clip=clip
                        )
                        row["sd_geo_true_MiD"] = 1e4 * st_.winsorized_sd(
                            v["GEO"][expanded] - truth, clip=clip
                        )
                        row["sd_evt_true_MiD"] = 1e4 * st_.winsorized_sd(
                            v["EVT"][expanded] - truth, clip=clip
                        )
                    rows.append(row)
    return rows


# -------------------------------------------------------------------- H3
def h3(st: Study) -> dict[str, Any]:
    d = st.d
    spec = importlib.util.spec_from_file_location("garl_official_score", scorer_path())
    if spec is None or spec.loader is None:
        raise ImportError(str(scorer_path()))
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    reference = [{"sample_token": str(i), "ttc": float(t)} for i, t in enumerate(st.truth)]
    checks = {}
    for name, ttc in (("REL_S7", st.rel[7]), ("RGB_GEO_BAND_S7", st.cand[7])):
        official = scorer.evaluate_rows(
            reference, {str(i): float(p) for i, p in enumerate(ttc)}, dT=0.1
        )
        mine = summarize(st.truth, ttc)
        checks[name] = {
            "official_overall": official["overall_MiD"],
            "vectorized_overall": mine["overall_MiD"],
            "abs_diff": abs(official["overall_MiD"] - mine["overall_MiD"]),
            "official_mean_MiD": official["mean_MiD"],
        }
    t = st.truth
    keys = np.char.add(np.char.add(st.tracks, "|"), d["train_timestamp_us"].astype(str))
    seq_time = np.char.add(np.char.add(st.groups, "|"), d["train_timestamp_us"].astype(str))
    test_keys = np.char.add(
        np.char.add(np.char.add(d["test_group_id"].astype(str), "|"), d["test_track_id"]),
        np.char.add("|", d["test_timestamp_us"].astype(str)),
    )
    test_bands = st_.predicted_band(st.test_rel)
    alias_rows = st.mass - 1
    return {
        "scorer_crosscheck": checks,
        "train_targets": {
            "rows": int(len(t)),
            "in_band_share": float((st.bands >= 0).mean()),
            "in_0_0p1": int(((t > 0) & (t <= 0.1)).sum()),
            "min": float(t.min()),
            "max": float(t.max()),
            "abs_gt_10": int((np.abs(t) > 10).sum()),
            "band_counts": {b: int((st.bands == i).sum()) for i, b in enumerate(BANDS)},
            "crucial_0_1": int(((t > 0) & (t <= 1)).sum()),
            "crucial_1_3": int(((t > 1) & (t <= 3)).sum()),
        },
        "train_duplicates": {
            "same_sequence_track_timestamp": int(len(keys) - len(np.unique(keys))),
            "same_sequence_timestamp_any_track": int(len(seq_time) - len(np.unique(seq_time))),
        },
        "test_population": {
            "physical": int(len(st.mass)),
            "official": int(st.mass.sum()),
            "aliases": int(alias_rows.sum()),
            "multiplicity_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(st.mass, return_counts=True), strict=True)
            },
            "physical_duplicate_sequence_track_timestamp": int(
                len(test_keys) - len(np.unique(test_keys))
            ),
            "alias_share_by_submitted_pred_band": {
                n: float(alias_rows[test_bands == i].sum() / max(alias_rows.sum(), 1))
                for i, n in enumerate(st_.PRED_BAND_NAMES)
            },
        },
        "official_mean_constraint": mean_constraint(),
    }


def mean_constraint() -> dict[str, Any]:
    """Range of target-band shares compatible with the official mean_MiD (all rows in band)."""
    from scipy.optimize import linprog

    mids = np.asarray([OFFICIAL_REL[b] for b in BANDS])
    out: dict[str, Any] = {
        "assumption": "every scored Test12 target lies in one of the four bands",
        "equation": "sum_b p_b * MiD_b = mean_MiD, sum_b p_b = 1, p_b >= 0",
    }
    for i, b in enumerate(BANDS):
        bounds = []
        for sign in (1.0, -1.0):
            cost = np.zeros(4)
            cost[i] = sign
            res = linprog(
                cost,
                A_eq=np.stack((mids, np.ones(4))),
                b_eq=[OFFICIAL_REL["mean_MiD"], 1.0],
                bounds=[(0, 1)] * 4,
            )
            bounds.append(float(res.x[i]) if res.success else float("nan"))
        out[f"share_{b}_range"] = [bounds[0], bounds[1]]
    return out


# -------------------------------------------------------------------- H4
def h4(st: Study) -> dict[str, Any]:
    crucial = st.bands == 0
    mid = st.seed_mean_mid(st.rel)
    mid_c = np.where(crucial, mid, 0.0)
    tracks, inverse = np.unique(st.tracks, return_inverse=True)
    per_track = np.bincount(inverse, weights=mid_c)
    per_track_rows = np.bincount(inverse, weights=crucial.astype(float))
    crucial_tracks = per_track_rows > 0
    seqs, seq_inverse = np.unique(st.groups, return_inverse=True)
    per_seq = np.bincount(seq_inverse, weights=mid_c)
    per_seq_rows = np.bincount(seq_inverse, weights=crucial.astype(float))
    order = np.argsort(per_track)[::-1][:20]
    top_tracks = [
        {
            "track": str(tracks[i]),
            "crucial_rows": int(per_track_rows[i]),
            "share_of_crucial_sum": float(per_track[i] / per_track.sum()),
            "mean_crucial_MiD": float(per_track[i] / per_track_rows[i]),
        }
        for i in order
    ]
    t = st.truth
    n_frames = raw(st.g_train, "n_frames")
    strata = {
        "T_0_1": crucial & (t <= 1),
        "T_1_3": crucial & (t > 1),
        "truncated_current": crucial & (trunc_current(st.g_train) > 0),
        "untruncated_current": crucial & (trunc_current(st.g_train) == 0),
        "n_frames_lt5": crucial & (n_frames < 5),
        "n_frames_ge5": crucial & (n_frames >= 5),
        "REL_pred_not_crucial": crucial & (st_.predicted_band(st.rel[7]) != 0),
        "REL_pred_crucial": crucial & (st_.predicted_band(st.rel[7]) == 0),
    }
    cand_mid = st.seed_mean_mid(st.cand)
    breakdown = {
        name: {
            "rows": int(mask.sum()),
            "share_of_crucial_rows": float(mask.sum() / crucial.sum()),
            "REL_mean_MiD": float(mid[mask].mean()),
            "CAND_mean_MiD": float(cand_mid[mask].mean()),
            "share_of_REL_crucial_sum": float(mid[mask].sum() / mid[crucial].sum()),
        }
        for name, mask in strata.items()
    }
    # Label-free proxies of the hard stratum on both sides (predicted crucial by REL).
    proxies = {}
    for side, pred, geometry, w in (
        ("train", st.rel[7], st.g_train, np.ones(len(st.truth))),
        ("test", st.test_rel, st.g_test, st.mass),
    ):
        pc = st_.predicted_band(pred) == 0
        proxies[side] = {
            "pred_crucial_share": float(w[pc].sum() / w.sum()),
            "pred_0_1_share": float(w[(pred > 0) & (pred <= 1)].sum() / w.sum()),
            "truncated_given_pred_crucial": float(
                w[pc & (trunc_current(geometry) > 0)].sum() / max(w[pc].sum(), 1e-9)
            ),
            "n_frames_lt5_given_pred_crucial": float(
                w[pc & (raw(geometry, "n_frames") < 5)].sum() / max(w[pc].sum(), 1e-9)
            ),
            "geometry_expanding_fast_share": float(
                w[raw(geometry, "phi_trunc_aware") > ttc_to_phase(np.asarray([3.0]))[0]].sum()
                / w.sum()
            ),
        }
    return {
        "crucial_rows": int(crucial.sum()),
        "crucial_tracks": int(crucial_tracks.sum()),
        "top10pct_tracks_share": st_.concentration_share(per_track[crucial_tracks], 0.10),
        "top5_tracks_share": float(np.sort(per_track)[::-1][:5].sum() / per_track.sum()),
        "top5_sequences_share": float(np.sort(per_seq)[::-1][:5].sum() / per_seq.sum()),
        "sequences_with_crucial_rows": int((per_seq_rows > 0).sum()),
        "per_sequence": [
            {
                "sequence": str(seqs[i]),
                "crucial_rows": int(per_seq_rows[i]),
                "share_of_crucial_sum": float(per_seq[i] / per_seq.sum()),
                "mean_crucial_MiD": float(per_seq[i] / per_seq_rows[i])
                if per_seq_rows[i]
                else None,
            }
            for i in np.argsort(per_seq)[::-1]
        ],
        "top_tracks": top_tracks,
        "strata": breakdown,
        "label_free_proxies": proxies,
    }


# -------------------------------------------------------------------- decision diagnostics
def decision(
    st: Study,
    variants: dict[str, dict[int, NDArray[np.float64]]],
    weights: dict[str, NDArray[np.float64]],
    boots: dict[str, Any],
) -> dict[str, Any]:
    rel_p, cand_p = phase(st.rel[7]), phase(st.cand[7])
    dis = mid_phase(cand_p, rel_p)
    e_rel, e_cand = 1e4 * (rel_p - st.phase_true), 1e4 * (cand_p - st.phase_true)
    by_bin = st_.conditional_wins(e_cand, e_rel, dis, DIS_EDGES)
    test_dis = mid_phase(phase(st.test_cand), phase(st.test_rel))
    bins = np.digitize(test_dis, DIS_EDGES[1:])
    test_share = [float(st.mass[bins == i].sum() / st.mass.sum()) for i in range(len(DIS_EDGES))]
    train_bins = np.digitize(dis, DIS_EDGES[1:])
    train_share = [float((train_bins == i).mean()) for i in range(len(DIS_EDGES))]
    for row, share, tshare in zip(by_bin, test_share, train_share, strict=True):
        row["test_share"] = share
        row["train_share"] = tshare
    weighted_win = float(sum(r["win_rate"] * r["test_share"] for r in by_bin))
    weighted_change = float(sum(r["mean_change"] * r["test_share"] for r in by_bin))
    by_band = []
    rel_bands = st_.predicted_band(st.rel[7])
    for i, name in enumerate(st_.PRED_BAND_NAMES):
        keep = rel_bands == i
        for row in st_.conditional_wins(e_cand[keep], e_rel[keep], dis[keep], DIS_EDGES):
            by_band.append({"rel_pred_band": name, **row})
    per_sequence = []
    m_rel, m_cand = row_mid(st.truth, st.rel[7]), row_mid(st.truth, st.cand[7])
    for seq in sorted(set(st.groups.tolist())):
        keep = st.groups == seq
        per_sequence.append(
            {
                "sequence": seq,
                "rows": int(keep.sum()),
                "win_rate": float((np.abs(e_cand[keep]) < np.abs(e_rel[keep])).mean()),
                "mean_MiD_REL": float(np.nanmean(m_rel[keep])),
                "mean_MiD_CAND": float(np.nanmean(m_cand[keep])),
                "overall_REL": st_.weighted_overall(m_rel[keep], st.bands[keep])["overall"],
                "overall_CAND": st_.weighted_overall(m_cand[keep], st.bands[keep])["overall"],
                "mean_disagreement_MiD": float(dis[keep].mean()),
            }
        )
    wins = np.asarray([r["win_rate"] for r in per_sequence])
    better = np.asarray([r["mean_MiD_CAND"] < r["mean_MiD_REL"] for r in per_sequence])
    # Test12: movement of each alternative relative to the submitted predictions.
    test_alts = st.test_variants()
    movement = {}
    for name, ttc in test_alts.items():
        m = mid_phase(phase(ttc), phase(st.test_rel))
        bands = st_.predicted_band(st.test_rel)
        movement[name] = {
            "mean_MiD_vs_submitted": st_.describe(m, st.mass)["mean"],
            "median": st_.describe(m, st.mass)["q50"],
            "p90": float(np.quantile(np.repeat(m, st.mass.astype(int)), 0.9)),
            "sign_changes": int(((ttc > 0) != (st.test_rel > 0)).sum()),
            "by_submitted_pred_band_mean": {
                n: float((m * st.mass)[bands == i].sum() / max(st.mass[bands == i].sum(), 1))
                for i, n in enumerate(st_.PRED_BAND_NAMES)
            },
            "pred_band_mix_official": st_.shares(
                st_.predicted_band(ttc), st_.PRED_BAND_NAMES, st.mass
            ),
        }
    alternatives = alternatives_table(st, variants)
    projections = project(boots)
    return {
        "conditional_wins_by_disagreement": by_bin,
        "test_weighted_win_rate": weighted_win,
        "test_weighted_mean_change_MiD": weighted_change,
        "conditional_wins_by_rel_pred_band": by_band,
        "per_sequence": per_sequence,
        "sequences_win_rate_gt_half": int((wins > 0.5).sum()),
        "sequences_mean_MiD_better": int(better.sum()),
        "per_sequence_win_rate_quantiles": st_.describe(wins),
        "test_movement": movement,
        "alternatives": alternatives,
        "projections": projections,
    }


def alternatives_table(
    st: Study, variants: dict[str, dict[int, NDArray[np.float64]]]
) -> list[dict[str, Any]]:
    from operational.phase_geometry.metrics import bootstrap_overall

    mids = {name: st.seed_mean_mid(v) for name, v in variants.items()}
    rows = []
    for name, by_seed in variants.items():
        scores = [summarize(st.truth, by_seed[s]) for s in SEEDS]
        row: dict[str, Any] = {
            "variant": name,
            "overall_mean": float(np.mean([s["overall_MiD"] for s in scores])),
            "overall_sd": float(np.std([s["overall_MiD"] for s in scores], ddof=1)),
            **{f"MiD{b}": float(np.mean([s[f"MiD{b}"] for s in scores])) for b in BANDS},
            "post_hoc": name.startswith(("BLEND", "GATE")),
            "sequences_better_than_REL_S7": int(
                sum(
                    np.nanmean(row_mid(st.truth, by_seed[7])[st.groups == g])
                    < np.nanmean(row_mid(st.truth, st.rel[7])[st.groups == g])
                    for g in sorted(set(st.groups.tolist()))
                )
            ),
        }
        for comparator in ("REL", "RGB_GEO_BAND"):
            if comparator == name:
                continue
            boot = bootstrap_overall(mids[name] - mids[comparator], st.bands, st.groups)
            row[f"delta_vs_{comparator}"] = boot["point"]
            row[f"delta_vs_{comparator}_ci95"] = [boot["ci95_low"], boot["ci95_high"]]
            row[f"delta_vs_{comparator}_band"] = boot["band_point"]
        rows.append(row)
    return rows


def project(boots: dict[str, Any]) -> list[dict[str, Any]]:
    """Multiplicative per-band transfer of fold ratios to the official REL bands (seed 7)."""
    rows = []
    for wname, reps in boots.items():
        rel = reps["REL"]
        point_rel = np.nanmean(rel, axis=0)
        for name, rep in reps.items():
            if name == "REL":
                continue
            ratio = rep / rel
            projected = st_.project_multiplicative(OFFICIAL_REL, ratio)
            point_ratio = np.nanmedian(ratio, axis=0)
            rows.append(
                {
                    "weighting": wname,
                    "variant": name,
                    "ratio_band_median": {b: float(point_ratio[i]) for i, b in enumerate(BANDS)},
                    "projected_overall_median": float(np.nanmedian(projected)),
                    "projected_overall_ci95": st_.ci(projected),
                    "p_worse_than_submitted": float(
                        np.mean(projected[np.isfinite(projected)] > OFFICIAL_REL["overall"])
                    ),
                    "fold_REL_bands_mean_of_replicates": {
                        b: float(point_rel[i]) for i, b in enumerate(BANDS)
                    },
                }
            )
    return rows


def garl_64() -> dict[str, Any]:
    frame = pd.read_csv(GARL_64)
    truth = frame["target_ttc"].to_numpy(np.float64)
    out = {"source": str(GARL_64), "rows": len(frame), "garl_in_sample": True}
    for name in ("Garl_full", "PHASE_ONLY_V16_S7", "R_CTX"):
        out[name] = summarize(truth, frame[name].to_numpy(np.float64))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        flat.append(
            {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
        )
    pd.DataFrame(flat).to_csv(path, index=False, lineterminator="\n", float_format="%.6g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    st = Study(load(args.work))
    variants = st.fold_variants()
    results: dict[str, Any] = {
        "schema": "level_gap_results_v1",
        "private_test_labels_read": False,
        "official_rel_test": OFFICIAL_REL,
        "garl_test": GARL_TEST,
        "band_weights": BAND_WEIGHTS,
    }
    results["H1"] = h1(st)
    desc_rows, desc_extra = descriptors(st)
    results["H2_descriptors"] = desc_extra
    weights, winfo = weightings(st)
    results["H2_weightings"] = winfo
    rw_rows, boots = reweighted(st, weights, variants)
    results["H2_reweighted"] = rw_rows
    results["H2_reweighted_REL_ci95"] = {
        w: {
            "REL_S7_overall_ci95": st_.ci(st_.overall_of(b["REL"])),
            "CAND_minus_REL_S7_ci95": st_.ci(
                st_.overall_of(b["RGB_GEO_BAND"]) - st_.overall_of(b["REL"])
            ),
        }
        for w, b in boots.items()
    }
    hat_rows = hat(st)
    results["H2_hat"] = hat_rows
    results["H3"] = h3(st)
    results["H3_garl_64_in_sample"] = garl_64()
    results["H4"] = h4(st)
    results["decision"] = decision(st, variants, weights, boots)
    from . import addenda

    folds7 = {name: v[7] for name, v in variants.items() if name != "REL_BAND"}
    tests = st.test_variants()
    results["addenda"] = {
        "A1_consistency": addenda.consistency(st),
        "A2_arbiter": addenda.arbiter(st),
        "A3_additive_projection": [
            *addenda.additive_projection(
                st, {k: v for k, v in folds7.items() if k in tests}, tests, "target"
            ),
            *addenda.additive_projection(
                st, {k: v for k, v in folds7.items() if k in tests}, tests, "predicted"
            ),
        ],
        "A4_difficulty_emulation": addenda.emulation(st, folds7),
        "A5_uninformative_moves": addenda.noise_bound(st, tests),
    }
    docs = args.docs
    docs.mkdir(parents=True, exist_ok=True)
    write_csv(docs / "DESCRIPTORS.csv", desc_rows)
    write_csv(docs / "REWEIGHTED.csv", rw_rows)
    write_csv(docs / "HAT.csv", hat_rows)
    write_csv(docs / "ALTERNATIVES.csv", results["decision"]["alternatives"])
    write_csv(docs / "PROJECTIONS.csv", results["decision"]["projections"])
    write_csv(
        docs / "CONDITIONAL_WINS.csv", results["decision"]["conditional_wins_by_rel_pred_band"]
    )
    write_csv(docs / "SEQUENCE_WINS.csv", results["decision"]["per_sequence"])
    write_csv(docs / "CONSISTENCY.csv", results["addenda"]["A1_consistency"])
    receipt = json.loads((args.work / "EXTRACT_RECEIPT.json").read_text(encoding="utf-8"))
    results["inputs_receipt"] = receipt
    (docs / "RESULTS.json").write_text(
        json.dumps(json_safe(results), indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    print("written", docs / "RESULTS.json")


if __name__ == "__main__":
    main()
