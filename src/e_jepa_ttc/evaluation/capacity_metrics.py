"""Prospective expanded-cohort estimand; no changes to historical strict metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage63_65 import BUCKETS, benchmark_phase


def present_bucket_mass(target_ttc: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Give each sequence equal mass, renormalizing only its present TTC buckets."""
    target = np.asarray(target_ttc, dtype=np.float64)
    sequence = np.asarray(sequences, dtype=str)
    if target.ndim != 1 or target.shape != sequence.shape or not len(target):
        raise ValueError("aligned nonempty target and sequence vectors required")
    benchmark_phase(target)
    mass = np.zeros(len(target), dtype=np.float64)
    names = np.unique(sequence)
    for seq in names:
        masks = [(sequence == seq) & (target > lo) & (target <= hi) for _, lo, hi, _ in BUCKETS]
        denominator = sum(
            bucket[3] for bucket, mask in zip(BUCKETS, masks, strict=True) if mask.any()
        )
        for bucket, mask in zip(BUCKETS, masks, strict=True):
            if mask.any():
                mass[mask] = bucket[3] / denominator / len(names) / int(mask.sum())
    if (mass <= 0).any() or not np.isclose(mass.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("target outside registered buckets; rows cannot be discarded")
    return mass


def acquisition_bootstrap(
    metadata: pd.DataFrame,
    losses: np.ndarray,
    *,
    draws: int = 8192,
    seed: int = 70,
) -> tuple[np.ndarray, dict[str, object]]:
    """Paired acquisition bootstrap of sequence scores, preserving related sequences.

    Temporal rows are not independent replicates. This cluster bootstrap keeps
    each sequence's fixed row/bucket estimand intact, including absent buckets.
    The draw multiplicities fully reproduce all columns with no model-specific
    resampling. It is Monte Carlo uncertainty, not an exact sequence calculation.
    """
    required = {"sample_token", "sequence_id", "acquisition_group", "target_ttc_s"}
    if not required <= set(metadata) or metadata.sample_token.duplicated().any():
        raise ValueError("unique row identities and acquisition groups required")
    if metadata[list(required)].isna().to_numpy().any():
        raise ValueError("missing row or acquisition identity")
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(metadata) or not np.isfinite(values).all():
        raise ValueError("all loss columns must be complete, finite and paired")
    if type(draws) is not int or draws < 2:
        raise ValueError("at least two bootstrap draws required")
    if (metadata.groupby("sequence_id").acquisition_group.nunique() != 1).any():
        raise ValueError("a sequence cannot belong to multiple acquisitions")
    sequence = metadata.sequence_id.astype(str).to_numpy()
    groups = metadata.acquisition_group.astype(str).to_numpy()
    names = sorted(set(sequence))
    group_names = sorted(set(groups))
    if len(group_names) < 3:
        raise ValueError("expanded uncertainty requires at least three acquisitions")
    mass = present_bucket_mass(metadata.target_ttc_s.to_numpy(np.float64), sequence)
    # Ratio of cluster totals to cluster sequence counts retains the sequence-macro estimand.
    totals = np.stack([len(names) * (mass[groups == g] @ values[groups == g]) for g in group_names])
    counts = np.array([len(set(sequence[groups == g])) for g in group_names])
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(group_names), np.full(len(group_names), 1 / len(group_names)), draws
    )
    sampled = (multiplicities @ totals) / (multiplicities @ counts)[:, None]
    support = {
        seq: [
            name
            for name, lo, hi, _ in BUCKETS
            if (
                (sequence == seq)
                & (metadata.target_ttc_s.to_numpy() > lo)
                & (metadata.target_ttc_s.to_numpy() <= hi)
            ).any()
        ]
        for seq in names
    }
    return sampled, {
        "method": "paired_acquisition_cluster_bootstrap_of_fixed_sequence_estimands",
        "seed": seed,
        "draws": draws,
        "acquisition_order": group_names,
        "sequence_count": len(names),
        "multiplicities": multiplicities.tolist(),
        "present_buckets": support,
        "support_rejected_draws": 0,
        "temporal_rows_resampled_independently": False,
        "exact": False,
        "fraction_meaning": "Monte Carlo bootstrap fraction, not Bayesian posterior",
    }
