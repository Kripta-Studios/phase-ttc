"""TRAIN-only phase masses and priors, kept outside the neural input contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.models.phase_distribution_v11 import benchmark_mass, phase_edges, ttc_to_phase
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork


@dataclass(frozen=True)
class Supervision:
    """Ordered supervised targets and population weights; never model input channels."""

    tokens: tuple[str, ...]
    target_phase: Tensor
    mass: Tensor
    prior_bin_mass: Tensor
    outside_support_mass: float
    identity_sha256: str


def train_supervision(view: TrainingView, labels: pd.DataFrame, *, expanded: bool) -> Supervision:
    """Fit strict old-TRAIN or explicitly present-bucket-normalized expanded-TRAIN mass."""
    required = {"sample_token", "sequence_id", "target_ttc", "bucket"}
    if not required <= set(labels.columns) or labels.sample_token.duplicated().any():
        raise ValueError("unique row-aligned TRAIN supervision required")
    indexed = labels.set_index("sample_token")
    if not set(view.tokens) <= set(indexed.index):
        raise ValueError("TRAIN labels missing required tokens")
    rows = indexed.loc[list(view.tokens)]
    metadata = view.reader.metadata.set_index("sample_token").loc[list(view.tokens)]
    if not np.array_equal(rows.sequence_id.astype(str), metadata.sequence_id.astype(str)):
        raise ValueError("label and count sequence lineage differ")
    names = sorted(set(rows.sequence_id.astype(str)))
    sequence = torch.tensor([names.index(str(seq)) for seq in rows.sequence_id], dtype=torch.int64)
    bucket_names = ("crucial", "small", "large", "negative")
    if not set(rows.bucket) <= set(bucket_names):
        raise ValueError("unknown benchmark bucket")
    target_values = rows.target_ttc.to_numpy(dtype=np.float64)
    expected_buckets = np.select(
        [target_values < 0, target_values <= 3, target_values <= 6],
        ["negative", "crucial", "small"],
        "large",
    )
    if not np.array_equal(rows.bucket.astype(str), expected_buckets):
        raise ValueError("bucket metadata disagrees with the signed TTC target")
    buckets = torch.tensor([bucket_names.index(str(bucket)) for bucket in rows.bucket])
    phase = ttc_to_phase(torch.tensor(rows.target_ttc.to_numpy(dtype=np.float64)))
    if expanded:
        mass = torch.zeros(len(rows), dtype=torch.float64)
        prescribed = (0.5, 0.3, 0.1, 0.1)
        for seq in sequence.unique():
            present = [k for k in range(4) if bool(((sequence == seq) & (buckets == k)).any())]
            denominator = sum(prescribed[k] for k in present)
            for k in present:
                keep = (sequence == seq) & (buckets == k)
                mass[keep] = prescribed[k] / denominator / len(names) / int(keep.sum())
    else:
        mass = benchmark_mass(sequence, buckets)
    edges = phase_edges(dtype=torch.float64)
    bin_ids = torch.searchsorted(edges, phase, right=True).sub(1).clamp(0, len(edges) - 2)
    prior = torch.zeros(len(edges) - 1, dtype=torch.float64)
    prior.scatter_add_(0, bin_ids, mass)
    outside = (phase < edges[0]) | (phase > edges[-1])
    identity = object_digest(
        {
            "training_view": view.identity,
            "ordered_tokens": view.tokens,
            "target_phase": phase.tolist(),
            "mass": mass.tolist(),
            "prior_bin_mass": prior.tolist(),
            "estimand": "present_bucket_normalized_TRAIN" if expanded else "strict_original_TRAIN",
            "outside_support_prior_assignment": "nearest_edge_bin; no target row discarded",
        }
    )
    return Supervision(
        view.tokens, phase.float(), mass, prior, float(mass[outside].sum()), identity
    )


@torch.no_grad()
def initialize_train_prior(model: ProjectivePhaseNetwork, supervision: Supervision) -> None:
    """Apply the fixed N(0,.001) readout and TRAIN-only log-bin-mass bias."""
    classifier = model.direct[-1]
    if not isinstance(classifier, nn.Linear) or classifier.bias is None:
        raise ValueError("unexpected direct classifier schema")
    classifier.weight.normal_(mean=0, std=0.001)
    bias = (supervision.prior_bin_mass + 1e-3).log()
    classifier.bias.copy_(bias.to(classifier.bias))
