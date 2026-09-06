"""Prospective replication selection and protected coordinated decode on synthetic files only."""

import copy
import json
from dataclasses import asdict, replace

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.producer_lineage_v11 import ProducerRecord
from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, object_digest
from e_jepa_ttc.data.confirmation_access_v11 import SealedTrainPartitions, verify_confirmation_lock
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.stage73_factorial import Availability
from e_jepa_ttc.training.stage76_confirmation import (
    ReplicationSummary,
    confirmation_prerequisites,
    select_confirmation_family,
)


def summary(scores=(140.0, 138.0, 139.0), latency=20.0):
    return ReplicationSummary(scores, (1.0, 1.0, 1.0), 0.005, 3.0, 146.0, latency, "b" * 64)


def sign(value):
    return {**value, "artifact_sha256": object_digest(value)}


def test_replication_gate_keeps_every_seed_and_has_no_ci_prerequisite():
    assert summary().stable
    assert not summary((100, 147, 100)).stable
    assert not replace(summary(), mean_sign_error_increase=0.0051).stable
    assert not replace(summary(), mean_crucial_delta=3.01).stable
    assert not replace(summary(), seed_finite_fractions=(1, 1, 0.99)).stable
    assert select_confirmation_family({"C_DIRECT": None, "C_PPS": summary()}) == "C_PPS"
    result = confirmation_prerequisites(
        {"C_DIRECT": summary(), "C_PPS": None},
        freshness_verified=False,
        credible_garl_available=True,
    )
    assert result["reason"] == "CONFIRMATION_ACQUISITION_OR_ANCESTRY_UNVERIFIED"


def test_selection_uses_mean_loss_then_latency_then_direct():
    direct = summary((139, 139, 139), latency=25)
    pps = summary((138, 138, 138), latency=30)
    assert select_confirmation_family({"C_DIRECT": direct, "C_PPS": pps}) == "C_PPS"
    pps = summary((138.95, 138.95, 138.95), latency=30)
    assert select_confirmation_family({"C_DIRECT": direct, "C_PPS": pps}) == "C_DIRECT"
    pps = replace(pps, deployment_latency_ms=20)
    assert select_confirmation_family({"C_DIRECT": direct, "C_PPS": pps}) == "C_PPS"
    assert select_confirmation_family({"C_DIRECT": direct, "C_PPS": direct}) == "C_DIRECT"


@pytest.fixture
def setup(tmp_path):
    groups = tuple(f"hold{i}" for i in range(6))
    roles = TrainRoles(
        tuple((x, x) for x in ("train", *groups)), ("train",), ("train",), (), groups, (), "fixture"
    )
    sources = {}

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(sign(value)))
        record = binding(path)
        sources[record["path"]] = record
        return record

    freshness = write(
        "freshness.json",
        dict(
            status="VERIFIED_DISJOINT_UNEXPOSED_TRAIN",
            role_sha256="a" * 64,
            confirmation_groups=groups,
            audited_ancestor_roots=["history"],
            acquisition_checks_complete=True,
        ),
    )
    garl = write(
        "garl.json",
        dict(
            status="CREDIBLE_TRAIN_ONLY_SOURCE_AUDITED",
            initialized_from_scratch=True,
            event_channels=40,
            architecture="ResNet50",
            native_batch=32,
            profile_selection_scope="FINAL_TRAIN_INNER_ONLY",
            selected_refit_updates=4000,
        ),
    )
    replication = write(
        "replication.json",
        dict(
            family="C_DIRECT",
            evaluation_scope="OLD_DEVELOPMENT",
            replication_summary=asdict(summary()),
        ),
    )
    endpoints, nodes = [], {}
    for family in ("C_DIRECT", "GARL"):
        for seed in (7, 13, 23):
            name = f"{family}_{seed}"
            path = tmp_path / f"{name}.weights"
            path.write_bytes(name.encode())  # Synthetic endpoint fixture, not a scientific model.
            weights = binding(path)
            sources[weights["path"]] = weights
            record = ProducerRecord(
                name,
                "garl" if family == "GARL" else "architecture",
                "a" * 64,
                "c" * 40,
                False,
                ("train",),
                (),
                ("train",),
                ("train",),
                (),
                (),
                (weights,),
                seed=seed,
            )
            nodes[name] = record.signed_record()
            updates = 4000 if family == "GARL" else 8000
            endpoints.append(
                dict(
                    family=family,
                    seed=seed,
                    completed_update=updates,
                    prescribed_update=updates,
                    technical=False,
                    inference_weights=weights,
                    producer_id=name,
                    pretext_completed_updates=0,
                )
            )
    outputs = {}
    for key, ids in (
        ("train_labels", ["train"]),
        ("confirmation_metadata_SEALED", groups),
        ("confirmation_labels_SEALED", groups),
    ):
        path = tmp_path / f"{key}.parquet"
        frame = pd.DataFrame(dict(sample_token=[f"token_{g}" for g in ids], sequence_id=ids))
        if "labels" in key:
            frame["ttc"] = 1.25
        frame.to_parquet(path, index=False)
        outputs[key] = binding(path)
    partitions = sign(dict(kind="administrative_public_TRAIN_role_partitions", outputs=outputs))
    lock = sign(
        dict(
            kind="S76_SIX_ENDPOINTS_LOCKED",
            role_sha256="a" * 64,
            confirmation_groups=groups,
            freshness_evidence=freshness,
            garl_comparator_evidence=garl,
            replication_evidence={"C_DIRECT": replication, "C_PPS": None},
            selected_family="C_DIRECT",
            availability=asdict(Availability(False, False, "d" * 64, "e" * 64)),
            selected_treatment="S72-DIRECT",
            endpoints=endpoints,
            partition_manifest_sha256=partitions["artifact_sha256"],
        )
    )
    access = SealedTrainPartitions(
        partitions, tmp_path / "evaluation", expected_manifest_sha256=partitions["artifact_sha256"]
    )
    kwargs = dict(roles=roles, role_sha256="a" * 64, lineage=nodes, allowed_sources=sources)
    return access, lock, kwargs, partitions


@pytest.mark.parametrize(
    "fault",
    [
        "missing_endpoint",
        "partial",
        "duplicate",
        "technical",
        "seed",
        "family",
        "pretext",
        "ancestry",
    ],
)
def test_bad_endpoint_or_lineage_never_decodes_confirmation(setup, fault, monkeypatch):
    access, lock, kwargs, _ = setup
    broken = copy.deepcopy(lock)
    if fault == "missing_endpoint":
        broken["endpoints"].pop()
    elif fault == "partial":
        broken["endpoints"][0]["completed_update"] = 7999
    elif fault == "duplicate":
        broken["endpoints"][1]["inference_weights"] = broken["endpoints"][0]["inference_weights"]
    elif fault == "technical":
        broken["endpoints"][0]["technical"] = True
    elif fault == "seed":
        broken["endpoints"][0]["seed"] = 23
    elif fault == "family":
        broken["selected_family"] = "C_PPS"
    elif fault == "pretext":
        broken["endpoints"][0]["pretext_completed_updates"] = 3000
    else:
        node = dict(kwargs["lineage"]["C_DIRECT_7"])
        node["pretext_sequences"] = ["hold0"]
        del node["artifact_sha256"]
        kwargs["lineage"]["C_DIRECT_7"] = sign(node)
    del broken["artifact_sha256"]
    broken = sign(broken)
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **k: pytest.fail("sealed targets decoded before authorization"),
    )
    with pytest.raises(ValueError):
        access.evaluate_once(broken, **kwargs, evaluate=lambda *_: {})
    assert not (access.output_root / "CONFIRMATION_CONSUMED.json").exists()


def test_training_and_single_coordinated_read_are_separate(setup, monkeypatch):
    access, lock, kwargs, _ = setup
    reads = []
    original = pd.read_parquet

    def recorded(path, **options):
        reads.append(path.name)
        return original(path, **options)

    monkeypatch.setattr(pd, "read_parquet", recorded)
    assert access.read_training_labels(["ttc"]).ttc.tolist() == [1.25]
    assert reads == ["train_labels.parquet"]
    verify_confirmation_lock(lock, **kwargs)
    result = access.evaluate_once(
        lock, **kwargs, evaluate=lambda metadata, labels: {"rows": len(labels)}
    )
    assert result == {"rows": 6}
    assert reads == [
        "train_labels.parquet",
        "confirmation_metadata_SEALED.parquet",
        "confirmation_labels_SEALED.parquet",
    ]
    with pytest.raises(ValueError, match="already consumed"):
        access.evaluate_once(lock, **kwargs, evaluate=lambda *_: pytest.fail("second evaluation"))
    assert len(reads) == 3


def test_post_read_failure_never_authorizes_another_tuning_attempt(setup):
    access, lock, kwargs, _ = setup

    def fail(*_):
        raise RuntimeError("synthetic evaluator interruption")

    with pytest.raises(RuntimeError):
        access.evaluate_once(lock, **kwargs, evaluate=fail)
    with pytest.raises(ValueError, match="already consumed"):
        access.evaluate_once(lock, **kwargs, evaluate=lambda *_: {})
    assert (access.output_root / "CONFIRMATION_INTERRUPTED.json").exists()


def test_resigned_training_confirmation_path_swap_is_not_accepted(setup):
    access, _, _, partitions = setup
    forged = copy.deepcopy(partitions)
    forged["outputs"]["train_labels"] = forged["outputs"]["confirmation_labels_SEALED"]
    del forged["artifact_sha256"]
    with pytest.raises(ValueError, match="immutable input reference"):
        SealedTrainPartitions(
            sign(forged), access.output_root, expected_manifest_sha256=partitions["artifact_sha256"]
        )


def test_scoped_missing_data_disposition_never_opens_confirmation(tmp_path):
    import json

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
    from e_jepa_ttc.training.stage76_confirmation import publish_blocked_confirmation

    source = tmp_path / "inventory.json"
    atomic_json(source, {"available_masks": 0})
    prerequisite = tmp_path / "prerequisite.json"
    atomic_json(
        prerequisite,
        {
            "status": "PREREQUISITE_BLOCKED",
            "blockers": ["missing foreground"],
            "evidence": [binding(source)],
        },
    )
    recipe = tmp_path / "future.json"
    atomic_json(recipe, {"confirmation_reads": 1, "endpoints_required": 6})
    lock = {
        "campaign_root": str(tmp_path),
        "artifact_sha256": "a" * 64,
        "confirmation_prerequisites": binding(prerequisite),
        "confirmation_future_recipe": binding(recipe),
    }
    output = tmp_path / "S76_DISPOSITION.json"
    result = publish_blocked_confirmation(lock, {"status": "NOT_TRIGGERED"}, output)
    assert result["status"] == "PREREQUISITE_BLOCKED"
    assert not result["confirmation_labels_read"]
    assert not result["scientific_negative"]
    assert result["optimizer_updates"] == 0
    assert publish_blocked_confirmation(lock, {"status": "NOT_TRIGGERED"}, output) == result
    saved = json.loads(output.read_text())
    saved["selected_stable_family"] = "forged"
    atomic_json(output, saved)
    with pytest.raises(ValueError, match="completed S76"):
        publish_blocked_confirmation(lock, {"status": "NOT_TRIGGERED"}, output)
