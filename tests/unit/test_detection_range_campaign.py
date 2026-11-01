"""Aligned population, strict config and complete checkpoint continuation contracts."""

import json
import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.benchmark_evaluation import PROFILE, PROFILE_SHA256, BenchmarkEvaluator
from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler, UpdateLRScheduler
from e_jepa_ttc.detection.range_campaign import (
    Dimensions,
    RangeConfig,
    patch_depth_source,
    raw_targets,
    restore_state,
    save_state,
)


def config():
    return RangeConfig(
        "phase_eap_range_campaign_v1",
        "fixture",
        7,
        "data",
        "release",
        "a" * 64,
        "population.json",
        "b" * 64,
        150.0,
        str(PROFILE["schema"]),
        PROFILE_SHA256,
        "dfull_reset_depth_classifier_keep_physical_embeddings_v1",
        Dimensions(640, 384),
        2,
        4,
        [0, 2, 4],
        2,
        5e-5,
        0.1,
        1e-4,
        120.0,
        2,
        1.0,
    )


def test_config_rejects_protocol_mix_unknown_fields_and_invalid_schedule(tmp_path):
    cfg = config()
    cfg.validate()
    path = tmp_path / "config.json"
    data = asdict(cfg)
    path.write_text(json.dumps(data))
    assert RangeConfig.load(path) == cfg
    for field, value in [
        ("unknown_option", True),
        ("evaluation_profile_sha256", "x" * 64),
        ("evaluation_updates", [0, 4, 2]),
        ("depth_max_m", float("nan")),
        ("batch_size", True),
    ]:
        bad = data | {field: value}
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            RangeConfig.load(path)


def test_raw_population_retains_far_and_behind_annotations():
    annotations = [
        {"category": "car", "bbox_3d_ego": [depth, 0, 1, 4, 1.5, 2, 0]} for depth in (-5, 30, 200)
    ]
    targets = raw_targets(annotations)
    assert len(targets) == 3
    assert targets[0]["size_lwh_m"] == [4, 2, 1.5]
    evaluator = BenchmarkEvaluator()
    evaluator.add_frame("a:0", [], targets)
    assert evaluator.summarize()["gt_boxes"] == 3


def objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = UpdateLRScheduler(
        optimizer, total_updates=4, kind="warmup_cosine", warmup_fraction=0.25, minimum_ratio=0.1
    )
    return model, optimizer, scheduler, ResumablePermutationSampler(5, 7)


def step(model, optimizer, scheduler, sampler, update):
    scheduler.apply(update)
    sampler.next_batch(2)
    optimizer.zero_grad()
    model(torch.randn(3, 2)).square().sum().backward()
    optimizer.step()


def test_resume_preserves_optimizer_schedule_sampler_and_all_rng(tmp_path):
    model, optimizer, scheduler, sampler = objects()
    step(model, optimizer, scheduler, sampler, 1)
    args = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        contract={"profile": PROFILE_SHA256, "depth": 150},
    )
    path = tmp_path / "last.pt"
    save_state(path, **args, update=1, best=0.1, elapsed=2.0)
    expected_random = (random.random(), np.random.rand())
    step(model, optimizer, scheduler, sampler, 2)
    expected_model = {k: v.clone() for k, v in model.state_dict().items()}
    expected_sampler = sampler.state_dict()
    assert restore_state(path, **args) == {"update": 1, "best": 0.1, "training_seconds": 2.0}
    assert (random.random(), np.random.rand()) == expected_random
    step(model, optimizer, scheduler, sampler, 2)
    assert all(torch.equal(v, expected_model[k]) for k, v in model.state_dict().items())
    assert sampler.position == expected_sampler["position"]
    assert torch.equal(sampler.generator.get_state(), expected_sampler["generator_state"])
    with pytest.raises(ValueError, match="contract"):
        restore_state(path, **(args | {"contract": {"depth": 60}}))


def test_depth_patch_refuses_unrecognized_source():
    with pytest.raises(ValueError, match="anchor"):
        patch_depth_source("unrecognized source", "unrecognized model")
