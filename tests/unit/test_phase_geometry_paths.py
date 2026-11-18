"""External input locations of the geometry study come only from environment variables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from operational.phase_geometry import data

VARIABLES = (data.ENV_POD_RETURN, data.ENV_GARLTTC_DATASET, data.ENV_GARLTTC_BENCHMARK)


def test_no_pin_carries_an_absolute_path():
    for spec in (*data.SOURCES.values(), *data.FILES.values()):
        for key in ("archive", "path"):
            if key in spec:
                assert not Path(spec[key]).is_absolute()
                assert ":" not in spec[key]
        assert spec["location"] in ("", *VARIABLES)


@pytest.mark.parametrize("variable", VARIABLES)
def test_unset_variable_names_itself_and_its_content(monkeypatch, variable):
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(data.MissingLocationError, match=variable) as info:
        data.location_root(variable)
    assert data.LOCATIONS[variable] in str(info.value)


def test_unset_variable_blocks_file_and_archive_resolution(monkeypatch, tmp_path):
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(data.MissingLocationError, match=data.ENV_GARLTTC_DATASET):
        data.verified_file("train_data", tmp_path)
    with pytest.raises(data.MissingLocationError, match=data.ENV_POD_RETURN):
        data.read_member("oof_phase20")
    with pytest.raises(data.MissingLocationError, match=data.ENV_POD_RETURN):
        data.resolved_sources()


def test_missing_directory_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(data.ENV_POD_RETURN, str(tmp_path / "absent"))
    with pytest.raises(data.MissingLocationError, match="not an existing directory"):
        data.location_root(data.ENV_POD_RETURN)


def test_locations_resolve_under_configured_roots(monkeypatch, tmp_path):
    for variable in VARIABLES:
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    files = data.resolved_files()
    assert files["train_data"]["path"] == str(
        tmp_path / data.ENV_GARLTTC_DATASET / "data" / "train.parquet"
    )
    checkpoint = "FINAL_CANDIDATE/checkpoints/PHASE_ONLY_EVENT_S7_TRAIN40.pt"
    assert files["released_checkpoint"]["path"] == str(tmp_path / data.ENV_POD_RETURN / checkpoint)
    assert files["blocks"]["path"] == data.FILES["blocks"]["path"]
    sources = data.resolved_sources()
    assert sources["test_inputs"]["archive"] == str(
        tmp_path / data.ENV_POD_RETURN / "final_inputs_63e2720817ba6521.tar"
    )
    assert set(sources["oof_phase20"]) == {"archive", "member", "sha256"}


def test_verified_file_checks_hash_under_configured_root(monkeypatch, tmp_path):
    scorer = tmp_path / "garl_ttc_benchmark/codabench/scoring_program/score.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_bytes(b"print('x')\n")
    monkeypatch.setenv(data.ENV_GARLTTC_BENCHMARK, str(tmp_path))
    digest = hashlib.sha256(scorer.read_bytes()).hexdigest()
    pinned = {**data.FILES["official_scorer"], "sha256": digest}
    monkeypatch.setitem(data.FILES, "official_scorer", pinned)
    assert data.verified_file("official_scorer", Path("unused")) == scorer
