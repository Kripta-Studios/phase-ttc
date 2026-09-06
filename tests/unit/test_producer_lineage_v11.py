"""Inject outer/inner leaks into every transitive producer exposure category."""

from dataclasses import replace

import pytest

from e_jepa_ttc.artifacts.producer_lineage_v11 import (
    EXPOSURE_FIELDS,
    ProducerRecord,
    verify_lineage,
)
from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, object_digest
from e_jepa_ttc.data.stage70_roles import TrainRoles


@pytest.fixture
def lineage(tmp_path):
    # A filename is not a protected data read: this is harmless source text.
    source = tmp_path / "public_validation_garl_ttc_code.py"
    source.write_text("# synthetic code fixture, no target data")
    record = binding(source)
    roles = TrainRoles(
        (
            ("fit", "fit"),
            ("outer", "outer"),
            ("alias", "outer"),
            ("inner", "inner"),
            ("extra", "extra"),
            ("confirm", "confirm"),
        ),
        ("fit", "outer", "inner"),
        ("fit", "outer", "inner"),
        (),
        ("confirm",),
        ("extra",),
        "fixture",
    )
    pretext = ProducerRecord(
        "encoder",
        "pretext",
        "a" * 64,
        "b" * 40,
        True,
        ("fit", "extra"),
        (),
        ("fit", "extra"),
        (),
        ("fit", "extra"),
        (),
        (record,),
        seed=7,
    )
    model = ProducerRecord(
        "model",
        "architecture",
        "a" * 64,
        "b" * 40,
        True,
        ("fit", "extra"),
        (),
        ("fit", "extra"),
        ("fit", "extra"),
        (),
        ("encoder",),
        (record,),
        seed=7,
    )
    return roles, pretext, model, record


def check(lineage, pretext=None, model=None, allowed=True):
    roles, original_pretext, original_model, source = lineage
    pretext, model = pretext or original_pretext, model or original_model
    return verify_lineage(
        {"encoder": pretext.signed_record(), "model": model.signed_record()},
        "model",
        roles,
        role_sha256="a" * 64,
        excluded_sequences=("outer", "inner"),
        allowed_sources={source["path"]: source} if allowed else {},
    )


def test_valid_transitive_sources_and_harmless_code_name_pass(lineage):
    assert check(lineage) == ("encoder", "model")


@pytest.mark.parametrize("field", EXPOSURE_FIELDS)
@pytest.mark.parametrize("held", ["outer", "inner", "alias", "confirm"])
def test_every_ancestor_exposure_rejects_held_groups_and_acquisition_aliases(lineage, field, held):
    pretext = replace(lineage[1], **{field: ("fit", held)})
    with pytest.raises(ValueError):
        check(lineage, pretext=pretext)


def test_cycle_and_missing_parent_fail_closed(lineage):
    with pytest.raises(ValueError, match="cycle"):
        check(lineage, pretext=replace(lineage[1], parents=("model",)))
    with pytest.raises(ValueError, match="missing ancestor"):
        check(lineage, model=replace(lineage[2], parents=("absent",)))


def test_unallowlisted_or_changed_source_bytes_fail(lineage, monkeypatch):
    import e_jepa_ttc.artifacts.producer_lineage_v11 as module

    with monkeypatch.context() as patch:
        patch.setattr(module, "verify", lambda _: pytest.fail("unallowlisted bytes were opened"))
        with pytest.raises(ValueError, match="physical allowlist"):
            check(lineage, allowed=False)
    from pathlib import Path

    Path(lineage[3]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="physical identity changed"):
        check(lineage)


def test_missing_pretext_field_is_not_assumed_empty(lineage):
    roles, pretext, model, source = lineage
    malformed = pretext.signed_record()
    del malformed["pretext_sequences"]
    del malformed["artifact_sha256"]
    malformed["artifact_sha256"] = object_digest(malformed)
    with pytest.raises(ValueError, match="missing explicit"):
        verify_lineage(
            {"encoder": malformed, "model": model.signed_record()},
            "model",
            roles,
            role_sha256="a" * 64,
            excluded_sequences=("outer",),
            allowed_sources={source["path"]: source},
        )
