"""Role contamination and target-free expansion sampling contracts."""

from dataclasses import fields, replace

import pytest

from e_jepa_ttc.data.stage70_roles import (
    SamplingRow,
    assign_train_roles,
    resolve_train_event,
    sample_expansion,
)


def make_roles(extra=12):
    mapping = {"old": "old", "old_alias": "old", "protected": "p", "used": "u"}
    mapping.update({f"fresh{i}": f"fresh{i}" for i in range(extra)})
    return assign_train_roles(
        mapping,
        original_sequences=["old"],
        historically_exposed_sequences=["old", "used"],
        protected_sequences=["protected"],
        role_prefix="fixed-fixture",
    )


def test_six_confirmation_groups_are_locked_before_fitting():
    roles = make_roles()
    assert len(roles.confirmation) == len(roles.expansion) == 6
    assert set(roles.confirmation).isdisjoint(roles.allowed_sequences(include_expansion=True))
    for forbidden in (roles.confirmation[0], "used", "protected"):
        with pytest.raises(ValueError, match="fixed TRAIN role"):
            roles.check_fit_sequences([forbidden], excluded_sequences=[], include_expansion=True)


def test_injected_overlap_in_resolved_role_manifest_is_rejected():
    roles = make_roles()
    with pytest.raises(ValueError, match="roles overlap"):
        replace(roles, expansion=roles.expansion + roles.confirmation[:1])
    with pytest.raises(ValueError, match="documented exposed"):
        replace(roles, expansion=roles.expansion + ("u",))


def test_inner_and_outer_exclusions_cover_acquisition_aliases():
    roles = make_roles()
    with pytest.raises(ValueError, match="acquisition"):
        roles.check_fit_sequences(
            ["old_alias"], excluded_sequences=["old"], include_expansion=False
        )


def test_insufficient_confirmation_does_not_block_expansion_or_subset():
    roles = make_roles(extra=11)
    assert not roles.confirmation
    assert len(roles.expansion) == 11
    roles.check_fit_sequences(["old"], excluded_sequences=[], include_expansion=False)


def test_roles_are_inventory_order_invariant():
    first = make_roles()
    mapping = dict(reversed(first.sequence_to_group))
    second = assign_train_roles(
        mapping,
        original_sequences=["old"],
        historically_exposed_sequences=["used", "old"],
        protected_sequences=["protected"],
        role_prefix="fixed-fixture",
    )
    assert first == second


def test_sampling_has_no_target_slot_and_is_order_invariant():
    assert {field.name for field in fields(SamplingRow)} == {
        "token",
        "sequence",
        "track",
        "anchor_us",
    }
    rows = [
        SamplingRow(f"{s}-{t}-{i}", s, t, i * 10_000)
        for s in ("a", "b")
        for t in ("x", "y")
        for i in range(10)
    ]
    kwargs = {"sequence_to_group": {"a": "ga", "b": "gb"}, "limit": 12, "prefix": "fixture"}
    selected, rejected = sample_expansion(rows, **kwargs)
    assert (selected, rejected) == sample_expansion(list(reversed(rows)), **kwargs)
    assert len(selected) == 12 and len(rejected) == len(rows) - 12
    chosen = [row for row in rows if row.token in selected]
    for first in chosen:
        for second in chosen:
            if first.token != second.token and (first.sequence, first.track) == (
                second.sequence,
                second.track,
            ):
                assert abs(first.anchor_us - second.anchor_us) >= 20_000
    assert [token[0] for token in selected[:4]].count("a") == 2


@pytest.mark.parametrize(
    "path",
    [
        "../a/events.h5",
        "data/test/a/events.h5",
        "C:/a/events.h5",
        "data/train/b/events.h5",
        "a/events.h5:stream",
    ],
)
def test_malicious_or_cross_role_paths_fail_before_file_read(tmp_path, path):
    with pytest.raises(ValueError, match="metadata path"):
        resolve_train_event(tmp_path, "a", path, frozenset({"a"}))


def test_harmless_source_name_does_not_trigger_false_exposure(tmp_path):
    root = tmp_path / "garl_ttc_train"
    (root / "a").mkdir(parents=True)
    file = root / "a" / "events.h5"
    file.write_bytes(b"synthetic fixture")
    assert resolve_train_event(root, "a", "data/train/a/events.h5", frozenset({"a"})) == file
