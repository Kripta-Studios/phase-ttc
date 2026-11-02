# ADR-0045: Preserve detection source identities during CI formatting

Date: 2026-11-02

Status: accepted

## Context

CI at `5a563243` fails Ruff's format check on 62 files. Eight of these files
still match source SHA-256 digests in published detection receipts. Formatting
them would break those identities even though their Python behavior is unchanged.

## Decision

Apply Ruff formatting to the 54 ordinary source and test files. Exclude exactly
eight receipt-bound paths from formatting, retaining all lint checks. The paths,
original digests and supporting receipt locations are enumerated in
[`test_detection_receipt_source_identity.py`](../../tests/regression/test_detection_receipt_source_identity.py).
The regression test checks each current file's bytes against its published digest
and verifies that the cited receipt contains that digest.

The exceptions cover the MonoCoP smoke audit, RGB media preflight and revision,
event representation probe, evaluation memory profile, campaign queue verifier,
event fusion and temporal fusion. They do not extend to their directories,
successors, or source versions that no longer match an executed receipt.

## Consequences

Historical receipts and packaged sources remain unchanged. A functional change to
one of these eight files requires a successor identity; it must not silently
rewrite the recorded experiment. The remaining code continues through the normal
formatter, lint, type, test and build checks. This decision makes no new claim
about model quality or experiment execution.
