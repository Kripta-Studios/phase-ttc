# Transfer cohort registration

Pass `--transfer-manifest` to `python -m operational.range_transfer.run` after
the V-only selection and all six control/candidate endpoints are frozen. Start
from `TRANSFER_MANIFEST.example.json`; every path must be absolute and every
hash is the SHA-256 of the named file.

The cache is an existing native Event PHASE17 `.npz`, never a TRAIN40 cache. It
contains exactly `features` FP32 `[N,L,17]`, `timing` FP32 `[N,L,4]`, `valid`
bool `[N,L]`, `expert_phase` FP32 `[N,3]`, unique `sample_token`, and
`group_id`. Add `target_phase` FP32 `[N]` when `labels_available` is true. A
label-free cache is accepted only when `benchmark` is `Test12`; it produces
predictions without local scores or submission.

The cohort manifest JSON contains:

```json
{
  "status": "COMPLETE",
  "cohort_id": "same-id-as-registration",
  "group_ids": ["complete", "set", "of", "groups"],
  "population_size": 123,
  "sample_token_sha256": "sha256-of-canonical-JSON-token-list-in-cache-order",
  "sample_token_group_sha256": "sha256-of-ordered-token/group-object-list"
}
```

The cache receipt JSON contains these bindings in addition to any producer
diagnostics:

```json
{
  "status": "COMPLETE",
  "modality": "event",
  "source_kind": "native_matched_event_history",
  "cohort_id": "same-id-as-registration",
  "output_sha256": "cache-sha256",
  "normalizer_sha256": "admitted-H-normalizer-sha256",
  "parent_sha256": "admitted-event-parent-freeze-sha256",
  "schema_sha256": "EVENT_PHASE17_SHA256 from the installed code",
  "cohort_manifest_sha256": "cohort-manifest-sha256",
  "precision": "float32"
}
```

The registered normalizer and parent freeze must be the same files admitted by
V14, and transfer groups must be disjoint from P/H/V. The cohort declares
`selection_uses_cohort: false`; transfer cannot select or rescue a recipe.
