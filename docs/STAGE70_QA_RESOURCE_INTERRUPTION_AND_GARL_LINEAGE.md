# QA resource interruption and recovered Garl lineage

The isolated source-pin suite was interrupted by the fixed 120 GB projected disk
minimum after completing 394 of 1,803 tests. The 1,409 pending IDs are explicit in
`baseline_qa/baseline_source57b_full/TEST_DISPOSITION.json`. This is partial baseline
evidence, not a completed baseline/current comparison. Preserve all observed failure
IDs and the interrupted test when continuing; do not label unexecuted tests passing.

The suite consumed 218 confirmed CPU optimizer updates across 82 optimizer instances,
including its inherited synthetic training subprocesses. All recorded processes
closed, no attempt is uncertain, and the unused reservation was released. Including
the separately verified one-update instrumentation acceptance, the campaign now has
479 confirmed technical updates, 100 observed optimizer instances and 17 prescribed
technical model endpoints. Scientific optimizer updates and scientific ridge fits
remain zero. Test optimizer instances are not scientific candidates or fixed endpoints.

The cache writer continues under its separate resource guard. Its outstanding
uncompressed write reservation plus the suite's minimum 3 GiB reservation caused
the joint disk block. Finish or resource-interrupt the cache, then reassess before
reserving further QA work. No percentage margin was reintroduced and nothing was
deleted. There is no scientific lock or outer evaluation authorization yet.

`preflight/GARL_CHECKPOINT_LINEAGE.json` verifies all three recovered historical
Garl checkpoints: exact physical hashes, finite tensors, strict local event-only
40-channel ResNet50 adapter loading, selected epoch and cache identity, original
TRAIN/dev group and token hashes, and per-fold predictions matching the recovered
8,192-row OOF table. Target values are float32-exact; float64 CSV parsing differences
are recorded explicitly. This consumed no optimizer update and no new inference pass.

The historical comparator's MiD remains 144.35302716561466. It selected checkpoints
on outer development folds and lacks foreground decoder supervision. Therefore this
receipt does not qualify it for isolated S76 confirmation. Full acquisition ancestry
and a fresh tensor inference replay remain separate checks; protected validation and
confirmation targets were not opened by this audit.

The audit's initial attempts exposed a missing binding-size argument, Windows text
encoding and float64 CSV roundtrip precision. Those technical read/audit failures
preceded the final receipt and changed no model or scientific recipe. Read intents
from those attempts remain in the audit journal.
