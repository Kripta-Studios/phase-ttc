# New router and confirmation controls before scientific fitting

The new producer manifest explicitly records fit, selection, normalization, prior
and pretext sequences, parent identities, physical source bindings, role hash,
code commit and neural optimization seed. The transitive verifier reuses the
existing ancestor traversal and additionally resolves held-out acquisition aliases,
checks each source against a resolved physical allowlist before hashing it, and
rejects missing exposure fields, changed bytes, missing parents and cycles.
Source filenames are not interpreted as evidence of target exposure.

The router permutation moves new TTC actions, benchmark phases and all six
diagnostics together across different tracks within the same sequence, using the
existing deterministic derangement. Old PHASE17 and old expert actions stay fixed.
New pair differences and supervised costs are recomputed against the original
row targets. A phase inconsistent with the emitted TTC fails validation. The
registered30/45-column schema is in
`configs/protocol/stage75_full_cost_features.json`. Actual nested producer creation,
lineage-bound router fitting and historical RISK17 replay still require integration.

S76 eligibility uses every seed7/13/23 loss, full coverage, the exact sign/crucial
guardrails and improvement of each new seed. It adds no CI prerequisite. Family
selection uses the lower mean development loss, then latency for differences
within0.1, then DIRECT. Missing stable replication, freshness or a credible
TRAIN-only Garl comparator yields a named branch block.

The application-level confirmation reader requires a content-bound administrative
partition manifest, signed/allowlisted freshness and source-audited Garl receipts,
development-only replication selection, the prerequisite-resolved treatment, and
six distinct complete final endpoints with seeds7/13/23 for both families. Every
endpoint's weight bytes, producer family/seed and transitive confirmation exclusions
are checked; prescribed pretext must have a same-seed ancestor. A partial checkpoint,
technical model, missing seed or substituted weight cannot authorize the read.

The reader's training method only addresses the fixed TRAIN-label partition.
Confirmation access uses an exclusively created durable consumption record before
the coordinated metadata/label decode. It validates row/group alignment before
calling the evaluator. A failure after consumption leaves an interruption receipt
and cannot authorize another tuning/read attempt. This is an application contract,
not an OS security sandbox; the earlier administrative monolithic parquet decode
remains accurately recorded.

Validation:37 synthetic tests cover transitive outer/inner/pretext/prior/selection
leaks, acquisition aliases, unallowlisted byte access, cycles, missing fields,
stable replication/tie selection, incomplete/reused endpoints, premature decoding,
partition substitution, successful one-time decoding and terminal interruption.
Five synthetic router tests cover consistent donor movement, fixed row targets,
schema/phase rejection and infeasible cross-track permutation. They open only
temporary fixtures and perform no optimizer updates or scientific ridge fits.

No real confirmation read has occurred. Freshness of the six actual groups and
the final Garl comparison remain unresolved. These controls must be wired into the
source-bound campaign executor and final training callsites before scientific lock;
passing their synthetic tests is not completion of S75 or S76.
