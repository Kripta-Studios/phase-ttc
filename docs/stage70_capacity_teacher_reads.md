# Capacity DINO: incidental TRAIN reads

The v11 producer deserializes complete supervised TRAIN cache records. Those
records contain TTC labels. Its manifest correctly records `ttc_labels_read=true`;
the legacy reader requires this field to be false. The capacity integration
therefore needs an explicit source-reviewed read contract. Historical caches and
readers retain their default no-label-read requirement.

The reviewed producer is commit `d1eda4d4f9a7b6aa53db9704a71b1e931ee00ac3`.
The materializer source SHA256 is
`43f58ac67f5cc39eee5e59dc5d3f6360a37402558e17a148908760e31b699564`.
`GarlTTCLHRCacheDataset.__getitem__` deserializes and converts record fields;
`GarlTTCObjectEventV4Dataset.__getitem__` checks precontext validity and adds a
channel-count field. The materializer never calls the supervised collator. It
uses object identities, common ROI, endpoint boxes and frame intervals to bind
RGB endpoints, then sends only RGB crops to the fixed DINO encoder. Local cosine
relations and validity masks depend on those features. No TTC value selects the
crop, enters the encoder, fits normalization, or changes teacher weights.

This review permits reuse of these fixed RGB features despite the incidental
read. It does not describe the producer as label-unexposed. It does not establish
independence of DINO's original web pretraining, accredited object tracking, or
equivalence to the historical preprocessing of A5/C2F/PAIR. The current ROI is
retrospective. Each new comparator still has to satisfy its own fold exclusions,
and the training wrapper exposes only that fold's TRAIN tokens.

Before accepting the exception, the auditor verifies the exact materializer
source, local teacher weights, manifest hashes, signed preparation, original
roles, TRAIN metadata partition, and selected token/track/sequence universe. A
changed producer needs a new visible review. The complete mmap index must equal
the authorized cohort before any fold subset is returned. The signed contract
is embedded in the audit and compiled training configuration. Missing contracts,
label-use claims, role contamination, changed paths, changed source identities
and changed row identities are rejected. The original manifest is never rewritten.

Use the existing `audit_stage63_frozen_teacher.py` with both
`--capacity-preparation` and `--capacity-preparation-sha256`, in addition to its
existing manifest, canonical metadata, local model, reference repository and
output arguments. The audit records the exact verified token-set hash required
by the comparator compiler. It refuses to overwrite a prior audit. An audit is
only publishable after every expected row and cache payload has been verified;
partial recovery blocks are not a complete teacher.

Consumer changes are developed in an isolated worktree while GPU materialization
continues. Do not replace the live producer's pinned files or its recovery
amendment. Regeneration must use the reviewed producer identity, or a separately
reviewed equivalent source identity, rather than silently using the newest code.

Verification uses synthetic CPU fixtures for historical rejection, explicit
capacity acceptance, malicious lineage, audit publication, exact token hashing,
fold-only training consumption and comparator recipe propagation. No scientific
optimizer updates or outer scores are needed for these checks. Completion of this
integration does not mean the 30,000-row production cache or v11 campaign is ready.
