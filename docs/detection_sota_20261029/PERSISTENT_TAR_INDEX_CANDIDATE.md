# Persistent TAR index candidate

Status: isolated CPU candidate only. It is not integrated into any campaign or admitted runtime.

The current strict reader recomputes each selected TAR's complete SHA-256 and enumerates every TAR
header in every process. The candidate persists the already validated offsets in canonical,
write-once JSON. Reuse requires the preregistered sidecar SHA, exact root and resolved paths, exact
filesystem identities, exact public TAR SHA map, and exact token/member/SHA ownership. It still
recomputes every full TAR SHA per process and every selected PNG SHA per read. Consequently, it
removes repeated header enumeration and offset discovery only; it does not remove the dominant
full-archive read.

The real 78-shard v2 receipt records 126.354 seconds for combined full hashing and index
construction. A separate 26-shard offset-index observation took 9.5599 seconds. Linear scaling gives
an unmeasured upper estimate of about 28.68 seconds of avoidable header/index work for 78 shards.
This estimate is not a benchmark result for the candidate.

## Narrowed consumed-input candidate

A separate isolated candidate relies on a sidecar created from a prior fully verified archive and
reads selected payloads directly by offset. It avoids an eager population-wide reread: it verifies the
admitted SHA synchronously on every `read()` before bytes can be decoded or transferred to GPU. That
preserves cryptographic identity of every consumed model RGB input. It does not attest the current
full archive, headers, duplicate or unrequested members, or unselected bytes. Its contract evidence
states:

- `whole_archive_reverified_this_process=false`;
- `all_selected_payloads_preflight_verified=false`;
- `selected_payloads_verified_per_read=true`;
- `unselected_archive_current_identity=NOT_CLAIMED`.

Each repeated verified read receives a contiguous cursor entry in a rolling-SHA ledger. Resume
accepts only the complete prior entries, exact next cursor and same sidecar, ownership, historical
shard map and physically rehashed dependency files. A failed payload check does not advance the
ledger and has no fallback. The current cursor means `verified_read`, not decoded sample, committed
optimizer update or completed evaluation item: `read()` advances before those downstream actions.
The class is sequential and has no lock or two-phase verified/committed lifecycle. Integration must
either enforce synchronous read-to-commit with checkpoint truncation and sampler/RNG binding, or add
an independently reviewed two-phase, thread-safe contract. A completed run may claim only observed
verified reads unless its runner separately proves downstream commitment and full traversal.

This is a new verified-input candidate, not the existing global archive admission. Fresh review
ships it only for isolated sequential CPU use. Runtime integration remains unadmitted and requires
the lifecycle fix above, CPU parity and an explicit runtime gate.
