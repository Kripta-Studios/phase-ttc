# E128-T8 count cache integration

The new raw adapter uses the original two half-open windows, eight integer-floor
bins per window, negative/positive channel order, the original common ROI and
event-x calibration offset. Raw sensor bounds remain 1280x720. The four contexts
retain the physical midpoint offsets checked by Stage70.

`event_context128` separates integer counts, TRAIN rate moments and neural input
tensors. The neural input record contains contexts, times and sensor support;
it has no target, sequence, track, bucket or individual box-height field. The
registered nonmissing support uses ROI pixel-center coverage of the original
sensor. Its interpretation is padding coverage, not object foreground truth.
Normalization uses all activity values, including zero counts, within that support.
Pixels outside support are neutral zero after normalization.

`event_context128_cache` reuses the original raw-access journal, component hashing,
atomic JSON publication and writer ownership. It holds one HDF5 handle and at most
two 64-MiB event windows, reads larger windows in bounded chunks, and publishes
up to 64 count rows per immutable shard. Counts cannot saturate. Overflow requires
a separate whole uint32 cache version and a new full disk reservation.

Resume verifies completed component hashes, layouts and row positions. A complete
cache also verifies its metadata identity and manifest signature. A deliberately
interrupted synthetic cache resumes without rewriting its completed shard; an
altered metadata file is rejected. An initial test exposed a tuple/list return
serialization mismatch; the canonical manifest now uses a list, with the original
failed test XML retained separately from the passing final receipt.

The first 64-token TRAIN raw probe completed with shape [64,2,8,2,128,128]. It is a
nonselectable construction probe, not a trained endpoint. Its executed count-source
files were preserved under `preflight/probe_cache_source_snapshot`. Version 2 adds
explicit producer-code bindings to the manifest before the full original cache is
built. Counts and observation tensors must remain equivalent across these versions.

The amended writer guard reserves every remaining count shard before work and
requires at least 120,000,000,000 bytes of disk headroom and 4 GiB available host
RAM. Raw and historical source files remain read-only.
