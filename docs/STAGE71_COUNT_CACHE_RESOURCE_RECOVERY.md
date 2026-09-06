# Lossless cache recovery under the fixed120GB disk rule

The expansion cache process exited1 at8,704/27,307 rows because its full remaining
write reservation would leave less than120,000,000,000 free bytes. The136
published shards were preserved. The producer already decreases its outstanding
reservation after each publication; this was a real resource threshold event,
not a duplicate-reservation bug or a scientific negative.

No dataset or artifact was deleted. With all affected writers closed, native
Windows NTFS compression was applied to the original128 count payloads and the
136 published expansion payloads. The expansion directory was marked compressed
so future files inherit the same lossless storage setting. Logical arrays,
dimensions, integer counts, ROI, time alignment and producer code did not change.

Native `compact` reported:

| Cache | Logical count bytes | Physical bytes after compression |
|---|---:|---:|
| Original8192 | 8,589,950,976 | 1,734,459,392 |
| Published expansion8704 | 9,126,822,912 | 1,911,812,096 |

The original production reader reverified every preexisting component hash after
compression. Its8,192 rows,128 shards and manifest identity remain unchanged.
The expansion producer's exact resume reverified all136 prior shards and published
new shards beginning at row8,704. The full remaining reservation still assumes
uncompressed future size, conservatively retaining the fixed120GB minimum.

Before restart,148,874,698,752 bytes were free and19,587,051,520 bytes remained
reserved, leaving129,287,647,232 projected free bytes. Other host allocations may
change live headroom; continuous monitoring remains enabled. Cache-backed training
cost measurements must reflect actual decompression work on this host.

Receipts are `preflight/CACHE_COMPRESSION_ORIGINAL_8192.json` and
`preflight/EXPANSION_RESOURCE_RECOVERY_8704.json`. The initial process was session
27807; its resumed process is88719. This execution setting creates no optimizer
updates and no scientific recipe amendment. Expansion availability remains pending
until the full cache and resolved training views finish validation.
