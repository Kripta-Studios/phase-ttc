# Local inference comparison — 12 November 2026

RTX 5070 Ti Laptop, Python 3.11.15, PyTorch 2.11.0+cu128, FP32, TF32 off, two PyTorch CPU threads. Frozen weights; no labels read and no new accuracy score.

## Parameter inventory and EvTTC inference

All 434 queries, five arms, three passes: 6,510 attempts. Timings below use all-five complete cases and include native API transfer, synchronization and host output. Sensor preparation is excluded from this inference clock.

| System | Stored parameters | Observed executed parameters | Median inference (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
| PHASE-TTC | 2,347,313 | 2,347,313 | 530.26 | 651.37 |
| Garl full release | 243,109,898 | 243,109,898 | 36.94 | 219.92 |
| Garl decoder bypass | 243,109,898 | 49,240,194 | 10.40 | 33.15 |
| Garl event-only | 24,674,178 | 24,674,178 | 5.42 | 19.15 |
| Garl RGB-only | 24,567,554 | 24,567,554 | 5.24 | 17.91 |

Executed counts come from one supported real profile query. The Garl bypass retains all checkpoint weights; this is not an exported pruned-model comparison.

## Native eAP RAW replay

120 physical queries across twelve sequences, three arms, three passes: 1,080 attempts. The clock includes real sensor reading and preparation, native inference and host output. PHASE reuses causal observations; each arm has separate per-sequence reader caches.

| System | Median total (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| PHASE native RAW | 342.45 | 615.19 | 756.00 |
| Garl full RAW | 224.02 | 368.19 | 578.85 |
| Garl bypass RAW | 202.31 | 323.26 | 364.97 |

The eAP population is a fixed chronological prefix, not a random sample of the full benchmark. EvTTC shares a sensor reader and cyclic order does not balance all ordered pairs. OS file caches and desktop scheduling are uncontrolled. All latency outliers remain in the ledgers. These paths differ from the previous paced 50 Hz service.

## English architecture paragraph

PHASE-TTC brings motion estimation, causal temporal refinement and phase-based fusion together in a complete system with 2,347,313 parameters. Its compact 1,955-parameter fusion head combines motion estimates from events and RGB. Garl-TTC uses two ResNet-50 backbones with late feature fusion; its decoder-bypassed path involved 49,240,194 parameters in our local execution profile. On the same RTX 5070 Ti Laptop GPU, median native RAW-to-host time across our 120-query eAP cohort was 342.45 ms for PHASE-TTC, 224.02 ms for the full Garl release and 202.31 ms with Garl's auxiliary decoders disabled. The measured parameter footprint and runtime are separate results.

[Protocols, per-call evidence and reproduction commands](../data/local_hardware_20261112/README.md). The paper includes preparation costs, memory, supported-operator arithmetic, coverage, pairwise populations, recording-level distributions and repeat-level means.
