# Persistent PHASE/Garl service comparison

Same RTX 5070 Ti Laptop, FP32/TF32 off, two PyTorch CPU threads. CPLA-low, one object, 250 causal arrivals at 50 Hz over five seconds; three independent replays per model, alternating order. RAW is preloaded; models receive only causally released sensor data. Initialization/disk preload are excluded.

| Service | Median of three p95s (ms) | Range of p95s (ms) | Complete outputs |
|---|---:|---:|---:|
| PHASE-TTC | Not established | — | 525/750 |
| Garl TTC-only | Not established | — | 114/750 |

Every graph-service replay is retained:

| Service/pass | Outputs | Complete-replay p95 (ms) | Drops | Failed responses |
|---|---:|---:|---:|---:|
| phase_1 | 250/250 | 26.81 | 0 | 0 |
| garl_1 | 19/250 | Not established | 1 | 230 |
| garl_2 | 89/250 | Not established | 2 | 159 |
| phase_2 | 25/250 | Not established | 1 | 224 |
| phase_3 | 250/250 | 28.88 | 0 | 0 |
| garl_3 | 6/250 | Not established | 1 | 243 |

The earlier eager-service control is retained in full:

| Service/pass | Outputs | Queue drops | Failed responses |
|---|---:|---:|---:|
| phase_1 | 250/250 | 0 | 0 |
| garl_1 | 29/250 | 2 | 219 |
| garl_2 | 61/250 | 2 | 187 |
| phase_2 | 247/250 | 1 | 2 |
| phase_3 | 169/250 | 1 | 80 |
| garl_3 | 16/250 | 1 | 233 |

Complete runs must pass every native parity check. Incomplete runs retain their drops and failed outputs; no complete-service p95 is assigned to a model unless all three runs complete. The primary clock includes acquisition/release, scheduling, queueing, preparation, inference and host delivery. These are repeats of the same 250 anchors, not 750 independent examples. Neither this experiment nor the historical 29.14 ms p95 proves a strict 20 ms deadline, physical-camera service, multi-object capacity or new accuracy ranking.

Reference verification executes between measured streams, with different GPU workloads for each model and no enforced cooldown. Before/after telemetry documents but does not control power, thermal or cache carryover. These observations do not establish a causal speed ranking, or identify the cause of a capacity failure.

Garl uses the two declared causal RGB endpoints from the shared manifest, exact cached native representations and paired RGB backbone features. Its event backbone and TTC heads run on every completed update, using separate native FP32 CUDA graphs. Auxiliary foreground decoders are disabled, with every delivered output checked against untouched native inference. Missing outputs remain failed checks. Input age and repeated sensor endpoints remain in the evidence.

## English architecture paragraph

PHASE-TTC builds its predictions from motion, causal temporal refinement and phase-based event–RGB fusion. It brings these stages together in a complete 2,347,313-parameter system, compared with 49,240,194 parameters in Garl-TTC's decoder-free TTC path (95.23% fewer). While Garl uses two ResNet-50 backbones and a height-ratio readout, PHASE uses compact motion producers, temporal refiners and an RGB-anchored residual fusion head. Its persistent service updates event state incrementally and reuses exact RGB features and transport summaries, preserving native predictions while avoiding repeated computation.

The earlier query-oriented API experiment is historical evidence only and is excluded from the article's production-service latency comparison.
