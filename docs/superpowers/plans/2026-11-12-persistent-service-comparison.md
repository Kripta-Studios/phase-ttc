# Persistent PHASE/Garl production-service comparison

> Execute natively with executing-plans; obtain an independent reviewer before delivery.

**Goal:** Replace the article's adapter/API latency comparison with measured persistent-service responses for PHASE and Garl on the same causal RAW replay, then rebuild and push the corrected publication.

**Spec:** User correction of 12 November 2026: use the persistent optimized PHASE service and the equivalent persistent Garl path exclusively for article latency comparisons.

**Architecture:** Preserve PHASE's existing prefix-event, RGB-feature-cache, reverse-summary and overlapping service. Reuse its preloaded causal RAW feeder and absolute scheduled-arrival clock for a persistent Garl TTC service. Garl reuses exact endpoint representations and paired-backbone features, disables only auxiliary decoders after parity verification, and retains native height-to-TTC arithmetic. No training or accuracy selection.

**Tech stack:** Existing Python 3.11.15/PyTorch CUDA environment, NumPy, pytest, LaTeX.

## Constraints and review focus

- Start from clean main 0eb888df; existing authorization includes correction and push. Work in the current checkout because frozen manifests bind this local workspace.
- Primary matched population: the historical CPLA-low 250-arrival, 50 Hz, one-object, five-second causal manifest, three independent passes per system with alternating order. Do not promise reproduction of a particular latency.
- Both feeders preload RAW only, expose preceding history before arrivals, and release new data only at causal timestamps. No representations or predictions may be precomputed from future data.
- Same FP32/TF32-off/two-CPU-thread environment and queue capacity two; report all scheduled arrivals, failures, drops and input age. Latency ends at synchronized host TTC. Startup and disk preload are separate.
- Garl uses its own native two-endpoint representation; cache keys bind frame identity, geometry, event windows and model. Repeated sensor endpoints are disclosed, not described as fresh measurements.
- Validate every output against a separate uncached reference outside timed spans. Keep native cadence conversion and signed TTC. Do not discard overloads or select the fastest run.
- Preserve earlier ledgers and snapshots as historical evidence. Remove their API timings from current article claims/tables/plot and social copy; preserve valid parameter inventory separately.
- Historical 29.14 ms is a p95 from the earlier experiment, not a guaranteed deadline or a substitute for the new result.

## Tasks

- [x] Trace and reproduce the existing persistent PHASE service; freeze the shared protocol.
- [x] Test generic paced output validation, bounded causal event history and exact Garl cache invalidation; implement persistent Garl service and runner.
- [x] Run matched repeated services sequentially, retain all outcomes, verify uncached parity and input age.
- [x] Generate a persistent-only latency report and article revision; retain old evidence as non-production history.
- [x] Run relevant checks, independent review, rebuild and visually verify PDF/ZIP.

Delivery uses the already authorized direct push to main. The final commit and
post-push CI result are recorded in GitHub and the delivery message.

## Execution ledger

- Ruling: existing user authorization covers direct execution and main push; no redundant design approval. PHASE service must remain the actual historical optimized path, not a relabeled generic RAW predictor.
- Root cause: previous comparison used query-oriented adapters; sharing model weights did not reproduce persistent service execution. The new primary workload uses the exact causal replay behind L40.
- Historical replay configuration and all required candidate/reference/manifests are available locally. Its input is EvTTC CPLA-low, not eAP; publication will identify that population explicitly.
- Correct historical service flags reproduced 20/20 pilot outputs with native parity. An earlier pilot with omitted historical flags overloaded; retained as a configuration failure, not a model comparison.
- Main01 freezes three matched passes. PHASE pass 1 completes 250/250 with p95 31.87548 ms. Eager persistent Garl overloads the capacity-two queue; failures are retained and cannot supply a complete-service p95.
- Prospective revision main02 adds native fixed-shape CUDA graphs to Garl, as PHASE already uses graphs. It changes neither cohort, offered rate nor queue capacity. New scripts/source identities and all-anchor native parity remain mandatory. Main01 is retained as the eager service control; no fastest-pass selection.
- Main01 completed all scheduled runs. PHASE coverage: 250/247/169 of 250; eager Garl: 29/61/16. Every delivered TTC passes the native oracle. Incomplete histories and queue losses remain recorded. A post-run audit verifies 2,811 files against their predeclared digests without changes.
- Graph CUDA miss/hit/replacement test passes; real 20-anchor graph pilot passes exact input/height/TTC checks with 12 cache hits and 8 misses. A preceding pilot launched before its runtime overlay finished and failed before pacing; its setup failure is retained separately.
- Main02 is predeclared before its six runs. PHASE pass 1 completes 250/250 with p95 26.811725 ms and full parity. Graph Garl pass 1 still has a capacity failure; no primary complete-service p95 will be derived from partial outputs. No further runtime tuning during this protocol.
- Independent reviewer identifies uncontrolled carryover: each model's GPU-heavy oracle runs between timed streams, with different workloads and no enforced cooldown. Telemetry does not remove this confound. The article must not attribute failures to thermal state or claim a causal speed ranking. Finish all six runs and retain capacity outcomes; do not chase a favorable protocol.
- Main02 final: PHASE 250/25/250 valid outputs; Garl graph 19/89/6. PHASE complete passes have p95 26.811725 and 28.87512 ms. Neither arm establishes complete coverage in all three passes. All delivered predictions preserve native parity; 2,820 predeclared files pass the terminal integrity check. Both complete protocols and setup/parity pilots are archived in the paper.
- Targeted paced/comparison/report tests and CUDA graph parity tests pass. Ruff passes. Type checks pass using the actual `.venv-transfer` environment; the repository's default `.venv` is the pre-existing broken Torch installation and cannot supply valid Torch type information.
- Independent final review finds no blocker. The article and supplement now exclude query-adapter latency values, while source tables, failed-cost conclusions and historical ledgers remain archived. The final PDF has 83 pages, embedded fonts, no LaTeX warnings or layout/reference errors, and visually inspected revised pages. The source ZIP verifies 988 scientific inputs and three final artifacts.
