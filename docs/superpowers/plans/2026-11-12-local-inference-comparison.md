# Same-hardware PHASE-TTC / Garl-TTC inference comparison

> Execute natively in this session using executing-plans; obtain an independent final review before the authorized push to main.

**Goal:** Execute real frozen-weight inference on this workstation, publish reproducible parameter/runtime/memory comparisons in the corrected paper, and push the verified delivery to main.

**Architecture:** Reuse the verified native Garl implementation and sealed original PHASE runtime. Preserve separate sensor preparation and prepared-input-to-host clocks, retain every query/failure, and record per-call measurements and model/source hashes. Generate manuscript tables from the result ledger.

**Tech stack:** Existing Python 3.11 / PyTorch CUDA environment, NumPy, native sensor adapters, pytest, Ruff, existing LaTeX publication pipeline.

**Spec:** User request of 12 November 2026; AGENTS.md scientific integrity and reproducibility requirements.

## Constraints and decisions

- No training, weight changes, hyperparameter selection using labels, official leaderboard submission, or paid compute.
- Use the original nine-component PHASE-TTC, full published Garl-TTC, and Garl event/RGB controls. Optional decoder-bypass Garl control must preserve TTC outputs and remain explicitly separate from its unmodified release.
- Main timing cohort: all 434 existing EvTTC query identities across ten recordings; identical queries, native model-specific representations/history. This is a computational comparison, not a newly independent accuracy benchmark.
- Inspect the available eAP/GarlTTC assets and add a deterministic real-input cohort if the native adapters support both models without changing their contracts. Record any limitation explicitly.
- FP32, batch one query, two CPU threads, TF32 disabled, no compilation/autocast, inference mode, CUDA synchronization at measurement boundaries. No concurrent GPU experiments. Three repeat passes; rotate model order and record warmup separately.
- Report resident and actually executed parameter counts separately; the 1,955-parameter PHASE fusion head is never the full model count. Profile arithmetic separately from timing; disclose unsupported operations.
- Publish per-query preparation, prepared inference, total wall time, median/p95/p99, coverage, CUDA allocated/reserved peak, observed process RSS, hardware/software/thermal telemetry, source/weight/input hashes and parity checks.
- No generic speedup claim across unequal measurement boundaries. Raw I/O and preprocessing inclusion must be explicit. Existing paced streaming results retain their distinct contract.
- User authorized main push. Work in the clean current checkout, preserve external models/data and historical evidence. Only new experimental evidence and current manuscript are changed.

## Review focus

1. Garl training decoders may remain active in released inference; distinguish retained, executed and required weights, and verify bypass parity.
2. Startup/missing support must remain in the denominator, with matched-success timing shown separately.
3. Asynchronous CUDA and hidden diagnostic hooks must not create misleading timings; delimit and disclose the runtime API.
4. Warm caches/thermal drift/model residency can bias comparison; rotate execution order and retain repeat-level telemetry.
5. Publication artifact/source binding and historical-source preservation must survive the rebuild and packaging.

## Tasks

- [x] Inventory models, datasets, environment and native inference contracts; freeze protocol and query identity before measured runs.
- [x] Add `scripts/benchmark_local_comparison.py` and reusable helpers in `operational/local_comparison/`; tests cover aggregation, failures, parameter accounting and parity gates before implementation.
- [x] Run a real-input smoke and parity check; execute the fixed cohort and repeats, save all successful/failed attempts, run separate component profiles.
- [x] Generate numerical tables and detailed report from JSON/JSONL, integrate into the expanded paper, update README/progress and provide precise English social copy.
- [ ] Run tests/lint/type checks, rebuild PDF/source package, verify hashes and visually review changed pages; independent review, commit and authorized push to main, then inspect CI.

## Execution ledger

- Initial checkout clean on main. Existing default `.venv` has an unusable torch installation; `.venv-transfer` provides Python 3.11.15 / torch 2.11.0+cu128 / CUDA. Upstream Python 3.8 launcher is broken, so both models will run in the same functional environment, as in prior verified local Garl runs.
- Ruling: proceed without another design/permission round because the user explicitly requested execution, paper integration and main push; reversible implementation and local inference are authorized.
- Native Garl inspection confirms that both auxiliary foreground decoders execute in the released full forward path. The parity-checked flag-bypass retains their weights; executed counts must not be described as a pruned export.
- Smoke01 completed 100 real calls (20 queries across ten recordings, five arms). Main01 was frozen before execution and uses 6,510 calls (434 queries, five arms, three passes). The measured API includes host conversion and native diagnostics. Parameters and memory are profiled separately after the timing passes.
- Added a native eAP cohort: 120 physical chronological-prefix queries from twelve sequences, three arms and three passes. Event inputs must reconstruct exactly from RAW. Preserve released Garl `model.dT=0.1`; sensor-clock spacings of 100001/100002 us are separately recorded. This follows native inference/evaluator source, confirmed in independent review.
- eAP uses separate Garl reader/frame caches per arm. EvTTC's already-frozen shared-reader/cyclic-order protocol remains unchanged; disclose that it does not balance all ordered pairs and cannot isolate cache-dependent preparation speedups.
- Independent review requested by the execution/review workflow identified publication gates: exact cohorts/repeats/model tuples, explicit complete-case vs pairwise populations, profile validation, source/weight binding, input/status/height parity, and equal-scope parameter comparisons. These are implemented in the report generator.
- The paper build now binds JSONL ledgers and new analysis receipts, regenerates tables and distribution plots, and preserves executed sources as `.py.txt` evidence. No changes to the running EvTTC entry point or helper are allowed until execution and source capture finish. Later type-only maintenance will retain the executed snapshots.
- Completed all 6,510 EvTTC and 1,080 eAP attempts. All 120 eAP event tensors match the frozen references byte-for-byte. Repeats and Garl decoder-bypass outputs preserve input/status/height/TTC parity. The independent reviewer found no remaining scientific blocker.
- Final paper: 87 pages, all fonts embedded, no LaTeX warnings, undefined references, missing glyphs or overfull boxes. Reviewed all pages in contact sheets and the changed summary/appendix pages at full resolution. The archive's 822 manifest entries match both ZIP and working-file bytes.
- All new targeted tests pass (35 passed, one historical external-artifact test skipped); global Ruff and formatting pass, maintained and new-code Pyright report zero errors, and the wheel builds. The full local suite is checked separately before delivery.
- Existing ONNX integration subprocesses initially chose the broken default `.venv`; setting `UV_PROJECT_ENVIRONMENT=.venv-transfer` makes both pass without changing model or test code.

- Full local suite finished with five failures in unchanged legacy files: four CUDA/global-state interactions pass in a fresh process; the TRAIN download receipt test still fails on same-size, same-mtime Windows overwrites. Fresh affected-group run: 12 passed, 1 failed. Recorded transparently in the publication review receipt; no full local-suite pass is claimed. Scientific evidence uses separate explicit byte/hash checks.
- Wheel build, both command entry points and PowerShell syntax checks pass. Authorized delivery proceeds with the unrelated local-suite limitation recorded; remote CI will be inspected after push.
