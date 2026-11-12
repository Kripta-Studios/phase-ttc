# Extended persistent inference implementation plan

> **For agentic workers:** Use superpowers:executing-plans; obtain a fresh final code review.

**Goal:** Expand the paper with reproducible, repeated, multi-sequence persistent PHASE/Garl inference measurements on this host.

**Architecture:** Preserve the sealed model implementations and causal feeder. Freeze a new campaign, run timed services first, then independent native oracles; retain every attempt. Generate tables and the chapter from archived ledgers.

**Tech stack:** Existing Python/PyTorch CUDA services, JSON evidence, NumPy and LaTeX.

**Spec:** The user's request for more repetitions, different samples, longer sequences and a complete inference comparison chapter; design decisions below.

## Design and constraints

- Scout all 32 registered EvTTC sequences using sensor clocks and causal annotation availability only, with a fixed 20,000 us grid starting at 700,000 us (or first event +620,000 us if later). No TTC targets or latency selection.
- Select six declared scenarios: CPLA-low, CCRm-medium-0-overlap-0, CCRs-3-low-100-overlap-100, CCRs-side-high, CPNA-high, CPNAO-high. Each uses its longest contiguous accepted interval; earliest breaks ties. Preserve the scout and all exclusions.
- Four fresh-process repetitions per model/scenario at 50 Hz, queue capacity 2; alternating paired order, rotate scenario order per repetition. Full interval, no truncation by performance.
- Two additional repetitions per model/scenario with queue capacity 64: buffered-capacity diagnostic, same 50 Hz arrivals, backlog included in response. These are not interchangeable with bounded-queue primary results.
- Equal preloaded RAW, FP32/TF32 off, one annotated object, native preprocessing, exact RGB reuse and CUDA graphs. No model/training changes. Preserve strict history-gap refusal policy.
- Fixed 15-second quiet interval immediately before each measured stream after initialization, with host/GPU receipts at its boundaries; no oracle or other agent GPU jobs during timing. This is a fixed rest policy, not thermal stabilization or exclusive hardware proof.
- Run independent native references only after all timing. Compute each scenario/model reference once from disk and compare every emitted result in every repetition, preserving explicit missing-output coverage. Never promote incomplete conditional quantiles to full-service p95.
- Six scenes contain real continuous intervals up to 9.3 s; no looped-stream or 60-second stability claim. The feeder declares a conservative 12 GiB memory envelope. A larger queue diagnoses throughput/backlog and is not a deployment recommendation.
- Preserve original campaign and all negative outcomes; report the CPU-arrival replay boundary and annotation assumption. No live-camera, detector-cost, eAP streaming or broad SOTA claim.
- Authorized integration: update paper/PDF/source package and push main, as previously requested. No external social/model publication.

## Review focus

1. Gaps cannot be closed when constructing intervals; test duplicate/gap selection.
2. Deferred oracles must compare the actual saved outputs and independent inputs; test missing/mismatched outputs.
3. Queue capacities/rates cannot be pooled; validate from raw timestamps and protocol.
4. Failed attempts remain in the population; validate expected run names and ledgers.
5. Freeze identities before timing and audit afterward; fail publication on altered evidence.

## Tasks

- [x] Implement interval selection, new launch wrappers, deferred oracle and campaign freeze; test selection and comparison failures before full execution.
- [x] Run short setup pilots, retaining them separately; freeze sources and execute all 72 declared timed passes.
- [x] Run independent disk/native references, audit every delivered output, archive exact evidence and failures.
- [x] Generate per-scenario/repeat latency, deadlines, throughput/backlog, coverage, startup/memory and cache analysis; extend the English paper with methodology, results and limitations.
- [x] Run relevant tests, independent review, regenerate/visually inspect PDF and verify source package.
- Integration is authorized on main; remote commit/push and CI status are reported in the delivery conversation.

## Execution ledger

- Initial HEAD cfbef60d; worktree clean. Existing user authorization resolves routine design choices without another approval round.
- Scout all 32 sequences found maximum accepted contiguous interval 465 updates (9.3 s). Missing/ambiguous annotation slots are preserved in the scout and not silently bridged.

- Ruling before timing: use bounded queue 64 for the buffered diagnostic, not whole-interval capacity, because the feeder conservatively reserves 128 MiB per payload on this 32 GiB host. Drops remain possible and must be reported.

- Pilots retained: pilot01 path protection rejected PHASE output placement (Garl completed); pilot02 four complete 20-update runs, native predictions passed, original tuple/list container false failures retained with a separate canonical-JSON correction receipt; pilot03 rejected scout comparison before timing (same tuple/list serialization issue); pilot04 four complete runs under final rest policy, PHASE native check passed. Garl native reference reached all anchors but its final integrity check exposed a local variable shadow; fixed before main01 freeze. Pilot02 independent Garl input hashes/heights/TTC already verified the numerical path.
- Independent reviewer findings resolved before main01: rest after feeder warmup, all-32 reconstruction, candidate inventory, frozen native launcher and pre/post integrity. PHASE input checks are exact metadata plus numerical parity, not prepared-tensor byte hashes.
- Main01 started with 72 fixed commands; frozen runtime will not be edited during measurement. Publication independently validates ordered mode/rate/queue/repeat population and reloads verified frozen comparator sources for portable recomputation.
- PDF skill's container marker script is unavailable on this native Windows host; existing LaTeX/render/contract workflow will be used and visually checked.

- Retrospective analysis decision after the first six-scene round: add an explicitly labeled first-response versus fixed latter-half distribution diagnostic for buffered runs, admitted only if both full runs complete with native parity. Motivated by observed early long calls followed by fast service and history-gap refusal cascades. This is not a change to prospective primary endpoints, clock, workloads, queue or repetitions.

- Timed campaign completed: 72/72 attempts; end audit passed for 2,844 files. Four queue2 repetitions/model/scene retained; queue64 PHASE12/12 complete, Garl7/12 complete before native verification. Core/maintained Pyright: 0 errors, Ruff: 3,639 files clean, targeted suite: 141 tests passed.
- Oracle-job controller was added during timing as a waiting/scheduling helper; its native reference implementation and model inputs were already frozen. Paper explicitly distinguishes this post-timing orchestration identity from the prospective timing/native-arithmetic freeze. Preserve executed controller bytes, then enforce prospective controller binding for future campaigns after the current controller finishes.

- Native S2 PHASE audit found three RGB intermediate tolerance failures in each buffered repetition; fused and event outputs at those anchors pass. Original rtol/atol 1e-5 and all frozen timing evidence unchanged. Post-reference reporting now explicitly separates complete delivery from full numerical admission, retains measured conditional timings with failed-parity counts, and suppresses admitted grouped p95 for failed native checks. Component statistics distinguish repeated failures from distinct scene/anchors. This is a retrospective accounting clarification, not retuning or retiming.

- Final native audit: all 12 jobs executed successfully and all 3,958 native model/anchor references are valid. Garl delivered checks all pass. PHASE has six repeated R_CTX failures at three S2 anchors, maximum absolute difference 0.001220703125 s; all delivered fused and event TTC checks pass. No tolerance was changed.
- Full evidence archive and enclosing publication inventory sealed. Q2 complete delivery/admission: PHASE 1/24, Garl 0/24. Q64 complete delivery: PHASE 12/12, Garl 7/12; strict numerical admission: PHASE 10/12, Garl 7/12. All 23,748 offered slots and 8,608 delivered outputs retained.
- After archival, maintained launchers received typing-only casts/device typing corrections; the waiting controller now requires prospective source binding for future campaigns. The exact historical executed controller and runtime snapshots remain unchanged in the archive and define this campaign.
- Final validation: 150 targeted tests passed; Ruff check/format clean for 3,639 files; core, maintained and new-launcher Pyright profiles report zero errors. The first bare Pyright invocation used the broken default .venv; the explicit .venv-transfer profile separated environment errors from the corrected launcher annotations.
- Independent final review found no remaining code/data blocker. Final PDF: 97 pages, all fonts embedded, zero overfull boxes, missing glyphs, unresolved references/citations or LaTeX warnings. Every page reviewed on contact sheets; main/new tables and final scope inspected enlarged, including the corrected Figure 29 caption.
- PDF SHA-256: 744fb89cd80cfdce0f0c01cfcf518f515126484cc1bf9e4654f5b088d866a732. Source ZIP SHA-256: e414b50cce13689e5d86be1fef16ae8041556aa7493edbe7c3216b8594a4f6ab. Extracted ZIP passes CRC, all 2,024 payload hashes, publication binding, and independent recomputation of all 72 run records without repository/model imports.
