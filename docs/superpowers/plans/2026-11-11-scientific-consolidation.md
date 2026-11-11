# PHASE-TTC scientific consolidation implementation plan

> **For agentic workers:** Execute the authorized revision inline, with a fresh final review. Preserve historical scientific artifacts.

**Goal:** Replace unresolved component-attribution claims with measured matched comparisons, recover omitted controls, and focus the manuscript on the complete TTC system.

**Architecture:** New analyses read sealed predictions and recorded features without changing released models. The main paper presents the system, official result and direct diagnostic comparisons; the supplement retains historical investigations. Independent validation is an explicit requirement, never inferred from new splits of previously inspected recordings.

**Tech stack:** Existing NumPy/PyTorch evaluation code, pytest, LaTeX and the publication verifier.

**Spec:** The user's eight-part critical assessment and explicit instruction to implement the revision. The user reports no known unused recordings.

## Global constraints

- Preserve frozen weights, historical evidence bytes, original scores, and all negative findings.
- Freeze each new comparison before calculating its results. These are retrospective development diagnostics, not preregistered independent tests.
- No test-driven hyperparameter or mixture selection. Report every specified arm.
- Resample complete recordings, retain common-support coverage, and separate raw causal predictions from offline negative interpolation.
- Use available local computation. Do not purchase cloud resources or send organizer messages.
- Keep a complete issue disposition, including unresolved external requirements.

## Design decision

Frame the article as an evaluated system. A universal superiority claim for the interface would require stronger evidence than currently available. Editorial changes alone would leave the scientific questions unanswered. Run the matched comparisons, then state exactly what they support; do not require favorable results to accept the revision.

## Review focus

1. Query identity and corrected targets must join exactly, without positional assumptions.
2. Missing geometry and unavailable RGB must not silently improve method denominators.
3. New fits must not reuse evaluation targets or confuse OOF feature generation with independent evaluation.
4. Offline interpolation must never be described as causal inference.
5. Rebuilt manuscript, numerical tables, source archive and publication hashes must agree.

## Tasks

- [ ] 1. Freeze comparison protocol and identify available final-model features, runtime evidence and control results.
- [ ] 2. Implement and test common-support evaluation; run the 434-query geometry/fallback/fixed-mixture/router comparison, including road-only, sequence-level and leave-Slider-750-out summaries.
- [ ] 3. Run matched final-pipeline fusion comparisons using identical upstream outputs. Recover final-fit features for equal-capacity head controls where available; distinguish inference interventions from retrained controls.
- [ ] 4. Reconcile V18 populations, recover the V8 controls, investigate historical numerical drift, and audit public-data options for genuinely independent evaluation.
- [ ] 5. Rewrite the main narrative, add a component input/output/supervision table, consolidate repeated material, and integrate the new results without claiming unmeasured superiority.
- [ ] 6. Regenerate numerical tables, compile and visually inspect the complete PDF, verify source package and tests, obtain final review, and update the authorized GitHub delivery.

## Execution record

Baseline: `7ec20a12f4df0526c556c31e1bbde5e3f5215497`; all three GitHub workflows passed. Existing environment `output/huggingface-check` uses CPU PyTorch and the locked pandas/numpy versions after the CI investigation. The original GPU environment must be checked separately before sensor inference.

Ruling: retain the existing workspace because the sealed experiment assets are rooted here; use a revision branch and separate output directories. Historical outputs will not be overwritten.
