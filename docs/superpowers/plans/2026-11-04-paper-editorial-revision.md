# Paper Editorial Revision Implementation Plan

**Goal:** Apply the supplied editorial guide to the expanded PHASE-TTC manuscript while preserving the measured results and evaluation populations.

**Approach:** Edit modular LaTeX sources, regenerate the consolidated manuscript, compile both versions, and package matching delivery files. Preserve the existing real sensor figures and historical evidence bytes.

**Stack:** LaTeX, the existing Python paper builders, PyMuPDF, SHA-256 manifests.

**Specification:** User instructions and `paper/from_motion_to_time_20261102/GUIA_REESCRITURA_NATURAL_PAPEL.tex`.

## Constraints

- Preserve generated numerical macros and full-precision result tables.
- Use impersonal academic prose consistent with single authorship.
- Retain development exposure, statistical intervals, negative outcomes, signed-output policy, and measurement boundaries where relevant.
- Do not introduce unsupported independent audits, blind evaluation, ADAS standards, causal explanations, released tags, or general JEPA conclusions.
- Keep historical IDs, checkpoints, result ledgers, and source evidence unchanged.

## Tasks

- [x] Rewrite abstract, contributions, model explanation, training protocol, results, runtime, and discussion around the scientific questions.
- [x] Replace the chronological history with a methodological account; move revision identifiers into a reproducibility table.
- [x] Rewrite first-person execution narration; explain internal variant names, group evidence markers, and round prose without changing tables, equations, exact timestamps, or decision thresholds.
- [x] Verify numerical macros, evidence inputs, labels, and sensor assets; compile and compare modular/consolidated PDF text, then inspect affected pages.
- [x] Regenerate publication manifest, delivery PDF/TeX/ZIP, and record the supported departures from the guide.

## Review focus

- Official MiD is 35.80 versus 45.03, with 20.49% relative reduction from the existing macros.
- The labeled transfer set has development exposure; its paired intervals include zero.
- Slider-750's evaluator discrepancy has no established causal explanation.
- Runtime uses a laptop GPU and preloaded data, not physical-camera hardware-in-the-loop input.
- Local adaptation, canonical-cadence candidates, and the submitted model remain separate variants.
