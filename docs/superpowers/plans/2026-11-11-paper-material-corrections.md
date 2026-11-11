# Material corrections before manuscript delivery

> Execution: native in this session; preserve existing working-tree corrections.

**Goal:** Deliver a synchronized corrected PDF, consolidated TeX, and source ZIP.

**Architecture:** Keep historical evidence immutable. Build the revised modular
manuscript, regenerate the corrected transfer statistics, and bind the delivery
to the exact inputs with a checked publication receipt.

**Tech stack:** Existing Python, NumPy, PyMuPDF, pdfLaTeX and BibTeX toolchain.

**Spec:** The user's accepted review: corrected 460-query transfer, fold-local
PHASE20 training description, study-specific bootstrap units, and current sources.

## Constraints and review focus

- No invented experiments, new training, official submission, or publication.
- Preserve the historical 468-query ledger and all previously saved predictions.
- Do not overwrite unrelated working-tree changes.
- Reject packaging after edits to manuscript, tables, evidence, figures or scripts.
- Reject packaging after edits to the built PDF/TeX or their validation receipt.
- Mark earlier review receipts historical; retain experiment dates and identities.
- Inspect new long tables, figures, references and conclusion in rendered pages.

## Tasks

- [x] Add regression tests for changed sources, added sources, changed PDF/TeX,
  missing build binding, and source edits during compilation. Observe failures.
- [x] Add `scripts/verify_publication.py` in the expanded manuscript directory;
  record source hashes around compilation and verify before packaging.
- [x] Update revision date, documentation and publication-boundary explanations.
  Preserve and incorporate the existing scientific source corrections.
- [x] Run correction and publication tests; rebuild both manuscript forms and
  resolve any layout/reference failures without changing evidence values.
- [x] Inspect all page contact sheets and detailed pages with revised content;
  record exact PDF hash and review scope.
- [x] Package; verify every ZIP entry and manifest hash, current evidence inclusion,
  and byte-identical delivered copies. Run final checks and update progress log.
