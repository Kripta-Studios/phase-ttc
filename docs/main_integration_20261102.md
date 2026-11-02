# Main integration — 2 November 2026

This integration reviews work from 28 October through 2 November across local
and remote branches. The comparison started at `main` / `origin/main`
`b736c9df9491c2e50bff8a5c2bed834cc935f871`, after fetching all remotes.

## Branch decisions

| Branch | Reviewed tip | Decision |
|---|---|---|
| `dev/eap-diagnostics-20261101` | `b89c3bf9e654514e3a824f1b823ea9b07764465a` | Merge all 113 commits absent from the starting main. |
| `dev/eap-s0-evaluation-20261029` | `3b6cfe2f5d9377cc5386e0533ae58c4d941decd2` | Fully contained in the diagnostics tip. |
| `dev/eap-detection-local-20261029` | `24b8f36f6e4aa17c965f9526d8cf486b512a8046` | Fully contained in the diagnostics tip. |
| `research/evttc-transfer-v2` | `335f00e083a732fe0882fd0472c492fb8dd5fa43` | Already included in main through PR #2. |
| `release/phase-ttc-final-20261029` | `3434a3be1209c0fed878c2839125aaf51f07d1d0` | Already included in main. |
| `perf/train40-persistent-50hz` | `0f10312dafe64bfccc85f4fa363712881dc1f9a2` | Already included in main, including the local tip beyond the remote tip. |
| `chore/phase-ttc-main-branding` | `b3d068457fc676284510223bb0137a8fedb15272` | Already included in main. |

The three conflicts in `.gitattributes`, `pyproject.toml` and `docs/progress.md`
were resolved by retaining both branches' evidence-preservation rules and
chronological records. No implementation or test paths conflicted.

## Additional work included

- The complete expanded *PHASE-TTC: From Motion to Time* article, its 65-page
  PDF, LaTeX sources, figures, numerical evidence and review records. The
  seven-page companion is explicitly secondary. The original 28 October
  manuscript remains available. All three editions remove the corporate logo
  and use the author's requested contact details, `alvaro@schwiedop.com` and
  `https://schwiedop.com/PHASE-TTC`; scientific results and figures are retained.
  Their compiled sources, PDFs, manifests and source bundles are synchronized.
- The old S0 worktree contained 52 additional source, test, documentation and
  receipt files absent from the diagnostics branch. The isolated candidates
  are preserved with historical-status notices and portable test fixtures;
  see [the candidate integration record](detection_sota_20261029/PENDING_CANDIDATES_INTEGRATION_20261102.md).
  The originating worktree is not modified.
- The wheel now includes the original Apache-2.0/SenseTime license associated
  with the adapted deformable-attention operator, in addition to the root MIT
  license. Executed source identities are unchanged.

Raw datasets, checkpoints, experiment output directories, browser inspection
snapshots and temporary build products remain outside Git. Small public
receipts, publication PDFs and reproducible source inputs are included.

## Scientific boundary

This is a source and publication integration. It does not promote an
experimental model, rerun training, alter a benchmark submission or establish
new accuracy or speed results. Original TRAIN40 TTC, adapted EvTTC TTC and RGB
3D detection retain separate model identities and evaluation populations.
The local historical AP, aligned local mAP and official detection score remain
separate. Failed controls and unresolved evaluation discrepancies are retained.

## Validation

Pre-push checks on the integrated tree:

- Ruff lint passes; all 3,577 checked Python files satisfy formatting, with
  documented exact-byte exceptions for executed sources.
- Pyright 1.1.411 reports zero errors for the maintained-module configuration,
  the manuscript CI entry point and the isolated static-attention configuration.
- All PowerShell scripts parse, the lockfile is current, and the final wheel
  builds. Both installed command names display help successfully. Both license
  texts in the wheel match their repository sources byte for byte.
- The recovered candidates pass 74 CPU tests, including portable fixtures and
  a source-identity guard. No GPU training or experiment is run for integration.
- The local complete test suite is also being run. Its five reported cases
  pass in isolated reruns: two ONNX-provenance cases, the pipeline dry-run and
  two CUDA attention-parity cases. The ONNX subprocesses had selected an
  incomplete local environment; `UV_PROJECT_ENVIRONMENT` selects the verified
  Python 3.11 environment for the reruns. The complete attention test module
  also passes on this GPU in isolation. This record does not claim that the
  ongoing full run has passed; the final clean-environment CI remains required.
- Manuscript validation checks the original 200-file / 201-member package,
  expanded and companion manifests, matching modular/consolidated text,
  embedded fonts and working contact links. The 52-, 65- and 7-page PDFs contain
  no corporate branding. For the original and expanded editions, every page
  after the cover is pixel-identical to the preceding reviewed proof.
- Independent review confirms branch-conflict unions, source/receipt hashes,
  isolated candidate status and absence of datasets, checkpoints or credential
  patterns in the proposed files.

The final pushed commit is also subject to the repository's `ci`, `smoke-train`
and `paper` workflows. Their authoritative outcomes are available in
[GitHub Actions](https://github.com/Kripta-Studios/phase-ttc/actions).
