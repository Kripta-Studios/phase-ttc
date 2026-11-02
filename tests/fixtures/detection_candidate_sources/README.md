# Frozen source fixtures for the isolated detection candidates

These are exact copies of source files already used by the historical local
campaign, retained so source-transformation and AST-equivalence tests run in a
clean checkout. They contain no model weights, dataset samples or new results.
The `.py.txt` suffix marks immutable source data; the tests parse the files and
the static-patch tests compile transformed source, without importing these
snapshots as active modules.

`source_manifest.json` records the original campaign-relative path, byte count
and SHA-256 for each file. Do not format or normalize their bytes.

- `det2d_transformer.py.txt` is the MonoDGP portable source admitted by the R0
  static-source contract. It is covered by the [upstream MIT license](../../../docs/detection_local_20261029/evidence/monodgp_LICENSE.txt),
  copyright 2025 Fanqi Pu.
- `r0_train_detection_campaign.py.txt` is the original R0 producer used by the
  fixed-state single-update AST comparison.
- `writer_train_detection_campaign.py.txt` is the earlier producer used by the
  writer-only migration allowlist checks. Both producer snapshots use the
  repository's [MIT license](../../../LICENSE).

The portable-attention source is already public at
`src/e_jepa_ttc/detection/portable_attention.py`; tests reuse it and verify its
declared historical SHA-256 instead of duplicating it here.
