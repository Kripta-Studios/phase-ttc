# Geometry reconstruction A/B/C — 2026-11-21

## Outcome

The clean replay is complete for tier A, the endpoint-defined tier B1, and the
three-seed refit tier C. A bounded real-media tier B2 was executed for one query
from each of the 12 sequences (12/5,805). B2 passes the final-TTC tolerance on
5/12 and the stricter endpoint tolerance on 0/12; it is therefore evidence of a
working media path, not a full-media equivalence claim. No private Test12 target
was read, no package was submitted, and no historical candidate was modified.

## Chains actually traversed

| Tier | Executed chain | Literal coverage | Result |
|---|---|---:|---|
| A/CSV | saved decimal CSV -> official JSON payload -> deterministic ZIP | 6,762 logical / 5,805 physical / 957 aliases | valid payload, but ZIP SHA `931fcfd6…`; decimal serialization is insufficient for byte identity |
| A/NPZ | saved float64 candidate NPZ -> official JSON payload -> deterministic ZIP | 6,762 / 5,805 / 957 | byte-identical SHA `8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df`; all TTC values exact |
| B1 | authenticated frozen PHASE20 endpoint diagnostics + public boxes -> geometry features -> frozen RGB_GEO_BAND seed-7 head -> 5,805 physical TTC -> 957 aliases -> ZIP | 6,762 / 5,805 / 957; 12 sequences | zero drift, zero sign changes, no missing/extras, byte-identical ZIP `8590f115…` |
| B2 multisequence | one real RGB history per sequence decoded from `E:\eAP_dataset` tar shards + authenticated frozen public event tensors -> nine hash-bound TRAIN40 components -> PHASE20 endpoint -> frozen geometry head | 12 / 5,805 physical; 0 aliases; all 12 sequences | endpoint gate passes 0/12; geometry TTC gate passes 5/12; this is path coverage, not numerical equivalence |
| C | public TRAIN40 split/features -> refit RGB_GEO_BAND for seeds 7, 13, 21 -> label-free Test12 physical predictions | 5,805 physical; 12 sequences per seed | seed 7 reproduces all 5,805 historical physical predictions exactly; seeds 13/21 expose material tail instability |
| Portable bundle | sealed endpoint NPZ + copied head state + explicit normalization contract -> 5,805 physical TTC -> aliases -> ZIP | 6,762 / 5,805 / 957 | byte-identical ZIP in normal and isolated read-only-input tests |

The B2 event branch was **not** regenerated from HDF5: it used the authenticated
frozen public event tensor.  Only RGB was regenerated from its media tar.  The
released `NativeObservation` ABI adapter was exercised before producer inference.
This distinction is intentional and machine-readable in `B_media_pilot/RESULT.json`.
No separate RGB-only head exists in this candidate, so an RGB-only TTC route was
not claimed or synthesized; RGB was exercised inside the sealed RGB+event chain.

## Drift and alias evidence

B1 has absolute TTC drift quantiles p0/p50/p90/p95/p99/p100 all equal to zero.
All 6,762 logical rows are finite and exact; the alias expansion is the frozen
957-to-physical mapping.  There are no sign changes, missing tokens, or extras.

B2 originally began with a one-query pilot, then expanded to one query from every
sequence. Across the 12-query multisequence replay:

- feature max absolute drift: p50 `4.6934e-06`, max `1.14358e-04`;
- width-logit max absolute drift: p50 `7.21514e-05`, max `6.55890e-04`;
- released-head TTC drift: p50 `2.28643e-04 s`, max `9.60541e-03 s`;
- geometry-head TTC drift: p50 `1.89304e-04 s`, max `8.43048e-04 s`;
- endpoint tolerance passes `0/12`; geometry TTC tolerance passes `5/12`.

The endpoint mismatch is consistent with numerical runtime drift, but this replay
does not attempt to prove its cause. The preregistered `1e-6` endpoint gate fails
for every sampled sequence, and the geometry tolerance also fails for 7/12.
At the measured post-load mean of `1.17169 s/query`, a linear estimate for 5,805
queries is `6,801.67 s` or `1.889 h`; this is not a measured full-run duration and
excludes one-time model loading.

For C, seed-7 refit versus the historical seed-7 state has zero drift at every
one of 5,805 physical queries despite non-identical torch container bytes.  Across
seeds, the per-row range has p50 `0.14773655 s`, p90 `0.49600420 s`, p95
`0.73096600 s`, p99 `2.12854334 s`, and max `119.99999237 s`.  Relative to seed
7, seed 13 changes sign once and TTC band 109 times; seed 21 changes sign three
times and band 135 times.  Worst rows and per-sequence maxima are retained in
`C_refit/RESULT.json` and `C_refit/PER_SEQUENCE.csv`; these tails must not be
hidden by the exact seed-7 replay.

## Authenticated inputs and weights

- public Test12 inventory SHA-256: `b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526`;
- official Test12 parquet SHA-256: `aedb9da5484ceedb642ef1cbc7cf6115dcf7f368e4444b55de947989fc19a362`;
- exact media inventory revision: `a814a946…` (local evidence inventory);
- sealed model SHA-256: `cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`;
- released PHASE checkpoint SHA-256: `a24bfc72ca7f53a93d53db806b6866c6a5a4b6abde074b5dfec50436b7005408`;
- historical geometry state SHA-256: `a2d68afc1232a7ebc74e0acd107917c758fe661f962f510f01d470b9d31f5229`.

The B2 result additionally records all nine deployed component hashes and both
normalizer hashes returned by the portable candidate verifier.

## Reproduction commands

Use Python 3.11 with repository `src` first on `PYTHONPATH`.  The core commands
used were:

```powershell
python -m operational.phase_geometry.data --root . --output operational/geo_reproduction/runs/B_endpoint_replay
python -m operational.geo_reproduction.media_pilot --candidate D:\E_JEPA_POD2_RETURN_20261027\FINAL_CANDIDATE --inputs ..\e-jepa-ttc-v16-multimodal-validation\artifacts\multimodal_research_20261012\test12_phase_only\INPUTS.json --output operational\geo_reproduction\runs\B_media_pilot --device cuda
pytest -q operational/geo_reproduction/test_replay.py
```

The exact refit states and predictions are in `C_refit`; the training invocation
was `run_job(Job("RGB_GEO_BAND", -1, seed), work)` for seeds 7, 13, and 21.
The refit runs did not submit predictions to the private evaluator.

## Evidence map

- `runs/A_saved_predictions/{RESULT,NPZ_RESULT}.json`: source-sufficiency and exact ZIP replay.
- `runs/B_endpoint_replay/{DATA_RECEIPT,RESULT}.json`, `PER_TOKEN.csv`: full endpoint replay.
- `runs/B_media_pilot/{RESULT,ENDPOINT_COMPARISON}.json`, `captured_endpoints.npz`: actual media inference and tensors.
- `runs/B_media_multiseq/RESULT.json`: 12-sequence media coverage, drift and timing estimate.
- `runs/C_refit/RESULT.json`, `PER_SEQUENCE.csv`, `PREDICTIONS.npz`, `jobs/*`: independent refits.
- `bundle/` plus `bundle_cli.py`: redistributable head, normalization contract, manifest and replay CLI.
- `runs/D_readonly_clean/READ_ONLY_TEST.json`: isolated read-only-input replay receipt.
- `PROTOCOL.md`: tolerances fixed before the B2 output was inspected.

All run paths above are relative to `operational/geo_reproduction/`.

The repository includes the portable head, per-sequence refit diagnostics and measured run summaries. Media, feature caches and intermediate tensors stay with their original data providers or in the local run directory. To check the scored ZIP from authenticated endpoint inputs, run the following from the repository root.

```powershell
python -m operational.geo_reproduction.bundle_cli --bundle operational/geo_reproduction/bundle --endpoint-inputs "$env:PHASE_GEO_ENDPOINT_INPUTS" --output artifacts/reproduced_geo
```
