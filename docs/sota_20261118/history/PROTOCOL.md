# History-length ablation of the PHASE-TTC context refiners: protocol

Status: registered 2026-11-18 before any truncated-arm training or comparison.
Only two things were done before writing this: inspecting the stored arrays (shapes, valid
masks, timing fields) and planning a fidelity check that reloads the archived released
checkpoints (no new fit). Any later deviation goes in `REPORT.md` under "Deviations".

## 1. Question

How much of PHASE-TTC's eAP out-of-fold (OOF) accuracy comes from the causal history
that the context refiners (E_CTX for events, R_CTX for RGB) read beyond the most recent
observation(s)? And does any benefit come from RGB history, event history, or both?

This is an internal control on development folds. It is not a test-set result, and it
does not retrain Garl-TTC (see section 8).

## 2. Stored inputs (no regeneration needed)

The archived pod return `D:\E_JEPA_POD2_RETURN_20261027\` holds everything the upper
stages need, for every one of the five released rotations `r = 0..4`:

| Item | Archive member (relative to rotation root) |
|---|---|
| refiner-role training cache (normalized PHASE17, timing, valid, expert phases, targets, groups) | `caches/{E,R}_CTX.npz` |
| refiner-role label-free normalizer | `normalizers/{E,R}_CTX.json` |
| generation-role raw PHASE17 histories (target-free) | `generation/{E,R}_CTX/fragment.npz` |
| released refiner checkpoint (fidelity check only) | `fits/{E,R}_CTX/checkpoint_last.pt` |
| released stitched OOF context outputs | `final/oof/PHASE20_CONTEXT.npz` |

Rotations 0–1 come from `final_inputs_63e2720817ba6521.tar`. Rotations 2–4 come from
`event_r{r}_{support,generation,fits}_*.tar` (events) and `rgb_fold_closure_06486d50af021f74.tar`
(RGB). Each array is `[N, 8 slots, 17]` with timing `[N, 8, 4]` and a contiguous-suffix
validity mask (slot 7 = current observation). Members are read in memory and never
extracted. Every member's SHA-256 is checked against `RESTORE_INDEX.json` and recorded in
`artifacts/sota_20261118/history/INPUT_MANIFEST.json`. Supervision (`target_ttc`,
`target_phase`, `group_id`) comes from the TRAIN40 role rows
`E:\E_JEPA_TTC_RGB_PORT\metadata\{P,H,V}_ROWS.parquet`. No Test12 file is read.

Observed slot occupancy (TRAIN40, 88,744 queries):

* Events: at most 7 valid slots (slot 0 is never valid under the 650 ms budget). Anchor
  lags are 0, 50, …, 300 ms (+ ≤20 ms query delay). Each event observation's own input
  span is about 300 ms (three event windows). It is bounded to (299, 330] ms by the
  650 ms suffix rule.
* RGB: at most 5 valid slots. Anchor lags are 0, 100, 200, 300, 400 ms. Each RGB
  observation is produced from 3 frames spanning about 200 ms (`producer_frame_count = 3`);
  the exact median of `actual_span_us` is recorded at run time.

## 3. Arms

Truncation operator `T_K` (applied to a single row; causal): keep only the `K` most recent
valid slots. For dropped slots, set `valid = False` and set features and timing to exactly 0.
On the oldest kept slot, set the inter-anchor gap timing field (index 2) to 0. This matches
how `clock_features` would build a `K`-observation history, so no clock information about
dropped observations survives. The operator is applied after per-slot normalization
(normalization is elementwise per valid slot, so the order does not matter). It is applied
identically to the training cache and the generation inputs.
`T_8` is the identity on these data.

`K ∈ {1, 2, 4, 8}` for each modality. Mapping to milliseconds:

| K | event anchor-lag support | event sensor support (≈ lag + 300 ms) | RGB anchor-lag support | RGB sensor support (lag + ≈200 ms) |
|---|---|---|---|---|
| 1 | 0 ms | ≈300 ms | 0 ms | ≈200 ms (3 frames) |
| 2 | 50 ms | ≈350 ms | 100 ms | ≈300 ms (4 frames) |
| 4 | 150 ms | ≈450 ms | 300 ms | ≈500 ms (6 frames) |
| 8 (released) | 300 ms (7 slots) | ≈600–650 ms | 400 ms (5 slots) | ≈600 ms (7 frames) |

Fusion arms `(K_RGB, K_EVENT)`, 10 in total:

* joint: (1,1), (2,2), (4,4), (8,8);
* RGB-only truncation: (1,8), (2,8), (4,8);
* event-only truncation: (8,1), (8,2), (8,4).

Reference (not an arm, fidelity only): fusion on the archived released `PHASE20_CONTEXT`
features. It uses the same fusion protocol.

## 4. Pipeline per arm (identical recipe for every arm)

1. **Context refiners: the released rotation scheme, unchanged.** For rotation `r`
   (generation block `B_r`, refiner block `B_{r+1 mod 5}`, producers `B_{r+2..r+4}`), train a
   fresh `TemporalRefiner(feature_count=17, hidden=160, gru, residual)` on `T_K(refiner
   cache)`. Use the released loop (`operational.final_refit.training._loss`, `learning_rate`,
   `_balanced_mass`, AdamW lr 3e-4, wd 1e-3, betas (0.9, 0.999), grad clip 1.0, batch 128,
   `torch.randint` sampler from a CPU generator seeded with the seed, deterministic
   algorithms). Updates are the released per-rotation counts
   `ceil(2500 · N_refiner / 18422)` = 2500, 2434, 2479, 2402, 2230. Then emit
   `frozen_context_state` on `T_K(normalize(generation fragment))`: 10 physical coordinates
   plus 2 width logits (RGB). Stitching the five generation blocks gives OOF context outputs
   for all 40 sequences, exactly as the released `PHASE20_CONTEXT` was built.
2. **Fusion head (PHASE_ONLY, released architecture `PhaseOnlyFinal`).** The released head
   was fit on all 40 sequences with no internal evaluation. Here it is refit in five
   sequence-grouped folds. For test block `B_b`, train on `B_{b+1}, B_{b+2}, B_{b+3}`
   (24 sequences) and exclude `B_{b-1}`, because `B_{b-1}`'s context features came from a
   refiner trained on `B_b`'s labels. The recipe is the released phase-only rule (batch 128,
   wd 1e-3, the same lr schedule, balanced group mass, updates
   `ceil(2500 · N_train / 18422)`). Predict `B_b`.
3. Seeds: 7 (primary, the released seed). 13 and 21 are added if the measured cost allows.
   The seed controls refiner init and sampling and fusion init and sampling. It is the same
   for every arm.

Device: CPU or CUDA with deterministic algorithms; the device is recorded. The released
fits ran on CUDA, so a retrained `K=8` is not expected to be bitwise identical to the
archive. Its agreement with the archived outputs is reported as a fidelity number.

## 5. Leakage accounting

For each query in test block `B_b`, no model in its own chain saw its sequence:
producers `B_{b+2..b+4}`, refiner `B_{b+1}`, fusion `B_{b+1..b+3}`. Paths that remain:

* **Fixed producers (second order).** The PHASE17 inputs of the fusion-training blocks were
  produced by rotation producers that include `B_b` among their 24 training sequences.
  `B_b`'s labels can therefore shape the inputs the fusion head is trained on, though never
  `B_b`'s own features. Removing this path needs regenerating producers (GPU-heavy) and is
  out of scope. It is shared by all arms, because producers are fixed.
* **Label-free normalizers** are fit on the refiner block (released, unchanged; no targets).
* **Design-level selection.** The architecture, recipe, 650 ms budget and update counts
  were developed on these TRAIN40 sequences with `K=8`. Any such selection effect favours
  the released arm, so differences against `K=8` may overstate the value of history.
* Test12 labels are never read. Nothing is selected on any evaluation fold: every arm is a
  fixed endpoint, with no early stopping and no checkpoint selection.

## 6. Metrics

* MiD per query is `1e4 · |log(1 − 0.1/T) − log(1 − 0.1/T̂)|`, computed exactly as the
  official scorer `score.py::evaluate_rows` does. Bands are by target: crucial (0,3],
  small (3,6], large (6,10], negative (−10,0]. Overall = 0.5c + 0.3s + 0.1l + 0.1n, with
  band means pooled over queries as in the official scorer.
* Seconds MAE `|T̂ − T|` over all queries and per band. `T̂ = phase_to_ttc(location)`,
  the released output map.
* Secondary diagnostics (no claims rest on them): context-only R_CTX and E_CTX point-phase
  MiD by `K`, before fusion.

## 7. Statistics

* Paired sequence bootstrap: 10,000 resamples of the 40 sequences with replacement, fixed
  seed 20261118, the same resample indices for every arm and contrast. Band means are
  ratio estimators (resampled band MiD sum / resampled band count). Report percentile 95%
  CIs.
* Pre-registered contrasts, each as `arm − (8,8)` (positive means truncation is worse):
  every arm; headline **(2,2) − (8,8)**, the "≈ two-observation" regime (100 ms RGB anchor
  lag, 50 ms event anchor lag).
* Attribution: RGB history = (2,8) − (8,8); event history = (8,2) − (8,8); interaction =
  (2,2) − (2,8) − (8,2) + (8,8). The same contrasts are reported at K=1.
* The primary inference uses seed 7. If extra seeds run, report per-seed values and the
  seed mean (the bootstrap of the seed mean uses the same sequence resamples across seeds).
* The answer is stated as point estimate plus CI, in Overall-MiD points, per band. A CI
  that includes 0 is reported as "not resolved", never as "no effect".

## 8. Limits stated up front

* Development-fold control on the 40 eAP TRAIN sequences. The levels are not comparable to
  the GarlTTC test leaderboard score (35.80 vs 45.03); only within-protocol differences are
  interpreted.
* Even `K=1` is not Garl-TTC's two-frame regime. Each single observation already comes from
  producers that see several frames: RGB 3 frames / ≈200 ms, events ≈300 ms of windows.
  "No history" means no refiner history, not a single frame.
* Garl-TTC is not retrained with longer history, so this measures what history buys
  PHASE-TTC, not what it would buy Garl-TTC.
* Training/inference spacing caveat: in this TRAIN40 pipeline event history uses 50 ms
  spacing and RGB uses 100 ms spacing. A `K` therefore means different milliseconds per
  modality (table in section 3).
