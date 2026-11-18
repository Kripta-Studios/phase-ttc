# Protocol: event-history mismatch test (training-matched vs delivered history)

Status: **pre-declared**. I wrote this file before computing any arm-B prediction and
before joining any label to an arm-B output. Repository commit when it was written:
`6eefa20dc558a94cd8d3d0bfd7b639417cf92642` (branch `audit/claim-scope-and-sota-20261118`).

Scope: a retrospective diagnostic on **EvTTC, which this project has already seen during
development** (the 32 recordings and the 460-query cohort have been inspected before). The
result is conditional on that cohort. It does not count as independent confirmation and
cannot support a SOTA claim. No model is trained, fine-tuned or selected here. All nine
sealed components stay frozen: `D:\E_JEPA_POD2_RETURN_20261027\FINAL_CANDIDATE`, verified
by `scripts.final_train40_transfer.verify_candidate`, seal
`cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.

## 1. The defect being tested

In training (`operational/final_refit/event_history.py::_query_plan`, `scientific_oof_features.py::event_raw`),
the event history of a query is built as follows:

* storage layout H16; lags `15..0 × 50 ms`; only the last 8 slots (lags 350, 300, …, 0 ms)
  are candidates;
* every slot is encoded **inside the current query's object crop square**
  (`square_xyxy` of the query) by `e_jepa_ttc.efficient_context.mapped_union.encode_union`
  with `roi_size=128`, the query's own three context windows shifted back by the lag, and
  5 bins per polarity;
* a slot is valid iff (a) raw support: `windows.min() - lag >= first_event_us`, and (b) causal
  budget: `available_us - (anchor_us - lag) + input_span_us <= 650 ms`, where
  `anchor_us = windows[-1,1]` and `input_span_us = windows[-1,1] - windows[0,0]`;
* PAIR input: delta = the query's own window spacing (`delta_t_s`), the same for every slot;
* timing = `clock_features(anchors=anchor_us - lag, arrivals=[available_us]*n, query=available_us)`.

Measured on the stored training cache (`final_generated_fd84565b0df06d79.tar`,
`oof/PHASE17_RAW_EVENT.npz`): 88,617 of 88,744 rows have valid pattern `01111111`
(lags 300..0 valid, lag 350 excluded by the budget because the exposure offset is positive).
The other 127 rows are cold starts with 1, 3 or 5 slots. In the fusion input
(`oof/PHASE20_CONTEXT.npz`), event field 17 (the gap between the last two anchors) is 0.05 s
(or 0 for single-slot rows), and field 19 (valid/8) takes only the values {0.125, 0.375, 0.625, 0.875}.

The EvTTC transfer inference that was delivered (`scripts/final_train40_transfer.py::FinalTransferPredictor`,
`operational/rgb_port_revision/transfer_inputs.py::prepare_event`) used instead:
`EVENT_LAGS_US = 300, 200, 100, 0 ms` (4 slots, 100 ms cadence). It used the same
`encode_union` in the current query square (`event_pixel_diff=0.0`, since the EvTTC square is
already projected into event coordinates), PAIR delta 0.1 s, and
`timing = clock_features(anchor - lag, [query]*4, query)` with `query = anchor_us`. So the
fusion head received gap **0.10 s** and valid **4/8 = 0.5**, two values that never occur in training.

Note on Test12 (eAP): Test12 inference (`operational/final_refit/test12.py`) chains the
observations of earlier queries, each encoded in its own crop with its own clocks. On EvTTC
the delivered history already uses the current query crop. **On EvTTC the tested difference
is therefore the slot grid (cadence, number of slots) and the timing fields, not the crop.**

## 2. Arms

All arms use the same raw events, the same crop (`row.square_xyxy`), the same windows
(`row.windows_us`: three 100 ms windows ending at the RGB frame timestamp `anchor_us`), the
same `event_pixel_diff=0.0`, the same `encode_union` call (`roi_size=128`,
`retained_bytes_max=1 GiB`), the same producer path (`operational.rgb_port.transfer._expert_block`
with `ProducerObservation` patched to `NativeObservation`, exactly as in `FinalTransferPredictor`),
the same normalizer, the same E_CTX (`frozen_context_state`) and the same released head
`PHASE_ONLY_EVENT_S7_TRAIN40`. The RGB context block (`features[:, :170]`) and the RGB width
logits come unchanged from the delivered states in `head_evaluation.npz`, because the RGB history
is the same in every arm.

| Arm | Event slots (lags, ms) | valid/8 | last gap | timing arrivals | Role |
|---|---|---|---|---|---|
| **A** (delivered) | 300, 200, 100, 0 | 0.5 | 0.10 s | `query` | reference: saved delivered values |
| A-replay | same as A, recomputed here | 0.5 | 0.10 s | `query` | parity check and same-platform reference |
| **B** (training-matched, primary) | 300, 250, 200, 150, 100, 50, 0, padded slot at 350 | 0.875 | 0.05 s | `query` for all slots | primary comparison |
| B8 (sensitivity) | 350, 300, …, 0 if raw support exists | 1.0 | 0.05 s | `query` | descriptive only |

Why B excludes lag 350: EvTTC has zero offset between the event anchor and query
availability (`query = anchor_us`). Applied literally, the training rule admits lag 350 at
exactly 650 ms, which gives 8/8 valid slots, a count that occurs in **0** of the 88,744
training rows. The training-matched history is therefore defined as the history every
300 ms-span training row actually received: slot 0 padded, lags 300..0 valid. B8 (literal
rule) is reported only as a sensitivity arm and plays no part in the decision. The total causal
budget of B is `query - (anchor - 300 ms) + 300 ms = 600 ms <= 650 ms`, and no event at or
after `anchor_us` (= query time) is read (half-open windows).

Residual mismatches that I will not remove, and will report in all arms: event fields 15/18
(query − anchor and arrival − anchor) are 0 on EvTTC, against a training range of
[0.000904, 0.019992] s. The PAIR input uses the inference `pair133` path in all arms.

## 3. Population

All 460 queries in `head_evaluation.npz` (32 recordings; tokens = `query_id` of
`output/phase_adapt_final/data_v3/POPULATION.json`, SHA256 `d976c31c…`). If the measured cost
of the parity sample (§4) projects more than 4 hours for all 460, the pre-declared fallback is
`uniform_indices(n_g, 6)` per recording in chronological order (192 queries, the same in every
arm). Labels: `head_evaluation.npz["target"]` (corrected targets, SHA256 of the file `bded0f0a…`).
I join them only in the scoring step, after the arm-B predictions are written to disk.

## 4. Reproduction gate (arm A)

Before running B, I replay A from raw events on the first 2 queries of each of the first
5 recordings in sorted order (10 queries). The gate requires median |Δ| of the F0 TTC ≤ 1e-4 s
and max |Δ| ≤ 1e-2 s against `released_prediction`, plus the same bounds for E_CTX TTC against
the delivered `E_CTX_TRAIN40` and max |Δ| ≤ 1e-4 for the 10 event physical fields
(features 330:340). If the gate fails, I stop and document the failure. A-replay is then also
computed for all queries and reported.

## 5. Metrics

For each arm and each of two readouts, the event context **E_CTX** (`phase_to_ttc(E_CTX point_phase)`)
and the released fusion **F0** (`PHASE_ONLY_EVENT_S7_TRAIN40` TTC):
MAE (s), median AE (s), p95 AE (s), RTE % = mean(100·|p−t|/|t|), and
MiD = mean(100·|φ(p) − φ(t)|) with φ(T) = −log(1 − 0.1/T) (`e_jepa_ttc.simplex_t.phase.ttc_to_phase`).
The R_CTX output (RGB context only) is reported as the no-event reference.

## 6. Statistics

Paired bootstrap at the recording level: 10,000 resamples of the 32 recordings with replacement,
`numpy.random.default_rng(20261118)`. Each resample pools all queries of the drawn recordings,
with multiplicity. The statistic is metric(B) − metric(A) under the same draw. The interval is
the percentile 95% CI. The same draws are used for all metrics and contrasts.

## 7. Pre-declared decision rule

Primary endpoint: ΔMAE_F0 = MAE(F0, B) − MAE(F0, A delivered).

* **"Matched history helps"** iff the upper bound of the 95% CI of ΔMAE_F0 is < 0 **and** the
  point estimate of ΔMiD_F0 is < 0.
* **"Matched history hurts"** iff the lower bound of the 95% CI of ΔMAE_F0 is > 0.
* Otherwise: **"no measurable effect"** on this cohort.

Secondary readouts use the same rule, are labelled secondary and do not change the primary
decision: (i) the E_CTX readout; (ii) "event branch benefit with matched history",
MAE(F0, B) − MAE(R_CTX); (iii) B vs A-replay (same platform).

## 8. Support of the fusion inputs

For each of the 20 PHASE_ONLY input fields (fields 0–9 RGB, 10–19 event), I compare arms A
and B with the training distribution `PHASE20_CONTEXT.npz` features [88744, 20]: min/max,
p0.5/p99.5, the fraction of queries outside [min, max] and outside [p0.5, p99.5], and, for
discrete fields (17, 19), membership in the training value set.

## 9. Test12 (only if raw eAP test events are present and time allows)

Same B rule with eAP parameters (`event_pixel_diff=5`, the training value). For each physical
query, I use its current event observation (windows, square, delta, raw path), lags 350..0 ×
50 ms, the training validity rule with that observation's `available_us`, and timing arrivals =
`available_us` of the current observation. Output: arm-B event blocks and E_CTX outputs, saved
as npz. Disagreement with submitted predictions is reported; accuracy is not (Test12 labels are
private and are never read). No package is built and nothing is submitted.
