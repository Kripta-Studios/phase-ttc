# P5 reliability gate — frozen exploratory protocol

Frozen on 2026-11-21 before this directory contained predictions or accuracy metrics.

## Status and population

This is one exploratory candidate on the real eAP TRAIN40 NPZ population (88,744 rows,
40 sequences). P3 outer-fold producer training is not yet executable in this checkout, so this
study is **OOF at head level only**: the five held-out sequence blocks exclude each evaluation
sequence from each newly fitted head, while the frozen RGB/event producers may have seen
TRAIN40. It is not full-system OOF and cannot close P3. FCWD and Test12 are forbidden inputs.

## Single candidate and matched control

Both arms use the same residual MLP, parameter count, optimizer, updates, seeds (7, 13, 21),
five sequence folds, RGB ten-vector, causal box geometry and causal robust-box fields.

- `RELIABILITY_CONTENT`: additionally receives the nine pre-existing causal event-content
  fields. At inference its output is admitted only when content is valid and box geometry is
  unreliable; otherwise the matched-control output is returned exactly.
- `RELIABILITY_NULL`: receives the same 42-column tensor and the same content-validity masks,
  row clocks and rows, but the seven continuous content fields are zero. The final two validity
  mask columns remain unchanged. Thus capacity and missingness information are matched while
  event content is removed.

The fixed, label-free unreliable-box rule is true when any current box edge is clipped, robust
geometry reports saturation, a missing robust source, or at least one clipped edge. Content is
valid only when its pre-existing `valid` flag is one. Invalid content always falls back exactly
to `RELIABILITY_NULL`; there is no learned or imputed fallback.

## Primary contrast and advance rule

The signed contrast is `MiD(RELIABILITY_CONTENT) - MiD(RELIABILITY_NULL)`; negative is better.
MiD uses the existing four-band Overall weighting. Sequence bootstrap uses 10,000 draws and
seed 20261121 after averaging the three seed predictions per row.

Advance only if all conditions hold:

1. `FREE` (box reliable): upper 95% CI is at most **+0.5 MiD** (non-inferiority).
2. `DIFFICULT` (box unreliable and content valid): point improvement is at least **2.0 MiD**
   and the upper 95% CI is below zero.
3. `ALL`: upper 95% CI is below zero.
4. Invalid-content predictions are byte-identical to the matched control and all 15 jobs cover
   their held-out rows exactly once.

Real saturation (expected to be small), synthetic saturation from the previously frozen
virtual crop, crucial/short/long/noncritical bands and approaching/away rows are descriptive
strata. An inconclusive real-saturation result is reported as such and does not get replaced by
FCWD tuning. No additional candidate, threshold, feature or seed may be selected after metrics.

## Causality and outputs

The run must write a no-future receipt checking the pinned upstream receipts, half-open event
windows ending at each saved event clock, positive backward lags, causal box-source contract,
fold exclusion and absence of Test12/FCWD paths. It also writes per-sequence results, failure
rows/reasons, strata, predictions, hashes and an explicit decision. CPU threads are fixed to one
inside each worker; the default runner uses at most two workers.

## Corrective amendment before the valid run

The first execution was invalidated because its band-mass vector was computed on all TRAIN40
targets before selecting the 32 training sequences. No result from that execution is admissible.
The valid run computes band mass from `truth[train_rows]` inside each fold; a regression test
changes every exterior target and requires identical training rows and masses. The valid run is
written to a new directory. After observing that averaging opposite-sign predictions can create
an undefined MiD even when each seed is finite, scorer coverage was added as a **post-hoc
validity diagnostic**, not retroactively called a preregistered gate. The report preserves the
frozen efficacy verdict separately and labels the operational interpretation inconclusive when
mean-prediction coverage is incomplete. The no-future artifact is likewise a structural
provenance receipt over cached producers, not a per-event replay.
