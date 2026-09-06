# Standalone diagnostic export contract

`architecture_diagnostics_v11.py` accepts inference-only arrays separately from
canonical metadata/targets and exactly token-indexed A5/C2F/PAIR predictions. It
does not read any data files or select checkpoints. The future inference executor
must bind model endpoint, row order, source roles and targets before this call.

The float64 export retains every input row, logits,108-bin posterior, the exact
stored phase edges, raw model median, emitted TTC and its benchmark phase, direct
logits, score correction, four-pair evidence and boolean observability. It records
posterior entropy, negative-phase probability, quantiles, tail mass, geometry
validity and RMS norms. The stored model median and float64 recomputation remain
separate so numerical rounding is visible. The phase displacement caused by finite
TTC clipping is separate from posterior uncertainty.

Target evaluation reuses the strict historical sequence/bucket mass and phase
transform. It exports signed TTC/phase errors, MiD, sign errors, all old expert
losses, old hard-selector and phase-hull oracle losses, old-plus-new oracle losses,
hull escape and whether the escaping prediction beats the old selector. Group
tables provide sequence and namespaced-track row counts, failure counts and
conditional weighted/unweighted losses. Oracle quantities are diagnostics only.

Calibration is explicitly `not_calibrated`. Phase-domain coverage50/80/95, widths,
CRPS, PIT and piecewise-uniform density NLL are measured without fitting a
calibrator. Zero posterior density produces an infinite row NLL and an explicit
count; the JSON aggregate is null, never silently truncated. Phase intervals are
not advertised as ordinary TTC intervals: signed inversion is discontinuous at
zero. Sensor/crop validity is not object foreground truth.

Any nonfinite or invalid-domain model output remains a failed row with NaN
derived fields. The complete model score, new oracle score and uncertainty
aggregates become unavailable rather than omitting that row. Historical old-only
oracle calculations remain available when their own inputs are valid.

Four synthetic tests pass: historical score parity, complete failure retention,
strict token ordering/boolean-mask validation, and a near-zero phase median whose
finite TTC phase differs materially. JSON summaries disallow NaN/Infinity.

This module is not the complete campaign analyzer. Paired grouped bootstrap,
joint multi-candidate/seed aggregation, mechanism interventions, full-cost timing,
endpoint provenance verification, decision records and packaging remain required
before scientific execution readiness or campaign completion can be claimed.
