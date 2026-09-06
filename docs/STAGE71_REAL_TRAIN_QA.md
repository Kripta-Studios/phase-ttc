# Real TRAIN integration evidence

These are nonselectable technical results, not S72 endpoints or accuracy evidence.
Scientific optimizer updates remain zero. The campaign ledger records60 completed
technical updates:20 synthetic CPU and40 real TRAIN GPU, across six technical fits.
No uncertain optimizer attempt remains.

The full E128 cache contains8192 original rows in128 closed shards. An independent
`numpy.histogramdd` implementation re-read64 fixed TRAIN tokens (128 windows) with
bounded slices and obtained exact agreement in every count cell, with event-count
conservation. It does not call the production rasterizer.

The real GPU smoke used the5461 original fold0 TRAIN rows, with normalization and
initial phase prior fitted on precisely that view. No outer prediction was made.
DIRECT and PPS-TRAJ each ran10 continuous updates and5+save/reconstruct/reload+5,
using effective batch16, physical microbatch2, accumulation8, FP32 and TF32 off.
The technical schedule was10 updates with warmup2, not the scientific8000 schedule.

Both arms reproduced model state, AdamW state, CPU/CUDA/Python/NumPy/generator RNG
states, all losses, gradient norms and learning rates exactly. Their80-microbatch
sampling traces also match byte-for-byte, consuming160 TRAIN examples per fit.
Normalization is a common immutable input binding in all four technical recipes.
Correspondence augmentation resume and its100-update learning acceptance remain
separate pending requirements.

DIRECT's steady effective-update time was about0.35 seconds. PPS-TRAJ took about32
seconds after the first update, using checkpointed phase chunks of8. Peak allocated
GPU memory was301355520 bytes for DIRECT and708511744 bytes for PPS-TRAJ. All
2037822 PPS parameters had finite gradients; DIRECT excludes the1569 unused support
parameters. These are short technical measurements, not campaign-long throughput
guarantees. Equivalence-preserving execution optimization remains advisable before
long fits; scientific architecture, grid, loss and effective batch must stay fixed.

The public TRAIN annotation source contains one monolithic row group. The
document04-authorized administrative partition builder decoded it without emitting
confirmation target values or target-derived diagnostics. It published70741 TRAIN
rows across31 original/expansion groups and13268 sealed confirmation rows across6
reserved groups. The4735 previously protected-group rows were discarded. This is
an administrative label decode, explicitly recorded; it is not a claim of a
physical sandbox or a claim that those bytes were never read. The original8192
labels exactly reproduce the pinned targets after the original float32 conversion.
Independent confirmation still requires complete acquisition/ancestor proof and
the S76 locks before training/selection callsites can access its targets.

Target-free metadata sampling selected27307 additional rows from22 expansion groups,
with the fixed hash prefix, round-robin groups/tracks and≥20ms within-track spacing.
Each prospective fold universe can reach32768 rows including all original TRAIN
rows. The estimated uint16 cache reservation passes the amended120GB free-space
rule. D is not yet declared available: raw hashes, physical bindings, selected-row
usability and the complete execution reservation remain pending.

Evidence:

- `artifacts/stage70_76_architecture/histogram_reference64/HISTOGRAM_PARITY.json`
- `artifacts/stage70_76_architecture/technical_real_train_gpu/REAL_TRAIN_GPU_EQUIVALENCE.json`
- `artifacts/stage70_76_architecture/technical_real_train_gpu/SAMPLER_NORMALIZATION_EQUIVALENCE.json`
- `artifacts/stage70_76_architecture/UPDATE_LEDGER.json`
- `artifacts/stage70_76_architecture/role_partitions/ROLE_PARTITIONS.json`
- `artifacts/stage70_76_architecture/role_partitions/ADMINISTRATOR_READ_INTENT.json`
- `artifacts/stage70_76_architecture/expansion_inventory/EXPANSION_CANDIDATES.json`

Full baseline/new failure-ID comparison, conditional stage execution, pretext
acceptance, full diagnostic integration and the scientific lock remain incomplete.
