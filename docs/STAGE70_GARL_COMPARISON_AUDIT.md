# Garl comparison audit for Stage70–76

The local source `E:\Garl-TTC` is clean at
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`, the handoff's required pin.
Source bindings and executable checks are saved under
`artifacts/stage70_76_architecture/preflight/`.

## Confirmed evidence

The exact historical Garl OOF CSV was recovered from
`e-jepa-ttc/artifacts/scientific_recovery_v7/baselines/garl_oof_predictions.csv`.
SHA-256 is `9b7ff010c0ae5ac2bc3c5e91c323ee238d4434f3b2194ec1beb26d660926f0bd`.
All 8,192 tokens, targets, sequences and outer folds match the current historical
tables. Its recomputed canonical MiD is **144.35302716561466**, with no invalid
TTC predictions. This is historical development evidence, with outer-dev
checkpoint selection; it is not independent confirmation.

The three historical `model_best.pt`, summary and prediction paths exist and
have been hashed without deserializing checkpoints. Their complete provenance
chain and trained-checkpoint forward replay still need verification.

The matched cached trainer imports the actual release `TTCNetwork` and
`prepare_optimizer`, clears both pretrained-checkpoint paths, and uses the source
loss. It does not train the compact EvTTC smoke replica as the historical Garl
comparator. Source `event_lhr.yaml` uses an event-only 40-channel ResNet50,
two raw predicted heights, two linear layers without an intervening activation,
and no foreground decoder. Its source-style schedule is Adam 0.001, no weight
decay, milestones 10/20/30/40, gamma 0.5, 50 epochs and native batch 128.

`audit_stage70_garl_source.py` verifies a strict random-weight state mapping
between this official event-only network and the local ResNet50 adapter. Both
have **24,674,178** parameters. On generated inputs, raw heights match exactly
and maximum TTC error is **2.9802322387695312e-08 seconds**. The timevolume
preprocessor matches the release exactly on generated odd-size ROIs and clocks
including a 2^40 microsecond offset. No pretrained checkpoint or real data is
used by this check. It performs zero optimizer updates.

These checks pass with Python 3.11.15 / PyTorch 2.11.0+cu128 from the existing
original project's environment. The environment is used without installing or
changing packages. Thirty-three directed unit tests pass in that interpreter
(24 existing Garl tests and nine new preflight-auditor tests). Full scientific
suite, real-data CUDA and exact-resume certification remain pending.

## Required prospective changes for S76

1. The historical runner always supplies `mask_target=None`. It can reproduce
   the official **event_lhr no-decoder ablation**, but cannot establish the
   handoff's foreground-supervised comparator. The default release
   `garl_ttc_eventdecoder.yaml` is RGB+event late fusion; calling that the
   event-only comparator would also be incorrect. Audit TRAIN-only foreground
   resources, freeze the explicit event-only decoder configuration and expose
   masks through the batch/loss interface before a foreground-Garl claim.
2. The historical grouped runner selects epochs on outer-dev. Preserve that
   historical result. S76 requires a distinct procedure: two profiles selected
   on three hashed FINAL-TRAIN internal groups, followed by three scratch
   refits. Confirmation groups must remain excluded throughout selection and
   training, and all six endpoints must freeze before their coordinated read.
3. Native batch size changes alter BatchNorm. Accumulation does not make batch
   32 equivalent to 128. Profile and disclose the authorized native batch
   before the pilots; record actual examples, epochs, updates and auxiliary
   exposure rather than claiming identical work from optimizer counts alone.
4. The official builder explicitly sets `target = selected[-1]`. Annotation
   timestamps and corrected exposure timestamps are separate fields. For
   physical PPS, trace the selected frame to its event-clock anchor. Do not
   equate the benchmark transform `-log(1-0.1/T)` with a measured two-frame
   scale, or parse the token suffix as the event clock. Actual 32-row bindings
   and raw-window timing are still pending.

The old materialized config contains a historical `public_validation_only`
metadata string. The grouped runner overrides source selection with its signed
TRAIN grouped contract; its summary records no public-validation tensor reads.
That string alone is not evidence of target exposure. A new S76 manifest must
state its actual resolved roles clearly.

No new Garl fit, foreground parity, independent confirmation, or SOTA result is
claimed. The comparison is partially audited, not execution-ready for S76.
