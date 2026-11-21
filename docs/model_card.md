# Model card — PHASE-TTC

Model release: 11 November 2026; benchmark and measurement update: 21 November 2026.
Author: Álvaro Schwiedop Souto.

PHASE-TTC estimates signed time-to-contact from causal event/RGB observations and supplied object regions. The current original model is `PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF`: four modality-specific motion backbones, two pairwise readouts, two temporal refiners and one RGB-anchored phase fusion head. The final model is not a JEPA encoder. Its maximum causal information history is 650 ms; the 1,955-parameter count refers to the fusion head alone.

| Download | Model identity and result scope |
|---|---|
| [PHASE-TTC](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) | Original frozen nine-component system. Archived 28 October [GarlTTC Test rank 1 of 3 entries](https://www.codabench.org/competitions/17289/#/results-tab): Overall MiD 35.80145 versus 45.03, 20.49% reduction, 6,762 scored tokens, 0% scorer-defined failure. |
| [PHASE-TTC-Adapt](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) | Original system plus three 688-parameter geometric quality routers trained on 2,608 development queries. Separate 434-query local evaluation; these weights have no verified official score. |
| [PHASE-TTC-Geo submission record](sota_20261118/SUBMISSION_LEDGER.md) | A different head using RGB-branch estimates and 600 ms of supplied-box history, with no event input. Submission 1058935 obtained 23.4724 Overall MiD and 0% scorer-defined failures on 18 November. Its scored package is not either Hub model repository. |
| [Box-history comparator record](sota_20261118/leaderboard/submission_1063261/RECORD.json) | A matched learned head using only 600 ms of supplied-box history, with no image or event input. Submission 1063261 obtained 27.97563 Overall MiD and 0% scorer-defined failures on 21 November. It is an official comparator, not a downloadable PHASE-TTC release. |
| [Recorded demo](https://huggingface.co/spaces/AlvaroSchwiedop/PHASE-TTC) | Recorded examples and download commands; live inference runs locally with the supplied CLI or Gradio app. |

The adaptation's final fit uses 24 road recordings plus Slider-500. Its ten evaluation recordings are excluded from that fit, but earlier development exposure applies. The local offline negative-output interpolation gives original/adapted MAE 0.82498/0.33590 s and relative error 49.894/17.250%. Slider-750's official original score remains unreconciled. Later router and canonical-cadence experiments did not replace these checkpoints. PHASE-TTC-Geo is a separate fitted head, not a renamed PHASE-TTC-Adapt router.

The official box-only comparator is 4.50325 MiD points above PHASE-TTC-Geo
(27.97563 versus 23.47238); equivalently, RGB plus geometry reduces error by
16.10% relative to box-only under these recipes. Both use the same 6,762-token
server population and have zero scorer-defined failures. The uploads are
sequential exposures to one private test server, not independent test sets. The
third downloaded scoring record is preserved in commit `8fa8dda6`.

The corrected original transfer study uses 460 eligible queries from 508 fixed candidates across 32 development-exposed sequences. Original fusion improves pooled MAE and p95 over RGB, while median and relative error favor RGB. Paired sequence-level confidence intervals include zero.

## Running and reproducing

The Hub bundles contain weights, exact source dependencies, normalizers, manifests, a tested inference command, synthetic and real-query regression references, evaluation projections and training source/recipes. See [reproducibility.md](reproducibility.md). Inputs require calibrated sensor preprocessing; a generic RGB image is insufficient.

## Limitations

The full original system contains 2,347,313 parameters. The expanded same-hardware
study compares persistent PHASE/Garl services over 72 fresh-process passes on six
EvTTC scenes, with continuous intervals of 3.1–9.3 seconds and 50 Hz offered input.
Both retain resident weights and causal state; Garl bypasses auxiliary foreground
decoders during service and is checked against native inference afterward.

Complete delivery, strict numerical admission and response deadlines are separate
results. With the larger queue PHASE delivers all twelve streams, but two fail the
frozen intermediate RGB tolerance at three repeated anchors; their final fused
TTC outputs pass. Garl delivers seven complete streams under that queue policy.
Small-queue losses and all failed checks are retained. Neither parameter counts
nor these runs establish a universal latency bound or continuous 20 ms responses.
See the [expanded inference report](../paper/from_motion_to_time_20261102/revision/extended_inference_report.md)
and the [initial comparison](../paper/from_motion_to_time_20261102/revision/persistent_comparison_report.md).

The later CPU box-only service covers eight host-paced 60-second cells and
24,000 offers. Seven cells pass its strict 20 ms rule; one records five drops,
so the all-cell gate fails despite 23,992 offers completing below 20 ms. This
replay starts from supplied TRAIN40 boxes and excludes detector/tracker cost.
This result covers only the CPU box-history path. See the
[CPU report](sota_20261121/geometry_service/REPORT.md).

The RGB + geometry replay ran while another training job occupied the GPU.
Across eight 60-second cells it returned 1,886 valid responses from 24,000
offers, with 22,114 queue drops and zero responses within 20 ms. The workload
cycled two real instants per object count. Frames were already decoded and
earlier observations cached. This measurement characterizes shared-GPU
operation. [Load conditions and traces](sota_20261121/geometry_service/RGB_STATUS.md)

The release expects supplied object regions and synchronized event/RGB inputs. It includes neither a general-purpose autonomous detector nor calibrated collision probabilities or confidence intervals. The paced 250-update replay measured 29.14 ms p95, above a 20 ms budget. Zero benchmark failures is a scoring result, not a safety assurance. Full original training requires external data and reconstruction of historical caches.

Code and released weights: MIT. External datasets retain their own terms. See the [current paper](../paper/from_motion_to_time_20261102/PHASE_TTC_From_Motion_to_Time_Expanded.pdf) for definitions, controls and complete evidence.

Funding: none. Competing interests: none. The MIT terms cover project-authored code and PHASE-TTC weights; third-party models and datasets retain their providers' terms.

Earlier candidate cards are preserved as [historical records](history/pre_hf_release_20261111/model_card.md); they do not describe the current downloadable model.
