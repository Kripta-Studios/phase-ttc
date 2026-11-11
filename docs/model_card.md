# Model card — PHASE-TTC

Current release: 11 November 2026. Author: Álvaro Schwiedop Souto.

PHASE-TTC estimates signed time-to-contact from causal event/RGB observations and supplied object regions. The current original model is `PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF`: four modality-specific motion backbones, two pairwise readouts, two temporal refiners and one RGB-anchored phase fusion head. The final model is not a JEPA encoder. Its maximum causal information history is 650 ms; the 1,955-parameter count refers to the fusion head alone.

| Download | Model identity and result scope |
|---|---|
| [PHASE-TTC](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) | Original frozen nine-component system. Archived 28 October [GarlTTC Test rank 1](https://www.codabench.org/competitions/17289/#/results-tab) among three displayed entries: Overall MiD 35.80145 versus 45.03, 20.49% reduction, 6,762 scored tokens, 0% scorer-defined failure. |
| [PHASE-TTC-Adapt](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) | Original system plus three 688-parameter geometric quality routers trained on 2,608 development queries. Separate 434-query local evaluation; no verified official adapted score. |
| [Recorded demo](https://huggingface.co/spaces/AlvaroSchwiedop/PHASE-TTC) | Recorded examples and download commands; live inference runs locally with the supplied CLI or Gradio app. |

The adaptation's final fit uses 24 road recordings plus Slider-500. Its ten evaluation recordings are excluded from that fit, but earlier development exposure applies. The local offline negative-output interpolation gives original/adapted MAE 0.82498/0.33590 s and relative error 49.894/17.250%. Slider-750's official original score remains unreconciled. Later router and canonical-cadence experiments did not replace these checkpoints.

The corrected original transfer study uses 460 eligible queries from 508 fixed candidates across 32 development-exposed sequences. Original fusion improves pooled MAE and p95 over RGB, while median and relative error favor RGB. Paired sequence-level confidence intervals include zero.

## Running and reproducing

The Hub bundles contain weights, exact source dependencies, normalizers, manifests, a tested inference command, synthetic and real-query regression references, evaluation projections and training source/recipes. See [reproducibility.md](reproducibility.md). Inputs require calibrated sensor preprocessing; a generic RGB image is insufficient.

## Limitations

The release expects supplied object regions and synchronized event/RGB inputs. It includes neither a general-purpose autonomous detector nor calibrated collision probabilities or confidence intervals. The paced 250-update replay measured 29.14 ms p95, above a 20 ms budget. Zero benchmark failures is a scoring result, not a safety assurance. Full original training requires external data and reconstruction of historical caches.

Code and released weights: MIT. External datasets retain their own terms. See the [current paper](../paper/from_motion_to_time_20261102/PHASE_TTC_From_Motion_to_Time_Expanded.pdf) for definitions, controls and complete evidence.

Earlier candidate cards are preserved as [historical records](history/pre_hf_release_20261111/model_card.md); they do not describe the current downloadable model.
