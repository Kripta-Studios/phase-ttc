## EvTTC adaptation

This repository releases **PHASE-TTC-Adapt-20261031**. It contains the exact frozen [original nine-component model](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) plus a three-seed ensemble of 688-parameter quality routers. The adapter combines native TTC estimates with causal bounding-box expansion in inverse-TTC coordinates. It returns to the native estimate when suitable geometric history is unavailable.

The routers were fitted on **2,608 development queries**: 2,531 road queries from 24 recordings and 77 Slider-500 queries. The ten evaluation recordings were excluded from that final fit, although the dataset and earlier model development were exposed. Subsequent refits failed their controls and do not replace this frozen release.

On the separate **434-query local EvTTC evaluation**, original/adapted pooled MAE is **0.82498 / 0.33590 s**, and relative absolute error is **49.894 / 17.250%**, under organizer-style offline negative-output interpolation. That postprocessing can use future outputs and is separate from causal inference. Nine sequence scores reconstruct the organizer's rounded original results; Slider-750 remains discrepant. **There is no verified official leaderboard score for this adaptation.** The original GarlTTC score is provided as provenance, not as an adapted-model result.

[Original PHASE-TTC result on the official GarlTTC leaderboard](https://www.codabench.org/competitions/17289/#/results-tab): rank 1, **35.80145 Overall MiD**, in the 28 October 2026 record. This result belongs to the original model.
