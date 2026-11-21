---
title: PHASE-TTC
emoji: ⏱️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: Verified model outputs, downloads and reproducible inference
models:
- AlvaroSchwiedop/PHASE-TTC
- AlvaroSchwiedop/PHASE-TTC-Adapt
---

# PHASE-TTC recorded inference explorer

This static Space displays measured outputs from the released model weights. It performs no live inference and accepts no uploads. Select an example and model, inspect the recorded predictions, and copy revision-pinned commands to run inference locally.

The synthetic example is an execution fixture without an accuracy target. The real EvTTC example contains outputs only; obtain raw data separately and use the extraction command to reproduce it.

`gradio_app.zip` contains the tested CPU Gradio app for live local inference. Extract it, install `requirements.txt` and run `python app.py`. Its `models.json` pins the released revisions. Python hosting on Hugging Face was unavailable on this account without PRO, so the public Space uses free static hosting.

[Original model](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) · [Adaptation](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) · [Paper](https://schwiedop.com/papers/phase-ttc/)

[Official GarlTTC leaderboard](https://www.codabench.org/competitions/17289/#/results-tab): original PHASE-TTC was the first of three entries on the archived GarlTTC Test leaderboard (28 October 2026), with **35.80145 Overall MiD**, a **20.49% reduction** from Garl-TTC's 45.03; the score measures the complete system. A pre-registered second submission on 18 November 2026, a different RGB + geometry head without event inputs that is not among these models, scores **23.47238**. A third submission on 21 November, box-only comparator `1063261`, scores **27.97563**, a **4.50325-point gap**. As observed on 18 November, before the third submission, the leaderboard showed one entry per participant and listed PHASE-TTC-Geo. Its display policy after the third submission is unknown, so no current rank is claimed.

Replay from the frozen PHASE20 endpoints reproduces the geometry submission's 5,805 physical predictions and ZIP byte for byte. Refitting its head reproduces every seed-7 prediction, while alternate seeds show unstable tails. A bounded media replay covers 12/5,805 physical queries, one per sequence, but retains the frozen public event tensor; it passes the endpoint gate on 0/12 and the geometry-TTC gate on 5/12, so it is not end-to-end parity. No private targets were read and no new private score was inferred. The [versioned evidence is fixed to public commit `8fa8dda6`](https://github.com/Kripta-Studios/phase-ttc/tree/8fa8dda64bcdf846331191eb4ad4c8171b979699/docs/sota_20261121).

The box-only submission's crucial / small / large / negative bands are **26.38540 / 28.00007 / 32.93790 / 30.89121**, versus **24.36932 / 23.95223 / 22.89423 / 18.12626** for RGB + geometry. Both heads use supplied-region history; only the better one adds RGB-branch estimates. This is a within-account protocol comparison, not a comparison between companies or matched end-to-end sensor systems. Neither head is one of the two downloadable models in this Space.

Projection from the published TRAIN40 annotations reproduced **88,744/88,744 integer boxes** (median pre-truncation IoU 0.9878; p95 maximum edge error 0.987 px). This validates projection from supplied annotations, not an autonomous detector. On **597 development-exposed FCWD queries**, robust geometry plus event content scale records **16.1% relative TTC error**, versus **26.8%** for the public Garl-TTC event model run locally; the protocols are not matched, this is not independent generalization, and this FCWD model is distinct from submissions `1058935` and `1063261`. Isolated p95 compute on 600 TRAIN40 queries is **36.22 ms** for the official RGB + geometry model on GPU, **0.98 ms** for the box-only model on CPU and **83.79 ms** for robust geometry + event content scale on CPU, with prior history cached. Reading, queueing and input production are excluded, so these are not end-to-end latency claims.
