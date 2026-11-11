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

[Official GarlTTC leaderboard](https://www.codabench.org/competitions/17289/#/results-tab): original PHASE-TTC ranked first in the 28 October 2026 record with **35.80145 Overall MiD**, a **20.49% reduction** from Garl-TTC's 45.03.
