---
title: PHASE-TTC
emoji: ⏱️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: true
license: mit
short_description: Run causal event and RGB time-to-contact models
models:
- AlvaroSchwiedop/PHASE-TTC
- AlvaroSchwiedop/PHASE-TTC-Adapt
---

# PHASE-TTC interactive inference

This Space runs the released PyTorch weights on CPU, using fixed model revisions in `models.json`.
Choose the original nine-component model or the separate EvTTC quality-router adaptation.
The bundled example is synthetic and tests execution, not benchmark accuracy.

You can also upload a prepared NPZ made with the model repository's `extract_example.py`.
Raw camera recordings and ordinary images are not accepted directly; see the model input contract.
Uploaded inputs are processed by the Space; use the local scripts for private data.

[Original model](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) ·
[Adaptation](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt) ·
[Paper](https://schwiedop.com/papers/phase-ttc/)
