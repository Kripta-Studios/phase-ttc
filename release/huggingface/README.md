---
license: mit
library_name: pytorch
language:
- en
tags:
- time-to-contact
- event-camera
- rgb
- causal-fusion
- robotics
- custom-code
---

# {{MODEL_NAME}}

PHASE-TTC estimates signed time-to-contact (TTC) from causal event-camera observations and RGB frames. It measures motion independently in each modality, refines those measurements over time, and combines them in a physical phase representation with a correction anchored in RGB.

[Paper and interactive reader](https://schwiedop.com/papers/phase-ttc/) · [Project](https://schwiedop.com/PHASE-TTC/) · [Code](https://github.com/Kripta-Studios/phase-ttc) · [Official leaderboard](https://www.codabench.org/competitions/17289/#/results-tab) · [Explore recorded examples](https://huggingface.co/spaces/AlvaroSchwiedop/PHASE-TTC)

{{VARIANT}}

## Download and run

Use Python 3.12 and a fresh environment. Install the Hugging Face CLI (`pip install huggingface_hub`) if needed, then:

```bash
hf download AlvaroSchwiedop/{{MODEL_NAME}} --local-dir {{MODEL_NAME}}
cd {{MODEL_NAME}}
python -m pip install -r requirements.txt
python predict.py --input examples/synthetic_looming.npz --verify examples/synthetic_expected.json
```

The example executes the supplied weights on synthetic expanding shapes. Its reference output tests software execution; it is not a benchmark accuracy measurement. CPU is the default. For CUDA, install the appropriate PyTorch 2.10.0 / torchvision 0.25.0 wheels and add `--device cuda`.

For a CPU-only installation, install PyTorch first from `https://download.pytorch.org/whl/cpu`. `requirements-cpu-tested.txt` records the versions tested on Windows with Python 3.12. Platform-specific wheels may differ on Linux.

`predict.py` verifies the SHA-256 manifest, loads the custom PyTorch graph and prints TTC in seconds. Download a specific Hub revision with `--revision <commit>` to keep an experiment fixed. This is a custom sensor model; `transformers.AutoModel` is not its loader.

## Try real observations

Obtain the [EvTTC benchmark data](https://nail-hnu.github.io/EvTTC/download/download/) separately. Keep its sequence directory names and calibration files. From this model directory:

```bash
python extract_example.py --data-root /path/to/EvTTC-Benchmark --sequence CCRm-low --row 10 --output real_input.npz
python predict.py --input real_input.npz --verify examples/real_CCRm_low_row10_expected.json
```

The second command checks the recorded query at 520329 microseconds; use the same benchmark sequence and row. Other queries should omit `--verify`. Raw sensor data stays on your machine. Inputs contain pixels, timestamps and supplied object regions, with no TTC labels. See [INPUT_FORMAT.md](INPUT_FORMAT.md) for custom inputs and sensor assumptions.

## What is included

- Exact checkpoints, normalizers, original seals and model recipe in `model/`.
- Runtime dependencies in `source/`, inference and extraction commands, and a synthetic input.
- Expected outputs and validation receipt in `examples/` and `VALIDATION.json`.
- Saved evaluation projections, labels, source identities and a metric recomputation command in `evaluation/`.
- A research source archive and the training/reproduction boundaries described in [REPRODUCTION.md](REPRODUCTION.md).

Historical identifiers and seals preserve their original bytes. Their old paths or status flags describe their creation, not the current release status. `RELEASE.json`, this card and `MANIFEST.json` describe this release. Auxiliary research modules imported by the runtime do not change the selected model.

## Scope and limitations

The model expects synchronized, calibrated event/RGB observations and supplied object regions. An autonomous detector is a separate component. The release exposes signed TTC and diagnostics; it does not provide calibrated collision probabilities or uncertainty intervals. The published paced replay measured 29.14 ms p95 over 250 updates, above a 20 ms budget. That experiment does not demonstrate sustained 50 Hz sensor-to-output operation.

The corrected EvTTC transfer cohort contains 460 eligible queries from 508 fixed candidates across 32 development-exposed sequences. Original fusion improves pooled MAE (0.8130 s versus RGB's 0.8513 s) and p95 error, while RGB has better median and relative error. Sequence-level paired confidence intervals include zero. These transfer data and the separate 434-query adaptation evaluation have different denominators and protocols.

JEPA experiments belong to the research history; the released final model is the phase-fusion system described above. No safety certification or deployment guarantee is implied by a zero benchmark failure rate.

## License and citation

Project code and model release: MIT, see [LICENSE](LICENSE). Obtain external datasets under their own terms; raw datasets are not bundled. [CITATION.cff](CITATION.cff) identifies the software and author. The paper is available from the links above.
