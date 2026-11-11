# Reproducing PHASE-TTC

The current downloadable models are [PHASE-TTC](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC) and the separate [PHASE-TTC-Adapt](https://huggingface.co/AlvaroSchwiedop/PHASE-TTC-Adapt). The final original system uses nine components and phase fusion; earlier JEPA candidates are research history.

## Download and execute

Use Python 3.12 in a fresh environment:

```bash
pip install huggingface_hub
hf download AlvaroSchwiedop/PHASE-TTC --local-dir PHASE-TTC
cd PHASE-TTC
pip install -r requirements.txt
python predict.py --input examples/synthetic_looming.npz --verify examples/synthetic_expected.json
python evaluate_saved.py
```

Replace the repository and directory with `PHASE-TTC-Adapt` to run the frozen adapted system. Pin `hf download --revision <commit>` for a repeatable experiment. The scripts check SHA-256 manifests before loading model weights. CPU is the default; CUDA is optional with a matching PyTorch installation.

The synthetic example executes the full selected graph. `evaluate_saved.py` recomputes the 460-query original transfer summaries from saved predictions and corrected public targets. It does not recreate private organizer labels or rerun the full benchmark.

## Real inputs

After obtaining the benchmark data separately:

```bash
python extract_example.py --data-root /path/to/EvTTC-Benchmark --sequence CCRm-low --row 10 --output real_input.npz
python predict.py --input real_input.npz --verify examples/real_CCRm_low_row10_expected.json
```

That command checks the archived query at 520329 microseconds. Other queries omit `--verify`. `INPUT_FORMAT.md` specifies timestamps, event encoding, RGB crops, availability and supplied object regions. The release was checked in a clean Python 3.12 CPU environment and against CUDA; `VALIDATION.json` contains the reference values and tolerances.

## Training and scientific reproduction

The Hub bundles include a research source archive and the original out-of-fold/final-refit recipe. Full original retraining requires licensed datasets, path remapping and reconstruction of intermediate caches; it has not been rerun as part of this packaging task. The adapted bundle includes derived training features, exclusion identities and the fixed three-seed router trainer. Its `REPRODUCTION.md` documents the exact command and boundaries.

Historical seals, checkpoint identities and source bindings retain their bytes. Current cards and `RELEASE.json` distinguish the released model from rejected later experiments. The source snapshot, not a potentially changing branch, defines the bundled implementation.

[Earlier research reproduction notes](history/pre_hf_release_20261111/reproducibility.md) remain available as historical records.
