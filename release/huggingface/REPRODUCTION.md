# Reproduction levels

## Run the released weights

Follow the model card. The synthetic execution test and a separately extracted real benchmark query were checked on CPU and CUDA. `VALIDATION.json` records the comparison, environment and tolerances. Expected files are regression references, not labels or measured accuracy on synthetic shapes.

## Recompute saved transfer metrics

```bash
python evaluate_saved.py
```

This recomputes the original model's corrected 460-query transfer summaries from the 508-row projection in `evaluation/` and checks them against the archived statistics. The archived official score is stored separately in `evaluation/official_benchmark.json`. The command does not rerun sensor inference or reconstruct the organizer's private targets. The 434-query adaptation evaluation is a different study; see the paper for its signed/offline protocol.

## Research training source

`reproduction/research_source.zip` contains the source, configuration and available manifests/splits needed to inspect and reconstruct the training pipeline. It excludes raw datasets, environments and credentials. The final original recipe is `model/upstream/components/OOF_RECIPE.json.json`; its five group-disjoint rotations generate out-of-fold features, while inference executes one nine-component graph. Initial event parents and final checkpoints are included in the model bundle.

The original research pipeline used data and intermediate caches outside the repository. Paths in the historical recipe identify those artifacts and must be remapped to your own licensed data. The release is independently runnable for inference; full original training is **not a one-command, freshly validated rerun**. The archive preserves the implementation and recipe without claiming that all historical intermediate caches are distributed.

## Refit the small adaptation

Only PHASE-TTC-Adapt includes `reproduction/inputs/POPULATION.npz` and `exclusions.json`. These are derived training features, targets and group identities, not raw camera recordings. The final fit excludes the listed eight road recordings and Slider-750/1000, producing 2,608 rows across 25 groups. Existing development exposure still applies.

From the model root, after installing requirements:

```bash
python train_router.py --output ../phase_ttc_refit
```

Choose an unused output directory. This uses the fixed three seeds, 300 AdamW updates and inherited quality-router objective. It rebuilds only the router; upstream checkpoints remain frozen. Numerical reproducibility depends on the recorded PyTorch environment. A new fit is not a promoted replacement checkpoint and must be evaluated under the stated protocol.

The release validation reran this fit on CPU: all three state dictionaries matched the distributed router weights exactly, and scripted/eager predictions matched on all 2,608 fit rows. This is a reproduction check, not a new generalization result. The portable trainer writes TorchScript through a byte buffer so Windows paths containing accented characters work.
