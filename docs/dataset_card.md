# Data used by PHASE-TTC

Current release overview, 11 November 2026. Raw camera recordings are obtained separately under their dataset terms and are not redistributed in the model repositories.

| Population | Use and scope |
|---|---|
| eAP TRAIN40, 88,744 queries in 40 sequences | Original final refit; five group-disjoint rotations produce upstream out-of-fold training features. |
| GarlTTC Test, 6,762 scored tokens | Official original-model evaluation; 5,805 physical queries plus aliases. Organizer targets are private. |
| EvTTC corrected transfer, 508 candidates / 460 eligible in 32 sequences | Positive-TTC transfer diagnostics with frozen predictions and repaired target-source mapping. Development exposure applies. |
| EvTTC adapter fit, 2,608 queries in 25 recordings | 2,531 road queries from 24 recordings plus 77 Slider-500 queries. |
| EvTTC competition comparison, 434 correspondences in ten sequences | Separate local original/adapted evaluation. Final adapter fit excludes these recordings; earlier development exposure applies. |
| Synthetic NPZ example | Deterministic execution fixture only, with no accuracy target. |

The transfer cohort and competition cohort have different query populations, association rules and metrics. The original 468-query transfer count is superseded by 460 eligible queries after source reassociation; historical projections are retained for auditability.

## Obtaining and preparing observations

Use the [EvTTC download page](https://nail-hnu.github.io/EvTTC/download/download/) for raw sequences and calibration. The Hub release includes `extract_example.py` for the benchmark layout, which produces causal prepared observations and supplied bounding-box history without labels. Custom cameras require clock, calibration, crop and event-representation adapters.

`evaluation/corrected_transfer_candidates.csv` contains saved predictions, target associations and exclusion reasons. The adaptation repository also includes derived fit features and group identities; these are not raw sensor recordings or an independently held-out test set. No inference input accepts TTC targets.

See [the paper](../paper/from_motion_to_time_20261102/PHASE_TTC_From_Motion_to_Time_Expanded.pdf) for split construction and [reproducibility.md](reproducibility.md) for reproduction commands. Earlier dataset experiments remain in the [historical card](history/pre_hf_release_20261111/dataset_card.md).
