# Stage70 clock audit and reference integration

The metadata join recovered exactly 8,192 historical development tokens. Both
physical windows, the later-frame label anchor, outer-fold identity and the common
ROI agree with the binding in the verified Stage66–69 results ZIP. Full SHA256
checks of all nine original TRAIN HDF5 files match that binding. A deterministic
32-token probe checks the exact start/end event indices for 64 half-open windows.
All development windows are contiguous and their four context midpoints are
strictly ordered. The audit did not read the fresh TRAIN annotation parquet.

The inspected Garl release builder assigns both the label and its timestamp to
the last selected annotation. This agrees with the existing numeric bindings.
The annotation clock differs substantially from stored event timestamps; token
suffixes are never used as HDF5 times. `stage70_clock.context_clock` subtracts
integers before casting and retains real gaps or overlaps. The source-code
interpretation is documented separately from historical acquisition provenance;
it is not an independent rerun of original annotation acquisition.

Durable receipts are under
`artifacts/stage70_76_architecture/time_charter/`: `LABEL_TIME_CHARTER.json`,
`CONTEXT_CLOCK_ROWS.csv`, `RAW_HASH_PROGRESS.json`, `SOURCE_READS.jsonl` and the
metadata-only 40-sequence inventory. Roles are not assigned by that inventory.

The four numerical modules were ported from the immutable handoff reference:
`phase_distribution_v11`, `projective_geometry_v11`, `projective_phase_sweep`
and `projective_phase_network`. Import paths, type declarations and formatting
were adapted; numerical recipes, parameter shapes and interventions were retained.
The reference practical gates and recursive ancestry check have corresponding
new modules. These are components of the campaign, not a completed trainer or
complete evidence validator.

All 62 reference tests pass against the integrated modules. Eight new clock tests
and eleven input/resource tests also pass on the supported Python 3.11 environment.
Ruff and the expanded audit Pyright scope pass. Full baseline-versus-new test
comparison, raw E128 integration, GPU gradients, technical fits and exact resume
remain mandatory before scientific readiness. No optimizer update has occurred.
