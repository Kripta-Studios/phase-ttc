# Prepared sensor input

`predict.py` accepts an NPZ created without pickle. Its `metadata` field is a scalar Unicode JSON string with schema `phase_ttc_prepared_input_v1`, an integer `query_us` and an `observations` list. Optional `boxes_xyxy` and `box_timestamps_us` supply causal geometry for the adapter.

Each observation contains exactly:

```json
{"modality":"rgb","pixels":"rgb_0","timestamps_us":[150000,250000,350000],"first_sensor_ts_us":150000,"anchor_us":350000,"available_us":350000}
```

`pixels` names another NPZ array: float32 `[T,3,128,128]` RGB in [0,1], or `[T,12,128,128]` events, with T=2 or 3. Event channels use the exact voxel/timing encoding in `source/operational/evttc_competition/prepare.py`; they are not twelve ordinary image channels. Use `extract_example.py` for the matching sensor preprocessing. The synthetic fixture is an execution test, not a substitute for real event encoding.

Timestamps are ordered integer microseconds. The anchor is the last endpoint. Availability cannot exceed the query, and the oldest sensor support must be within 650 ms. For RGB, support starts at the first frame; event support includes the accumulation interval before its first endpoint. One unique anchor per modality is allowed. Inference handles incomplete modality histories through its existing fallback logic.

Boxes are finite `[x1,y1,x2,y2]` coordinates with positive extent in a consistent image plane; clocks are strictly ordered and no later than the query. The adapter applies its own history window. No target TTC, depth, velocity or label fields are accepted.

Raw extraction currently supports the EvTTC benchmark directory layout via its preserved reader and calibration adapter. Another camera requires an adapter matching this contract, including calibration, clock origins, ROI crop policy, event channel construction and availability. Feeding an arbitrary resized video into these arrays is not equivalent to that preprocessing.
