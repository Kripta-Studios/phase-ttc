# S4 event-accumulation CPU experiment

## Outcome

The isolated candidate preserves the frozen S4 representation exactly and was
modestly faster on the preregistered real 50 ms window. It is not adopted by
this experiment: `event_data.py` remains unchanged pending review.

The frozen implementation already uses two weighted `numpy.bincount` calls per
chunk, one for each temporal interpolation endpoint. The candidate combines
both endpoint index/weight vectors and performs one weighted `bincount`. It
still accumulates exact integer common-denominator numerators in `int64` and
normalizes only after all chunks.

## Real profile

Source window: `qGsgzl4Q8B:000001037_000033536`, exactly the admitted v3 probe
window `[1036986031, 1037036031)` microseconds. The HDF5 file size and pinned
expected SHA were reused from that receipt; the 10.2 GB file was not rehashed.

Both methods ran three times in the alternating order
`old,new,new,old,old,new` in one CPU environment with the same open HDF5
reader.

| Measure | Frozen two calls | Single-call candidate |
|---|---:|---:|
| Times, seconds | 2.6993, 2.1319, 1.5263 | 1.8404, 1.9479, 2.1871 |
| Median, seconds | 2.1319 | 1.9479 |
| Median ratio | 1.000× | 1.0945× faster |

The run-to-run spread is material, so this is evidence for a bounded candidate,
not a throughput guarantee. A training-path adoption still requires review and
broader profiling.

Parity and resources:

| Check | Result |
|---|---:|
| Events retained on every run | 2,014,179 |
| Float32 values differing | 0 |
| Maximum absolute float32 difference | 0.0 |
| Raw numerator SHA-256 | `020479dc807e7cd101b7ca5cd1e9301b95aa1559d15d1dcf519c25011a6f99bd` |
| Voxel float32 SHA-256 | `8eaf3fac6175cc389ecb11521b403465b840ac4fdf247320f1ef7149c1d1ba96` |
| Baseline / sampled peak RSS | 605,048,832 / 781,291,520 bytes |
| GPU, labels, test, dataset mutation | None |

The profiler also ran the new module's two-call reference path and obtained the
same raw-integer SHA. Unit tests cover polarities, temporal boundaries, source
coordinate extremes, chunk partitioning, invalid indices and remainders, the
`2**53` weighted-bincount bound, and cross-chunk `int64` overflow.

## Evidence and reproduction

Archived [receipt](evidence/pending_candidates_20261102/event_feasibility/event_accumulation_profile_qGsgzl4Q8B_v1.json), originally
`output/event_feasibility/event_accumulation_profile_qGsgzl4Q8B_v1.json`.

Receipt SHA-256:
`1f2790429bd0b31bb372142679fd0b884d0bd53dd8c6427bf37bd383c1a84d4f`

Bound source hashes:

- Frozen `event_data.py`: `0ee69247ae11ee766c960c3da283eb166ec64f2c99b7c612572820f17df74d55`
- Candidate module: `386a02fdbf7800e76add5d5aa3b2861e276b4139e1335c904b3fe18f5fc4342d`
- Profiler: `08440a9b7cd67d973410d7908b0e4c2560032df0e970890731d0708a080fbb80`

```powershell
$env:PYTHONPATH = 'src'
python scripts/profile_detection_event_accumulation.py `
  --root E:/eAP_dataset `
  --source-probe output/event_feasibility/event_representation_50ms_qGsgzl4Q8B_v3.json `
  --output output/event_feasibility/event_accumulation_profile_qGsgzl4Q8B_v1.json `
  --repeats 3
```

The output path is write-once. The profiler fails unless every run retains the
source event count, both output hashes match, the row-local float32 difference
is zero, and sampled RSS including the HDF5 reader remains below 1 GiB.
