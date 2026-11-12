# Same-hardware PHASE-TTC / Garl-TTC inference, 12 November 2026

These are local computational measurements with frozen published weights and
real sensor input. No TTC labels are read, no model is trained, and no new
official score is submitted.

`evttc/` contains all 434 supplied queries across ten recordings, five model
arms and three passes: 6,510 measured attempts. The arms are the original
nine-component PHASE system, full Garl release, full Garl with native decoder
flags disabled, and Garl's separate event-only and RGB-only checkpoints.

`eap/` contains the first ten chronological physical queries per frozen Test12
sequence after alias removal: 120 queries, three arms and three passes, or
1,080 attempts. It compares the native PHASE RAW predictor with observation
reuse against full and decoder-bypassed Garl. Each arm has independent reader
and frame-cache state within a sequence. The nominal Garl TTC conversion stays
at the released model's 0.1 s; the observed sensor-clock intervals are recorded
separately.

## Evidence and scope

- `protocol.json`: pre-run query identities, model and source hashes,
  environment, warmup and measurement boundaries.
- `trials.jsonl`: every measured attempt, timing, output availability and TTC.
  Garl rows bind the actual prepared input bytes. EvTTC also binds PHASE's
  prepared observations.
- `summary.json`: completion state and exact protocol/trial hashes.
- `telemetry.jsonl`: GPU thermal, power, clock, memory and process RSS samples.
- `loading.json`: strict Garl checkpoint/config bindings.
- `profiles.json` (EvTTC): separately measured stored and executed parameter
  counts, isolated CUDA allocated/reserved memory, and profiler-supported FLOPs.
- `input_parity.jsonl` (eAP): bit-exact raw event reconstruction against frozen
  input references before measured calls.
- `executed_source/`: byte-preserving source snapshots with `.py.txt` suffixes.
  These are evidence, not alternate maintained modules.

The main EvTTC arms share a sensor reader. Cyclic order rotates positions but
does not balance every ordered pair. Shared caches and the Windows desktop
remain measurement limitations. eAP arms use separate reader caches; OS file
caches remain uncontrolled. No latency outlier is removed.

The decoder bypass retains the entire full checkpoint in memory. Its executed
parameter count describes modules called by one profiled real query. It is
neither a pruned checkpoint size nor a complete count of the arithmetic needed
by the algorithm. The 1,955-parameter PHASE fusion head is not its full model.

The previous complete eAP media receipt supplies historical full-file hashes;
this run verifies selected event representations and records prepared Garl
input hashes, rather than rehashing all media. Model/source hashes and the
selected query manifests remain public; raw datasets and third-party weights
are not copied into this repository.

## Reproduction

Use a CUDA-enabled Python environment with the project's dependencies, the
released PHASE upstream bundle, the strict Garl inventory/checkpoints and
licensed local datasets. The recorded protocols give the original paths and
software versions. Run the GPU experiments sequentially in fresh directories:

```powershell
python scripts/benchmark_local_comparison.py --data-root ../EvTTC-Benchmark --garl-freeze output/garl_evttc_competition_20261028/RUN_FREEZE.json --phase-bundle output/huggingface/PHASE-TTC/model/upstream --output output/local_comparison_new/evttc --repeats 3 --warmup 10
python scripts/benchmark_local_eap.py --public artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --garl-freeze output/garl_evttc_competition_20261028/RUN_FREEZE.json --phase-bundle output/huggingface/PHASE-TTC/model/upstream --media-freeze output/release_research_20261031/native_raw_latency_v1/FREEZE.json --output output/local_comparison_new/eap --repeats 3 --per-sequence 10
```

These commands execute inference; they do not regenerate the publication from
saved evidence. To regenerate the published tables and distributions without
GPU or dataset access:

```powershell
python paper/from_motion_to_time_20261102/scripts/build_local_comparison.py
```

The generator rejects incomplete populations, altered ledgers, unsupported
profile queries, and failed input/output parity. It reports all attempts,
successful calls, all-model complete cases and separate pairwise complete
cases. The technical supplement explains the measurement boundaries. The
existing 50 Hz paced-service result uses a separate implementation and workload.
