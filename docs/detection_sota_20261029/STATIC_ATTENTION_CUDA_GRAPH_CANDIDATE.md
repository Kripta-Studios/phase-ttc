# Static attention and CUDA Graph evaluation candidate

Status: isolated CPU-tested B1 evaluation candidate. No real model was instantiated, no GPU was
used, no frozen source was edited, and no latency or parity claim is made.

## Implemented increment

`static_attention_candidate.py` provides an immutable layout keyed by feature shapes, batch size,
device, value dtype and index dtype. A copied-source patch replaces only:

1. per-forward construction of `spatial_shapes` and `level_start_index` in the 2D transformer with
   cached tensors after a host-only assertion over freshly observed Python feature shapes; and
2. the portable attention `apply` body with the same `grid_sample`, weighting, level order and sum
   order, using precomputed Python split sizes.

Sampling-location finiteness and runtime layout equality remain checked. Their device flags are
combined into one host decision after the complete model forward. Failure reruns ordered slow-path
checks and retains actionable errors. Model outputs then pass through the existing fused validator,
which still visits every tensor leaf and uses the historical recursive validator on failure.

The wrapper is evaluation-only and fixed to one preregistered batch/layout. B2, another geometry,
dtype, device, level count or index layout fails closed. It has no eager fallback. CPU fixtures show
exact forward tensors and exact gradients against the existing portable attention arithmetic, one
host decision across repeated attention calls, nonfinite and layout failures, context cleanup and
fused output validation. This does not establish CUDA or full-model parity.

`static_eval_source_patch.py` verifies the frozen runtime preregistration, declared code manifest,
complete portable tree, exact two input-source hashes, checkpoint and candidate-module hashes. It
materializes a new tree and reviewable patch; the original producer is never modified. The manifest
marks state-dict module names unchanged and both full-model GPU parity and CUDA Graph profiling
pending.

## Remaining CUDA Graph blockers in the inspected code

This increment removes host decisions inside portable attention, but it is not itself a CUDA Graph
replay harness:

- `StaticAttentionForwardGuard` uses Python context and accumulates flag tensors while an eager
  forward executes. Raw graph replay does not rerun that Python. A capture harness needs persistent
  flag/output buffers and must validate them after every replay, outside capture.
- The detector still constructs masks, valid ratios, positional tensors, flattened lists and output
  dictionaries on each eager call. Their shapes are fixed at B1, but graph capture requires stable
  input and intermediate addresses or allocator capture that is demonstrated rather than assumed.
- The outer fused finite decision performs a host `.item()` and must remain after replay. Decode,
  NumPy calibration operations, metrics and reader work also remain outside capture.
- Portable attention uses CUDA `grid_sample`. Capture support and stable workspace behavior for the
  installed PyTorch 2.11/CUDA 12.8 stack have not been exercised.
- The evaluation tail is safe only because this candidate is B1. Any later B4 design needs separate
  fixed buffers and an explicit tail policy; the rejected numerical B2/B4 result cannot be reused.
- Training capture remains out of scope because variable targets, CPU/SciPy matching, backward and
  optimizer state introduce different control flow and synchronization.

## Required GPU gates after the active campaign

1. Materialize the new source tree from the exact frozen runtime and checkpoint; independently
   review its manifest and changed-file set.
2. On four fixed valid frames, compare historical B1 with candidate B1 for every nested output,
   velocity and decoded prediction. Required maximum error is zero; state hashes must remain
   unchanged.
3. Profile host synchronizations and forward time without CUDA Graphs. A reduction in Python-stack
   samples is diagnostic only and cannot establish wall-time gain.
4. Only after gates 1-3 pass, build a separate fixed-buffer graph smoke profile. Compare eager
   candidate versus replay over a fixed 16/32-token panel, record capture time, replay time, graph
   breaks, peak VRAM and post-replay finite checks. Any mismatch rejects capture.

No runner integration, AP claim or speedup claim is authorized by the CPU evidence.
