# Stage71 deterministic CUDA corrections before freeze

Both failures below occurred with generated inputs, Python 3.11.15,
PyTorch 2.11.0+cu128 and the local RTX 5070 Ti Laptop GPU. No optimizer
update, target-dependent selection or scientific candidate was involved.

1. CUDA `grid_sampler_2d_backward_cuda` rejects
   `torch.use_deterministic_algorithms(True)`. The single-channel correlation
   sampler now uses four bilinear gathers with zero padding and the original
   `align_corners=False` coordinate convention. CPU/CUDA value and input/grid
   gradient parity are tested against the reference sampler. Repeated backward
   results are bitwise identical in deterministic mode.
2. Full-resolution DIRECT backward then exposed the same PyTorch limitation for
   `adaptive_avg_pool2d_backward_cuda`. The adaptive 2x2 pool now uses explicit
   quadrant means with the same floor/ceiling interval definitions. Even and odd
   dimensions, values and input gradients are tested against the reference pool.

These are two concrete pre-freeze engineering corrections. The phase grid,
encoder/head parameterization, physical operator, training objective and scientific
arm identities did not change. There are still 2,037,822 parameters. Further
technical acceptance failures must be reported against the campaign's correction
limit rather than becoming an unbounded architecture search.

Sweep activation recomputation is an explicitly permitted execution setting.
The optional `checkpoint_sweep` flag preserves forward values and all tested
feature/support gradients bitwise. It bounds the interpolation activation memory;
the selected execution configuration must be bound in the eventual training lock.

The full 128x128 generated-input profile passes for microbatch 2 in strict
deterministic FP32 mode, TF32 disabled, with recomputation enabled:

| Arm | Peak allocated GPU bytes | One forward/backward, seconds |
| --- | ---: | ---: |
| DIRECT | 257084416 | 0.3050183000013931 |
| PPS-TRAJ | 681277440 | 1.1263865000000806 |

These are initial generated-input measurements, not steady-state throughput or
real TRAIN acceptance. The receipt and executed profiler are under
`artifacts/stage70_76_architecture/preflight/GENERATED_GPU_PROFILE.json` and
`generated_gpu_profile.py`. The 13 kernel tests and 62 integrated reference tests
pass; `DETERMINISTIC_KERNELS_PARITY.xml` records their failure IDs (none).
The required real TRAIN gradient, learnability and continuous/resumed fit tests
remain pending.
