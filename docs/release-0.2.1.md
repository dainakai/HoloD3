# HoloD3 0.2.1: output-preserving CUDA execution

This release reduces repeated MinIP memory operations and depth-model launch overhead while retaining the configured reconstruction planes, model weights, numeric precision, routing rules, and diameter propagation order.

## Changes

- Raw MinIP reconstruction caches the original `T**slice_step`, reuses its frequency and inverse-FFT buffers, and fuses central intensity, clipping, and minimum reduction into one CUDA kernel per plane. Explicit rounded multiplies/addition preserve the reference arithmetic; NaNs and subnormals retain Torch semantics. CPU, complex128, differentiable calls, and missing CUDA-kernel support use the original Torch implementation.
- Production depth inference captures full slice-block model scoring with CUDA Graphs. Each frame/model owns at most one graph, checks initial replay against eager output, and releases its graph when the frame scorer is replaced. New inputs are copied into stable buffers; partial block shapes run eagerly. Failed captures or unequal validation outputs use the same backend eagerly.
- `runtime.depth_cuda_graphs` and `--[no-]depth-cuda-graphs` control replay. Production enables it; portable Torch leaves it disabled. Metrics expose captures, replays, capture time, and fallback count.
- Package, CLI, lockfile, and citation metadata identify version 0.2.1.

## Validation contract

Exact equality is checked within each backend against its unchanged arithmetic and model engine. This is not a claim that Torch FP32 and TensorRT FP16 agree with each other, nor that experimental measurements match independent physical truth.

The CUDA MinIP tests cover all six included frames, both dual-camera phase retrieval and single-Gabor construction, and MinIP strides 1, 2, and 7 over the full 1,024-plane acquisition. Every float32 pixel agrees with the independent pre-0.2.1 reference. Smaller fixtures cover odd padding offsets, one plane, oversized stride, clipping, NaNs, infinities, subnormals, another GPU/current stream, missing CUDA support, complex128, and autograd.

The reproducible depth audit checks 2,437,120 primary/fallback scores across six frames and 2,273 particle rows, plus all columns of the depth, diameter, and final CSVs. Each eager/graph pair uses the same loaded models and compiled engines. The graph tests also cover changed inputs, partial shapes, caller-modified outputs, unavailable capture, and an intentionally incorrect capture.

Both the TensorRT FP16 audit on RTX 4080 SUPER and the Torch FP32 audit on RTX 4060 Ti passed all these exact comparisons. A separate public-CLI production demo completed all 1,024 depth planes for 380 rows, generated CSV and HTML outputs, and recorded two successful captures, 256 replays, and zero graph fallbacks.

Release checks: **124 tests passed** on the GPU host, including the model/pipeline integrations; the CPU-only fast suite passed 97 tests with 21 CUDA skips and 6 integration deselections. Ruff, Git whitespace checks, and offline lockfile validation passed. Both the 0.2.1 wheel and source distribution built successfully.

Run the audit with [the validation command](validation.md#automated-coverage). The first pair synchronously copies all scores to the CPU and is excluded from performance samples. Subsequent pairs include graph warmup/capture for each new frame and alternate execution order.

## Performance scope

Recorded on the included six-frame, 1,024-plane acquisition with Torch 2.12.1+cu130. MinIP uses the existing stride 2 (512 planes), 1,536-square complex64 FFTs, and a 1,024-square output. No grid or precision settings were reduced.

| Stage and hardware | Eager/reference | Optimized | Speed ratio |
| --- | ---: | ---: | ---: |
| MinIP, RTX 4080 SUPER | 104.58 ms/frame | 77.63 ms/frame | 1.347x |
| MinIP, RTX 4060 Ti | 342.26 ms/frame | 243.09 ms/frame | 1.408x |
| Depth + diameter core, TensorRT FP16, RTX 4080 SUPER | 5.198 s/6 frames | 4.853 s/6 frames | 1.071x |
| Depth + diameter core, Torch FP32, RTX 4060 Ti (opt-in graph experiment) | 48.092 s/6 frames | 48.815 s/6 frames | 0.985x |

MinIP values are medians of 18 warm samples per path. The core values are medians of two six-frame passes per path after the separate audit. The Graph runs perform 12 captures and 1,536 replays per six-frame pass; capture costs 0.184–0.191 seconds of the reported total. Peak Torch tensor allocations increase from approximately 613 to 665 MiB; this excludes allocations made directly by TensorRT/CUDA and is not total VRAM usage.

The capture-time and memory figures above describe the TensorRT run. Torch's frame captures cost 1.590–1.602 seconds per six-frame pass and its total was about 1.5% slower with graphs, supporting the decision to leave graphs off in `portable-torch`. Raw samples, exact-score audit counts, and graph/memory metrics are saved for [RTX 4080 SUPER / TensorRT](benchmarks/0.2.1-rtx4080-super.json) and [RTX 4060 Ti / Torch](benchmarks/0.2.1-rtx4060-ti.json).

These are stage measurements, not full CLI speedups. They exclude detector/ROI preprocessing, engine compilation, subprocess startup, and cold MinIP CUDA compilation. A supplied MinIP skips reconstruction entirely. Engine construction can dominate a short production invocation. Results depend on GPU, particle count, frame shapes, backend, and CPU submission load; this small sample does not establish a statistical performance guarantee. The two device audits ran concurrently on separate idle GPUs in the same host, so host-side contention may affect timings.

## Boundaries for follow-up work

The release does not retain winning depth crops for diameter inference: the original diameter pass advances propagation by gaps, and reusing the sequential depth crop changed diameter outputs in the earlier prototype. It also does not enable channels-last, lower reconstruction counts, reduced precision, or FFT callbacks. Such changes need independent output validation before they can replace the current path.

The CUDA kernel and Graph paths remain optional. Custom models and CUDA environments must pass capture validation or continue eagerly. The new audit script supports `--backend torch` and another device index for checking a deployment before enabling graphs there.
