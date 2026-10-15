# FFT pre/postprocessing fusion: design and handoff

Status: **proposal, not implemented**. Start from the 0.2.1 implementation commit `d44fdd4` in [PR #2](https://github.com/dainakai/HoloD3/pull/2). Read the [release measurements](release-0.2.1.md) before measuring an additional speedup. The next experiment should target **MinIP store-side fusion first**, with an unchanged Torch fallback and exact-output acceptance gates.

The objective is to remove intermediate transfers and launches around the existing full-field FFT. Keep the acquisition grid, propagation recurrence, complex64 arithmetic, model inputs, and all configured reconstruction planes. Do not replace the padded field with independent ROI FFTs or smaller spatial tiles: their boundary conditions and global propagation are different.

## What is already complete

Version 0.2.1 implements two independently validated changes:

1. [MinIP reconstruction](../holod3/reconstruction.py) caches the original transfer power, updates frequency in place, reuses the inverse-FFT output buffer, and calls [one central-region reduction kernel](../src/common/cuda_minip_kernels.py) per plane.
2. [Frame-bounded CUDA Graph scoring](../src/common/cuda_graph.py) reduces repeated model/preprocessing launches in [the depth pipeline](../src/detection/depth_and_slice_fused.py). Production enables it; portable Torch does not.

The test suite has 124 passing tests. MinIP matches the original float32 output on six real frames, two wavefront modes, and strides 1, 2, and 7. Both TensorRT and Torch audits match every one of 2,437,120 primary/fallback scores and all three CSVs for 2,273 particles. Current measured MinIP medians are 77.63 ms on RTX 4080 SUPER and 243.09 ms on RTX 4060 Ti. The TensorRT depth/diameter core takes 4.853 seconds for six frames, including graph capture. These are the starting points, not gains that a future FFT prototype may claim again.

## Current data flow and numerical contract

Let `P = 1536`, `N = 1024`, `offset = (P-N)//2`, `Z = 1024`, and `s` be the configured MinIP stride (2 in the included acquisition). The central result has `N*N` pixels; the transform always has `P*P` complex values.

```text
wavefront -> mean padding -> FFT2 -> multiply initial transfer -> F0

for each selected plane:
    if this is not the initial plane:
        F = complex_multiply(F, T**gap)    # preserve this recurrence
    u = normalized_IFFT2(F)              # complex64, full padded field
    central = u[offset:offset+N, offset:offset+N]
    MinIP = minimum(MinIP, clamp(real(central)**2 + imag(central)**2, 0, 1))
```

For MinIP, `gap` is the same configured stride throughout. Depth uses the one-based list returned by `slice_values`, which can append a final endpoint with a different gap. Diameter starts again from the base frequency and advances between the sorted winning slices; its gaps differ from the sequential depth scan. Keep each path's exact transfer powers and operation order.

The installed Torch `ifft2` default is `norm="backward"`, so inverse scaling is `1/(P*P)`. A custom backend must match the real and imaginary normalized values **before** squaring. Computing `abs2(raw_IFFT) / (P*P)**2` can round differently. Also do not assume that a mathematically equivalent two-pass row/column FFT is bitwise identical to the selected cuFFT plan.

The crop and MinIP references differ deliberately:

- MinIP's separate Torch squares/addition are reproduced with explicit rounded operations and without fast-math in 0.2.1.
- The existing [CUDA ROI cropper](../src/common/cuda_roi_kernels.py) uses `--use_fast_math`, clamps intensity, computes each ROI's valid-pixel mean with a particular 256-thread reduction tree, and fills out-of-image pixels with that mean. A future crop fusion must preserve **that existing CUDA path**, including its arithmetic and reduction order.
- [Diameter preprocessing](../src/diam/inference.py) rounds each crop to the 8-bit grid before model inference. Small upstream differences can cross a rounding boundary and change the diameter output.

## What can fit in cache

These capacities were queried from the CUDA runtime on the validation host; they are device attributes, not an assumed marketing specification.

| Property | RTX 4080 SUPER | RTX 4060 Ti |
| --- | ---: | ---: |
| L2 cache | 64 MiB | 32 MiB |
| Maximum persisting L2 set-aside | 44 MiB | 22 MiB |
| Shared memory per block, default / opt-in | 48 / 99 KiB | 48 / 99 KiB |
| Shared memory per SM | 100 KiB | 100 KiB |

| Buffer for this acquisition | Size |
| --- | ---: |
| Central float32 intensity or MinIP accumulator | 4 MiB |
| Full padded float32 intensity | 9 MiB |
| One padded complex64 frequency/transfer/field | 18 MiB |
| Three hot buffers: frequency + step transfer + field | 54 MiB |
| Eight-plane crop block for 380 particles at 64 x 64 float32 | 47.5 MiB |
| Winning crops for 380 particles | 5.94 MiB |
| All 1,024 central float32 slices | 4 GiB |

The 54 MiB figure is only a minimal hot working set. MinIP also holds the accumulator and original transfer tensors; with distinct initial transfer, slice transfer, cached step power, frequency, and field, those five complex arrays alone total 90 MiB. FFT workspace, the wavefront, and other allocations add more. Depth also retains the base frequency, crops, and model buffers. Not all resident arrays are touched every plane, but it is incorrect to infer full residency from `18 MiB < L2`.

A Torch CUDA tensor still has a device-memory address when its accesses hit L2. The useful question is which reads/writes can be removed or served from L2, not whether an ordinary allocation can be permanently moved into cache. A central slice is small enough to benefit from either device's L2, while the padded field is much larger than one block's shared memory.

CUDA's access-policy window gives selected global accesses preferential L2 retention; it is not a residency guarantee. Limits and simultaneous users matter, and persistence should be reset after use. See the [CUDA 13.0 L2 access-management guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#device-memory-l2-access-management). Profile L2 and DRAM separately before adding a policy. Pinning a transfer array can evict FFT workspace or crops, and must be compared against normal caching.

## Library facts that constrain the prototype

The [CUDA 13.0 cuFFT callback specification](https://docs.nvidia.com/cuda/archive/13.0.0/cufft/index.html#cufft-callback-routines) provides the relevant contract:

- Set LTO callbacks after `cufftCreate` and before plan generation. NVRTC/nvJitLink must satisfy its toolkit-version compatibility rules.
- Dimensions must factor into primes smaller than 127; `1536 = 3 * 2**9` qualifies.
- C2C LTO loads and stores each call the respective callback once per element. Block execution order is unspecified.
- cuFFT can use the output array for intermediate stages, so it must remain large enough for the complete transform, even when a callback redirects final writes.
- LTO callbacks support the relevant single-GPU graph use; do not generalize that to deprecated legacy callback combinations or multi-GPU transforms.

**Retain the full output allocation required by the plan.** Store fusion may eliminate its *final materialized field* and subsequent consumer reads, but does not establish that cuFFT's output/workspace allocations or internal memory passes can be removed.

## Candidate A: store the MinIP directly from the IFFT

This is the first candidate because there is one destination value per spatial pixel and one independent plane execution at a time. Keep the existing Torch transfer recurrence initially, isolating output fusion from frequency-update changes.

Proposed sequence:

1. First build an ordinary custom C2C inverse plan with no callback. Match the current normalized complex output exactly on representative inputs. If this already fails, stop before integrating a reduction.
2. Add a store callback that only reproduces normalization into a full complex result. Verify equality again; changing the plan can change FFT arithmetic.
3. Change that callback to normalize in registers, select the central region, reproduce MinIP intensity/clipping/NaN behavior, and update a separate float32 accumulator. Retain a full-sized output buffer and the plan's required workspace even if final complex values are not materialized.
4. Execute all selected planes sequentially on the caller's CUDA stream. Initialize the accumulator exactly as in 0.2.1, then reset it for every frame.

Conceptual callback, **not compilable API code**:

```text
store(logical_spatial_offset, raw_complex_value, parameters):
    row, col = decode_offset(logical_spatial_offset)
    if inside_central_region(row, col):
        value = normalize_complex_like_torch(raw_complex_value)
        intensity = add_round_nearest(square_rn(value.real), square_rn(value.imag))
        clipped = clamp_with_torch_nan_semantics(intensity, 0, 1)
        accumulator[row-offset, col-offset] = torch_style_minimum(
            accumulator[row-offset, col-offset], clipped)
```

Each central output has one writer in a plane; sequential execution orders updates across planes. No float atomic minimum is needed in this design. Batched planes or concurrent streams sharing the accumulator invalidate that argument. A later batched experiment needs separate plane results or partial accumulators and a defined final reduction; do not silently add atomics or reorder NaN handling.

The offset must be interpreted for the actual plan layout. Start with contiguous square C2C, one transform per execution, and explicit center coordinates. Generalize strides/batches only after testing their index mapping.

### Traffic estimate, not a measured speedup

At the array-access level, omitting one final complex field store (18 MiB) and the central complex reread (8 MiB) removes up to 26 MiB per selected plane. Across 512 planes this is 13 GiB of **logical** traffic. This is not necessarily 13 GiB of DRAM traffic: cache hits, internal FFT passes, normalization implementation, and callback effects all matter.

If tracing confirms a separate normalization pass, its traffic is an additional candidate; do not count it without verifying the installed implementation or count the same output store twice. At hypothetical effective bandwidths of 0.5–1.0 TB/s, 13 GiB corresponds to roughly 28–14 ms. That calculation assumes all those accesses reach that memory level and all other costs are unchanged. It is a screening model, not a prediction for either measured GPU.

For the current 77.63 ms RTX 4080 SUPER MinIP, sensitivity to an actually removable fraction is:

| Fraction of current wall time removed | New time | Speed ratio |
| --- | ---: | ---: |
| 5% | 73.75 ms | 1.053x |
| 10% | 69.87 ms | 1.111x |
| 20% | 62.10 ms | 1.250x |

Use a 5–20% reduction as an initial experiment target, not a committed benefit. Zero gain or regression remains possible if the callback changes FFT kernels, increases register pressure, or competes with already-effective L2 reuse. Benchmark cold plan/JIT cost separately and amortize it over real frame counts.

## Candidate B: fuse the propagation multiply into the IFFT load

Only investigate after the store-only path passes. Replacing the recurrence with `F0 * exp(i*k*phase)` or `F0 * T**k` changes floating-point history and is outside the exact-output contract.

A conservative design preserves the recurrence with two frequency buffers:

```text
load(i):
    next_value = complex_multiply_like_torch(F_current[i], transfer_for_this_gap[i])
    F_next[i] = next_value
    return next_value

execute inverse FFT using this load callback
swap(F_current, F_next) after the stream-ordered execution
```

Each input index reads only its own previous frequency and coefficient and writes the same index of a distinct next buffer. The next plane reads that buffer after the previous operation. Do not rely on cross-block order, modify unrelated input elements, or reuse an output/workspace buffer as `F_next`. Start with a store-only plan for the initial plane; do not create its frequency by dividing by a transfer coefficient merely to avoid a special case.

This merges the separate multiply's input reads with the FFT's first input load, but the updated frequency must still be written for the next plane. A simple accounting removes one 18 MiB frequency reread per advancing plane; it does not eliminate every multiply read/write. The second frequency buffer adds 18 MiB of live storage and may reduce cache locality enough to offset the saving. An in-place callback update is a later experiment requiring an explicit aliasing proof and independent tests.

Determine the precise complex multiplication behavior from Torch and compare frequency arrays after each step. Do not blindly copy the MinIP kernel's FMA settings: its separate-square reference is different from Torch's complex multiplication. Keep `T**gap` results produced by the existing code, including a nonuniform last depth gap and diameter's selected-slice gaps. Returning a value with the correct formula is insufficient if its rounding differs.

## Candidate C: fuse depth/diameter intensity production

The existing ROI kernel already fuses abs2, clipping, ROI extraction, and mean padding. Replacing it with a store callback that simply writes one pixel to one crop is incomplete: ROIs overlap, may cross image boundaries, and need per-ROI means.

The lower-complexity first design is:

1. Preserve the full-field IFFT, but have its store callback produce a central float32 intensity map with the **existing ROI kernel's** intensity arithmetic.
2. Change a separate experimental crop kernel to read that map while keeping the original ROI iteration, summation tree, counts, and mean-fill order.
3. Compare every crop element before passing it to the unchanged score function and diameter quantizer.

This adds/retains a 4 MiB intensity map but can reduce complex-value rereads. It computes central intensity even at pixels outside every ROI, so sparse particle counts and the selected-slice diameter phase may behave differently from a dense depth scan. Measure them separately. Do not assume the full-depth speedup applies to diameter, where many reconstructed planes contain only one or a few requested crops.

A more complex alternative builds a pixel-to-ROI destination list once per frame and scatters final intensities directly into all matching crops. That adds indexing storage, potentially irregular writes, and a separate canonical mean-padding reduction. Prove that every crop destination has one producer; retain the original mean-reduction order. Avoid unordered atomic summation. Benchmark the map-building cost and memory, including frames with very few particles, many overlapping particles, and boundary ROIs. This is a later candidate, not the first integration target.

## Candidate D: cuFFTDx or a persistent kernel

cuFFTDx exposes FFT operations inside custom kernels, making deeper fusion conceivable. However, its [large 2D examples](https://docs.nvidia.com/cuda/cufftdx/examples.html#d-3d-fft-advanced-examples) use global exchange when data exceeds shared memory; the single-kernel variant uses cooperative grid synchronization. A single launch does not imply that a 1,536-square transform remains in registers/shared memory.

A row/column design might fuse a transfer multiply into the first pass and intensity into the last pass. It still needs inter-pass exchange, a valid grid-wide schedule, and hardware-specific resource tuning. Different factorization and butterfly order can change numerical results even with float32 everywhere. Given the small measured share of the depth core spent in FFT, prioritize callbacks and profiling before a custom FFT implementation. Do not use mixed precision, smaller padding, or independent local FFT tiles to make its memory footprint fit.

## How much can the whole pipeline gain?

A [historical pre-0.2.1 profile](benchmarks/pre-0.2.1-fft-profile.json) on RTX 4080 SUPER used frame 005302, 380 fixed ROIs, 1,024 depth planes, and 303 selected diameter planes. The second instrumented run recorded:

- summed CUDA-event intervals around `ifft2`: 90.50 ms;
- ROI crop intervals: 20.77 ms;
- depth preprocessing/model batches: 530.60 ms;
- separate uninstrumented eager core median: 907.33 ms.

These are diagnostic intervals, not a complete partition: instrumentation and launch gaps can affect them, and frequency multiplication was not isolated. The inverse-FFT wrapper covers the depth and selected-diameter passes; it does not cover the separate wavefront/phase-retrieval FFTs. The measurements precede the new frame-bounded graph implementation. Reprofile 0.2.1 before claiming an incremental gain.

Holding all other costs fixed, the historical FFT-only Amdahl model is

```text
new_core = 907.33 ms - 90.50 ms * (1 - 1 / FFT_speedup)
whole_core_speedup = 907.33 ms / new_core
```

| Hypothetical FFT improvement | Whole-core speed ratio |
| --- | ---: |
| 1.25x | 1.020x |
| 1.5x | 1.034x |
| 2x | 1.052x |
| Profiled inverse-FFT calls take zero time | 1.111x |

Eliminating both measured FFT and crop intervals would give about 1.140x in that model, an unattainable zero-time bound rather than a forecast. These bounds apply to the specified components; they do not bound separately unmeasured transfer multiplication, phase-retrieval FFTs, or CPU overhead. MinIP fusion can have a larger *stage* benefit, but the detector may use an already-supplied MinIP, in which case that benefit is absent. Account for engine compilation and I/O when discussing complete CLI latency.

The useful near-term expectation for the depth/diameter core is therefore a few percent if only FFT time improves substantially, subject to a new profile. A 2x overall inference claim is unsupported by these measurements.

## Prototype integration and environment

At handoff the local environment has Torch 2.12.1+cu130, cuda-python 13.3.1, nvmath-python 1.0.0, nvidia-cufft 12.0.0.61, nvJitLink 13.0.88, and Torch-TensorRT 2.12.1. CUDA toolkit and cuFFT library version numbers are different; inspect the actually loaded library when resuming. `nvcc` and Nsight Compute (`ncu`) were not on PATH. No compiler/profiler was installed for this task.

The installed nvmath sources expose FFT `prolog`/`epilog` and `nvmath.bindings.cufft.xt_set_jit_callback`; the current HoloD3 `fft2`/`ifft2` wrapper accepts only `torch`. A separate experimental adapter using the existing bindings/NVRTC is a reasonable first route. Verify that it exposes the required layout, normalization, output pointer, stream, and caller-data lifetime; a high-level wrapper may impose extra allocations. Prefer the environment-matched CUDA 13.0 specification over assumptions from newer toolkit documentation.

Proposed ownership boundaries for future code:

| Proposed component | Responsibility |
| --- | --- |
| Experimental `src/common/cuda_fft_callbacks.py` | Runtime compilation, callback symbols, plan/workspace lifecycle, stream/device handling, bounded plan reuse |
| `holod3/reconstruction.py` | Optional MinIP backend selection and original fallback; keep acquisition semantics here |
| `src/detection/depth_runtime.py` and `depth_and_slice_fused.py` | Later depth integration, explicit gap/route handling, unchanged model invocation |
| A new FFT-fusion test module and audit script | Complex-field, crop, score, CSV equality; cold/warm performance and memory evidence |

Keep buffers alive until their work completes. Use Torch's current device and stream, avoid host copies of the field, and handle producer/consumer stream dependencies explicitly. Plans executing concurrently need independent workspace; do not share a mutable callback parameter block between in-flight frames. Bound caches by device, transform shape/layout/type, callback variant, and supported library version rather than retaining a plan per frame forever. Reuse stable device parameter storage only with ordered updates and an explicit lifecycle.

Compile and warm plans before capturing a graph. First validate each callback without graph capture, then add capture to avoid mixing two sources of failure. If using a capability/equality check, retain an actionable diagnostic and the same original Torch FFT path; a successful API call alone does not establish output equivalence.

Prefer choosing a fallback before starting a frame. If a recoverable failure occurs after frequency/accumulator updates, rebuild state from the original wavefront before restarting that frame. A corrupted CUDA context is a surfaced failure, not a safe opportunity to continue with partially updated buffers.

## Acceptance gates and experiment order

1. **Freeze the 0.2.1 baseline.** Record commit, checkpoint hashes, GPU/driver/library versions, acquisition hash, grid, precision, routing, and existing graph policy. Keep model engines identical within each comparison. Reserve an idle GPU and avoid simultaneous host/GPU workloads for timing.
2. **Reprofile one complete frame.** Separate transfer-power setup, frequency multiply, inverse FFT, normalization, MinIP reduction, ROI extraction, score preprocessing/model, and diameter work. Record CPU wall time and CUDA-event time. If Nsight becomes available, inspect DRAM/L2 bytes, hit rates, launch count, occupancy, and register spills; do not infer DRAM savings from allocation size.
3. **Prove a plain custom inverse FFT matches.** Test the normalized complex tensor on small deterministic fixtures and full 1,536-square real acquisition spectra. Compare both real and imaginary values with zero tolerance. Report max error and differing-element count for diagnosis, but do not relax the acceptance condition silently.
4. **Implement store-only MinIP fusion.** Compare every float32 MinIP pixel before image conversion on all six frames, both modes, and strides 1/2/7; include one plane, nonuniform/odd padding, clipping, NaNs/subnormals, and custom streams. Confirm the saved detector input images also match.
5. **Measure the isolated stage.** Compare with 0.2.1's fused MinIP kernel, not just the old Torch implementation. Warm plans/kernels, alternate A/B order, collect repeated samples, synchronize consistently, and include reset/parameter-update overhead. Report cold first frame, steady frame, and peak memory separately.
6. **Evaluate load fusion and then depth crops independently.** Check recurrent frequency after every step; check every normalized complex/crop element, including ROI overlaps and border means; then audit all scores, winning slices, uncertainty, routes, and final CSV values. A small error that leaves one frame's winning slice unchanged is not sufficient.
7. **Integrate only successful variants.** Keep unsupported devices/shapes on the original path, test graph/callback failures and cleanup, and rerun full pipeline checks. Report a regression as a result; do not compensate by reducing planes, padding, precision, or model work.

Useful current commands, run from the repository root after installing/fetching dependencies:

```bash
uv run pytest tests/test_cuda_optimizations.py -q
uv run python -m scripts.validate_cuda_optimizations \
  --run-dir runs/fft-baseline-trt --backend tensorrt --device 0 --repeats 3
uv run python -m scripts.validate_cuda_optimizations \
  --run-dir runs/fft-baseline-torch --backend torch --device 1 --repeats 3
uv run pytest -q
uv run ruff check .
```

Run the two device commands sequentially for cleaner timings. Output directories must be new. For acquisitions with background removal, use the prepared corrected acquisition when running the audit. The existing script measures the released paths; extend a separate experiment to select a candidate and compare intermediate FFT outputs before proposing production integration.

## Paths that should not be revived without new evidence

- **Winning depth-crop reuse for diameter:** an earlier one-frame prototype kept depth scores/slices but changed final diameters by up to about 0.0724 micrometres, because the sequential and gap-propagated fields differ before 8-bit quantization. It is excluded from 0.2.1.
- **Channels-last:** the earlier one-frame prototype changed 4 of 380 selected slices. It is excluded from this exact-output optimization scope.
- **L2 persistence by capacity alone:** a previous MinIP hint experiment did not improve its fused baseline. Obtain cache/traffic evidence before revisiting it.
- **New FFT order or precision justified only by small relative error:** retain the exact reference path unless a separate scientific acceptance criterion is explicitly agreed.

The first next-thread deliverable should be a plain-plan equality result and a store-only MinIP prototype with measured cold/warm costs. This design does not authorize claiming that an unimplemented callback is faster or output-equivalent.
