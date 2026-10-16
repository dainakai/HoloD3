# Validation

## Automated coverage

The test suite covers:

- acquisition schema, path resolution, stem synchronization, selection, and mode constraints;
- built-in and file/module custom transform contracts;
- dual-camera phase retrieval and single-Gabor wavefront construction on CPU fixtures;
- MinIP preparation and image-shape validation;
- exact CUDA/Torch MinIP equality, including NaNs, subnormal values, odd padding offsets, slice strides, autograd fallback, and non-default device streams;
- CUDA Graph input replay, shape bounds, capture/equality failure fallback, and packaged depth-model score equality;
- TensorRT input/conditioning reuse across batches, partial blocks, frame resets, and CUDA Graph replay;
- normalized complex64 IFFT equality, special values, stream/alias guards, preflight fallback, and exceptional cleanup;
- winning-crop retention across routing, ties, invalid scores, non-unit depth steps, and recenter/per-depth-batch fallback;
- semantic inference/model configuration;
- pipeline command construction and model overrides;
- optional fallback rules and physical coordinate scaling;
- animated 3D HTML generation;
- private model and dataset manifest integrity;
- archive checksum, traversal, link, required-path, and atomic-install behavior;
- Web detector/full-pipeline jobs and protected result downloads;
- complete training-ledger safety and dry-run behavior; and
- repository English/privacy/naming rules.

Run the fast suite:

```bash
uv run pytest -m "not integration and not slow"
uv run ruff check .
```

Run model/GPU integration checks after fetching checkpoints:

```bash
uv run holod3 fetch-models --scope production
uv run pytest -m integration
```

Audit every depth score, compare all depth/diameter/final CSV columns, and benchmark the CUDA optimizations with identical compiled engines:

```bash
uv sync --frozen --extra gpu
uv run python -m scripts.validate_cuda_optimizations \
  --run-dir runs/cuda-validation --backend tensorrt --device 0 --repeats 3
```

This requires a new output directory and runs all six demo frames at the full 1,024-plane depth grid. The first eager/graph pair copies every score to the CPU for an exact audit and is excluded from timing. Later pairs alternate execution order, include per-frame graph capture, and reuse the same loaded models and TensorRT engines. MinIP timing is separate and includes all configured MinIP planes; its first FFT/kernel setup is excluded. `results.json` records raw timing samples, exact-comparison counts, graph statistics, and peak Torch tensor memory. These core timings exclude engine compilation, detector/ROI preprocessing, and subprocess startup; they are not end-to-end CLI speedups.

## Version 0.2.2 TensorRT validation

The release audit compares the unmodified v0.2.1 core with v0.2.2 using the same loaded checkpoints and compiled TensorRT engines within each GPU campaign. It covers all six experimental frames, 2,273 particles, and all 1,024 depth planes. The variants disable all new options, enable input reuse plus IFFT normalization, enable only winning-crop reuse, or enable all three options.

To reproduce from a Git checkout containing the v0.2.1 tag:

```bash
uv sync --frozen --extra gpu
uv run holod3 infer \
  --acquisition data/demo/experimental/acquisition.yaml \
  --run-dir runs/reuse-inputs --preset production --limit 0 \
  --stop-after-preprocessing
uv run python -m scripts.validate_tensorrt_reuse \
  --input runs/reuse-inputs/_work/depth_diam/tracks_with_diameter.csv \
  --run-dir runs/reuse-validation --device 0 --repeats 5
```

Use new output directories and an idle GPU host. The audit refuses other compute workloads before and after each run; do not run GPU campaigns concurrently. It records source/checkpoint/input hashes and raw results in `results.json`. The first run of each variant audits every depth score and diameter crop; active normalized-IFFT variants additionally compare every reconstructed field with Torch. These instrumented runs are excluded from timing. Five measured repeats rotate/reverse variant order and check every CSV against that variant's audited output.

Core elapsed time includes hologram I/O, phase retrieval, per-frame CUDA Graph capture, and per-run IFFT-plan preflight. It excludes model compilation, detector/ROI preprocessing, MinIP generation, and process startup. These are core-stage measurements, not complete CLI latency or model-accuracy measurements.

Median elapsed seconds for six frames (five measured runs per variant; percentages are shorter elapsed time versus v0.2.1):

| Variant | RTX 4080 SUPER | RTX 4060 Ti |
| --- | ---: | ---: |
| v0.2.1 reference | 4.5184 s | 13.5525 s |
| TensorRT input reuse + normalized IFFT | 4.3471 s (3.79%) | 13.3143 s (1.76%) |
| Winning-crop reuse only | 3.8613 s (14.54%) | 12.7346 s (6.04%) |
| v0.2.2: both changes | 3.7091 s (17.91%) | 12.5502 s (7.40%) |

The combined path ranged from 3.6880–3.7139 s on the RTX 4080 SUPER and 12.5296–12.5692 s on the RTX 4060 Ti. IFFT-plan setup/preflight after the audit warm-up took about 22–24 ms per combined run. Cold model/FFT compilation and short-run CLI latency must be assessed separately.

On both GPUs, disabling the new options reproduced all three v0.2.1 CSVs exactly. Input reuse plus normalized IFFT also preserved every audited score, diameter crop, and CSV value. Each candidate retained the same 2,437,120 primary/fallback scores and selected depths. The combined path matched the crop-reuse-only output exactly, and all measured repetitions matched their own audited output. CUDA Graphs captured 12 times and replayed 1,536 times per run, with zero fallbacks.

Winning-crop reuse caused the following final-diameter differences against v0.2.1:

| GPU | Changed particles / total | Maximum absolute difference | RMSE |
| --- | ---: | ---: | ---: |
| RTX 4080 SUPER | 308 / 2,273 | 0.4500 µm | 0.02319 µm |
| RTX 4060 Ti | 327 / 2,273 | 0.4500 µm | 0.02344 µm |

The raw crop maximum differences were `3.04e-6` and `3.90e-6`, respectively. Of 9,310,208 pixels, 1,678 / 1,880 changed by one level after eight-bit quantization; no pixel changed by more than one level. All non-diameter CSV columns, including depth and fallback-source decisions, were unchanged. The normalized-IFFT path matched Torch on every reconstructed field (6,144 depth fields per GPU with crop reuse). Frequency propagation and the phase-retrieval feedback loop were unchanged; diameter crops are terminal consumers and cannot introduce new error into those recurrences. This audit does not establish a universal diameter tolerance for other inputs or models.

The recorded environment used Torch `2.12.1+cu130`, Torch-TensorRT `2.12.1`, TensorRT `10.16.1.11`, nvmath-python `1.0.0`, CUDA Python `13.3.1`, and cuFFT `12.0.0.61`. Compiled engines were shared between variants within each GPU; equality across separately compiled engines or GPU models is not implied.

Release checks passed: 153 tests with CUDA available; 117 passed and 36 GPU-only skips with CUDA disabled; Ruff; offline lock consistency; and wheel/source-distribution builds. The public `holod3 demo --preset production --limit 1` command also produced 380 particle rows, CSV/HTML/provenance outputs, and active TensorRT/reuse/normalized-IFFT paths, with two graph captures and no graph or FFT fallback.

## Experimental demo smoke result

The included experimental acquisition completed one full frame with the packaged models and `portable-torch` backends:

- selected frame: 1 of 6;
- reconstruction planes: 1,024;
- final particle rows: 380;
- primary depth routes: 365;
- fallback depth routes: 15;
- total elapsed time in the recorded environment: about 21.36 seconds;
- fused depth/diameter stage: about 4.99 seconds; and
- required outputs: `particles_3d.csv`, `pipeline_summary.json`, and `particles_3d.html`.

This verifies execution and output contracts, not measurement accuracy; the experimental demo has no independent per-particle physical truth.

## Private bundle verification

All six archives were independently downloaded from their pinned private dataset revision into an empty asset root, checksum-verified, safely extracted, and loaded through production training/evaluation loaders:

- 149,323 installed files;
- 104,172 primary-depth rows / 5,200 particles;
- 30,600 fallback-depth rows / 1,600 particles;
- 14,080 unique diameter crops; and
- 12 synchronized benchmark frames.

The extracted text corpus was scanned for source-workstation paths, host/user/storage identifiers, opaque workspace labels, and camera-directory codes; no matches were found.

## Pinned evaluation-bundle smoke result

One independently downloaded benchmark frame completed with the packaged models and the `portable-torch` preset:

- detector rows / final rows: 485 / 485;
- primary / fallback depth routes: 396 / 89;
- depth crops scored: 496,640 across 1,024 reconstruction planes;
- matched truth particles: 182 of 200 (`0.91` recall at the configured center gate);
- matched-particle depth MAE: `96.86 µm`; and
- matched-particle final-diameter MAE: `1.2063 µm`.

The detector produced duplicate associations on this synthetic frame, so its raw precision was `0.3753` and F1 was `0.5314`; the result must not be presented as an independent experimental accuracy claim. The benchmark is held out from training but comes from the same simulator family. The complete evaluator also reports per-diameter-bin recall, ROI coverage, bias, percentiles, and duplicate rates.

The strict `production` TensorRT FP16 preset also completed the same frame with 485 rows and identical primary/fallback route counts. Geometry-matched Torch/TensorRT rows selected the same slice in 479/485 cases and a slice within one plane in 483/485 cases. See [production-equivalence.md](production-equivalence.md) for interpretation.

## Recorded model metrics

Packaging metadata recorded:

- detector experimental mAP50 `0.96779` and mAP50–95 `0.56034`;
- fixed same-simulator blind detector F1 `0.7540`;
- primary-depth validation argmax MAE `89.28 µm`;
- fallback-depth validation argmax MAE `72.00 µm`; and
- diameter validation MAE `1.1269 µm`.

These metrics retain their original dataset-domain limitations described in [model-card.md](model-card.md).

## Acceptance for a new instrument

Before quantitative use on a new acquisition system:

1. validate acquisition YAML and transform outputs;
2. retain exact model hashes and backend metadata;
3. compare detector boxes with approved annotations;
4. compare depth against independently known axial positions across diameter bins;
5. compare final diameter against physical truth across the supported range;
6. evaluate fallback routes separately; and
7. repeat on an acquisition not used for calibration or checkpoint selection.

For Gabor acquisitions, domain-matched depth and diameter validation is mandatory because packaged learned checkpoints were trained on dual-camera phase-retrieval crops.
