# Validation

## Automated coverage

The test suite covers:

- acquisition schema, path resolution, stem synchronization, selection, and mode constraints;
- built-in and file/module custom transform contracts;
- dual-camera phase retrieval and single-Gabor wavefront construction on CPU fixtures;
- MinIP preparation and image-shape validation;
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
