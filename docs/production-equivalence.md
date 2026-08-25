# Production-equivalence statement

The `production` and `portable-torch` presets preserve the validated model semantics of the original operational particle-measurement chain while removing workstation-specific orchestration and data names.

## Preserved behavior

- The exact four production checkpoint bytes and SHA-256 hashes.
- MinIP detector clipping/scaling and YOLO thresholds.
- Contrast-area ROI measurement and detector-box fallback.
- Full-grid primary depth scoring and global score argmax.
- Row-level fallback depth routing for small fallback measurements.
- Combined focused/dense SliceDiamModel inference at the selected slice.
- Large-particle learned-diameter underprediction safety rule.
- Default crop size, slice block, batch sizes, and TensorRT/Torch backend semantics.

## Explicitly externalized behavior

The old implementation inferred acquisition layout and optical values from one workstation directory convention. HoloD3 now requires an acquisition YAML containing those values. The included experimental acquisition encodes the original experiment's wavelength, pixel pitch, reconstruction origin, axial spacing/count, phase-retrieval distance, iteration count, FFT padding, and secondary-camera calibration.

The fused core reads those values rather than embedding camera IDs, scene numbers, directory dates, or fixed physical scale constants. For an equivalent acquisition YAML, the dual-camera wavefront, propagation, model inputs, depth route, diameter route, and final row semantics remain equivalent.

## Backend equivalence

`production` preserves strict TensorRT FP16 execution for depth and diameter. `portable-torch` uses the same checkpoint parameters through Torch and is intended for portability and inspection. Floating-point scores can differ slightly by backend; near-tied slice scores can select neighbouring planes. Use the strict preset for formal operational comparisons and retain backend metadata.

On one pinned evaluation-bundle frame, both presets emitted 485 rows and the same 396/89 primary/fallback routing counts. After pairing rows by detector-box center (necessary because tiny confidence differences can reorder otherwise equivalent detections), 479/485 selected exactly the same slice and 483/485 differed by at most one slice; the maximum difference was five slices. The median and 95th-percentile absolute final-diameter differences were `0.0295 µm` and `0.1527 µm`. These are backend-parity observations for one frame, not universal numerical tolerances.

## New opt-in capabilities

The following additions do not alter validated dual-camera defaults:

- user-specified image transforms;
- automatic MinIP creation when projections are absent;
- true single-hologram Gabor wavefronts;
- semantic checkpoint overrides;
- animated HTML visualization;
- external checksum-verified dataset bundles; and
- trusted-local Web workflow.

Single-Gabor mode is an execution capability, not a claim that dual-camera-trained learned checkpoints are calibrated for that domain.

## Reproduction interpretation

The repository preserves exact training data manifests, selected production checkpoint bytes, initializers, commands, seeds, and selection rules. Hardware/software variation and the historically unnamed detector-base selected epoch prevent a general promise of byte-identical from-scratch retraining. See [training.md](training.md) for the two detector-initialization paths and exact limitations.
