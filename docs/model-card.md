# Model card

## Intended use

The packaged checkpoints perform per-frame particle detection, axial focus selection, and diameter estimation for the validated HoloD3 dual-camera phase-retrieval domain. They are research artifacts for controlled holographic acquisitions, not general-purpose object or depth models.

## Artifacts

| Role | File | SHA-256 |
| --- | --- | --- |
| MinIP detector | `models/production/detector.pt` | `4072d26f7e0f71e6a46183db643126db216b4bff9c2e88c4e602571eb3df044f` |
| Primary depth scorer | `models/production/depth-primary.pt` | `c69ea9c713f5d588ab5b1c43ef52702a449496b0c3d591b87f0349fa8f9e0d21` |
| Fallback depth scorer | `models/production/depth-fallback.pt` | `cfb3632c4c9e96a3e0da4d25f6ff23183f1e49869400ff6d39542fa1bdca3ea3` |
| Diameter regressor | `models/production/diameter.pt` | `50faaf673cd4e7fb2034d76c5e249c8a8b7eb7fcc98d8670039fc53c9f9a1f26` |

The private model manifest pins one immutable Hugging Face revision. HoloD3 verifies size and SHA-256 before inference.

## Detector

The detector is an Ultralytics YOLO26l checkpoint trained on MinIP particle boxes. Its final fine-tuning schedule combines experimental and synthetic projections with an explicit 4:1 experimental repeat ratio and uses experimental-only validation.

Recorded validation at packaging:

- experimental validation mAP50: `0.96779`;
- experimental validation mAP50–95: `0.56034`; and
- fixed same-simulator blind benchmark F1: `0.7540`.

The detector expects grayscale MinIP values clipped from `[0, 75]` and scaled to `[0, 255]`, confidence threshold `0.10`, NMS IoU `0.15`, and at most 600 detections in the full pipeline.

## Depth scorers

Both depth checkpoints are compact pairwise FocusScoreNet models that score 64×64 reconstructed particle crops with diameter scalar conditioning. The full pipeline computes a score at every selected slice and takes a global argmax.

The primary scorer was trained on 104,172 crop rows / 5,200 particles with frame-separated validation and robustness augmentation. Recorded selected-checkpoint validation argmax MAE was `89.28 µm`; the recorded worst diameter-bin MAE was `277.97 µm`.

The fallback scorer was trained on 30,600 crop rows / 1,600 particles using the preserved earlier training recipe. Recorded validation argmax MAE was `72.00 µm`.

The optional router applies the fallback scorer only to rows with a fallback conditioning measurement and diameter at most `75 µm`. The CSV records which scorer selected each result.

## Diameter regressor

The packaged SliceDiamModel combines 7,680 focused-domain and 6,400 dense-domain reconstructed crops. It predicts log diameter and uncertainty at the depth-selected slice. The selected artifact was epoch 52. Recorded validation MAE was `1.1269 µm`, and the recorded weighted checkpoint-selection score was `1.133`.

An optional safety rule replaces a severe learned underprediction with contrast-area diameter when the average detector-box side is at least `250 µm` and the learned diameter is below `0.35 × box side`.

## Domain and limitations

- Training and regression evaluation use 10 µm pixel scaling, 100 µm axial spacing, 64×64 particle crops, and dual-camera phase-retrieved wavefronts.
- Acquisition YAML can represent other optics, but changing physical sampling creates a model-domain shift unless checkpoints are retrained or calibrated.
- Single-hologram Gabor reconstruction is implemented; packaged depth and diameter weights were not trained or validated on Gabor crops.
- The synthetic benchmark is held out from training manifests but belongs to the same simulator family.
- The included experimental demo has no independent particle-level physical 3D truth.
- Very dense, overlapping, clipped, saturated, poorly background-corrected, or out-of-range particles may fail detection or bias ROI and learned estimates.
- HoloD3 performs independent per-frame measurement. `track_id` does not indicate temporal identity.
- HoloD3 does not infer trajectories, velocity, collision events, or particle material.

## Responsible interpretation

Inspect `diameter_method`, `depth_router_source`, `final_diameter_source`, uncertainty, detector confidence, acquisition parameters, and model hashes with every result. Validate custom instruments and Gabor acquisitions against physical 3D/diameter truth before quantitative scientific use.
