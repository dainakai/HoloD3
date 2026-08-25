# Inference

## Processing flow

For every selected frame, HoloD3:

1. copies a supplied MinIP or reconstructs one from raw holograms;
2. applies the detector contrast mapping and YOLO particle detection;
3. measures a contrast-area region for particle centre and depth conditioning;
4. reconstructs the configured depth planes;
5. scores every selected plane with the primary depth model;
6. optionally routes small-particle rows through the fallback depth model;
7. evaluates the diameter model at the selected plane;
8. optionally applies the large-particle diameter safety rule; and
9. writes `particles_3d.csv`, provenance, timing, and an animated HTML view.

No step links a particle across frames.

## Presets

`portable-torch` uses Torch for depth and diameter inference. It is the recommended first run because it avoids TensorRT compilation. Its `auto` device policy selects CUDA when available and otherwise selects CPU. A full 1,024-plane CPU run is valid but usually impractical; reduce the reconstruction grid for a functional CPU check.

`production` uses TensorRT FP16 for the two learned reconstruction models and refuses silent backend fallback. It requires CUDA, `torch-tensorrt`, and a compatible GPU. The first shape invocation may compile engines; later invocations are faster.

Depth and diameter state-dictionary checkpoints are loaded with PyTorch's restricted weights-only loader. Ultralytics detector checkpoints are executable pickle-based artifacts; use only checkpoints you created or obtained from a trusted source. This warning applies especially to checkpoint paths entered in the trusted-local Web UI.

Both presets use the same detector, thresholds, model weights, reconstruction grid from the acquisition file, and fallback semantics.

```bash
uv run holod3 config --preset portable-torch
uv run holod3 config --preset production
```

## Complete CLI example

```bash
uv run holod3 infer \
  --acquisition my-acquisition/acquisition.yaml \
  --run-dir runs/my-acquisition \
  --preset portable-torch \
  --limit 0
```

Selection options are zero-based:

- `--limit 1` processes one frame; `--limit 0` processes all selected frames.
- `--start-index N` starts at sorted frame index `N`.
- `--end-index N` ends at sorted frame index `N`, inclusive.

`--dry-run` prints the exact internal command without running it. `--overwrite` explicitly allows replacement of run outputs. `--stop-after-preprocessing` retains detector/ROI staging files and stops before depth inference. Start a new run directory when model, threshold, acquisition, or preprocessing settings change; HoloD3 does not silently reuse staging data across runs.

## Model overrides

The normal production checkpoint locations are:

| Role | Path |
| --- | --- |
| MinIP detector | `models/production/detector.pt` |
| Primary depth scorer | `models/production/depth-primary.pt` |
| Small-particle depth fallback | `models/production/depth-fallback.pt` |
| Reconstructed-slice diameter regressor | `models/production/diameter.pt` |

Use `--yolo-weights`, `--depth-primary-weights`, `--depth-fallback-weights`, and `--diameter-weights` to replace them independently. A custom inference YAML may also replace paths and all thresholds.

## Optional fallback rules

All fallbacks are explicit Boolean settings and may be disabled independently:

```bash
uv run holod3 infer \
  --acquisition my-acquisition/acquisition.yaml \
  --run-dir runs/no-fallbacks \
  --no-bbox-fallback \
  --no-depth-router \
  --no-diameter-fallback
```

The validated preset uses:

- **MinIP bounding-box fallback:** if contrast-area measurement cannot identify a valid component, use detector-box size and centre.
- **Depth router:** use the fallback depth scorer when the measured diameter is at most `75 µm` and the row's measurement method indicates fallback.
- **Diameter underprediction fallback:** for an average detector-box side of at least `250 µm`, use contrast-area diameter when the learned estimate is below `0.35 × box side`.

The CSV records route and source columns for every decision.

## Output contract

The main stable columns are:

| Column | Unit/meaning |
| --- | --- |
| `frame` | Zero-based selected-frame order. |
| `file` | Model-ready MinIP filename. |
| `row_id` | Stable row identifier inside the run. |
| `track_id` | Compatibility identifier unique per detection; it is not a temporal track. |
| `conf` | Detector confidence. |
| `seg_xc`, `seg_yc` | Measured centre in pixels. |
| `x_um`, `y_um` | Centre scaled by acquisition `pixel_pitch_um`. |
| `slice` | Selected one-based reconstruction-plane index. |
| `z_um` | Axial offset from slice 1 in micrometres. |
| `depth_um` | Absolute configured reconstruction depth: `reconstruction_start_um + z_um`. |
| `diameter_um` | Contrast-area or detector-box conditioning diameter. |
| `slice_diam_pred_um` | Learned diameter estimate at the selected slice. |
| `final_diameter_um` | Final estimate after the optional safety rule. |
| `diameter_method` | How the conditioning diameter was measured. |
| `depth_router_source` | Primary or fallback depth model choice. |
| `final_diameter_source` | Learned slice model or contrast-area fallback. |
| `*_checkpoint_id` | Portable semantic checkpoint identifier; external paths are reduced to a filename label. |
| `*_checkpoint_sha256` | SHA-256 of the exact detector/depth/diameter file used. |
| `*_checkpoint_bytes` | Exact checkpoint byte size. |

The file also retains detector boxes, normalized coordinates, ROI bounds, model scores, uncertainty estimates, and checkpoint provenance.

`pipeline_summary.json` records acquisition mode and configuration hash, optical values, selected frame count, model hashes/sizes, fallback configuration, backend options, outputs, and elapsed times. Shareable metadata uses `repository:`, `acquisition:`, `run:`, or `external:` identifiers rather than embedding workstation roots. Intermediate CSV references are set to `null` when those files are intentionally removed after fusion.

## Visualization

Inference creates `particles_3d.html` unless `--no-visualization` is passed. To visualize an existing CSV:

```bash
uv run holod3 visualize runs/my-acquisition/particles_3d.csv \
  --output runs/my-acquisition/particles_3d.html
```

The HTML embeds Plotly by default and opens without a running HoloD3 server. Marker size and colour encode final diameter. The frame slider and play button animate independent measurements, not trajectories.

## Common failures

### Missing private checkpoints

```bash
uv run hf auth login
uv run holod3 fetch-models --scope production
uv run holod3 verify
```

### Frame-stem mismatch

Run `holod3 validate-acquisition`. Every configured image directory must contain exactly the same set of stems.

### Out of memory

Use `portable-torch`, process fewer frames, or copy an inference YAML and reduce `runtime.depth_batch_size`, `runtime.diameter_batch_size`, and prefetch counts. The full reconstruction grid can be expensive.

### CPU-only execution

The portable preset selects CPU automatically when CUDA is unavailable. For a quick functional check, copy the acquisition YAML and temporarily reduce `optics.slice_count`; restore the scientifically required grid before measurement. The production preset never falls back to CPU.

### Gabor warning

The warning is intentional: execution support and checkpoint calibration are separate claims. Use domain-matched weights for quantitative single-Gabor measurements.
