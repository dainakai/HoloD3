# Python and pipeline integration

## Minimal API

```python
from holod3 import HoloD3Pipeline

pipeline = HoloD3Pipeline("portable-torch")
pipeline.validate_inputs("my-acquisition/acquisition.yaml")

result = pipeline.run(
    acquisition="my-acquisition/acquisition.yaml",
    run_dir="runs/integration",
    limit=0,
)

particles = result.particles()       # pandas.DataFrame
summary = result.summary()           # dict
print(result.particles_csv)
print(result.visualization_html)
```

`PipelineResult` also exposes the raw detection CSV, fused depth/diameter metrics, hybrid-diameter metrics, and run directory. For a `stop_after_preprocessing=True` run, the particle path refers to `raw_detections.csv` and both learned-stage metric paths are `None` because those stages did not run.

## Separate acquisition and model policy

An acquisition configuration owns scientific input geometry: image paths, holography mode, wavelength, pixel pitch, reconstruction origin/grid, calibration, and transforms.

An inference configuration owns model/runtime policy: checkpoint paths, detector thresholds, learned-model crop/slice selection, fallback rules, GPU device, backend, batch sizes, and prefetch settings.

Keeping them separate allows one acquisition to run with several model sets without copying physical parameters.

```python
from holod3.acquisition import AcquisitionConfig
from holod3.config import PipelineConfig

acquisition = AcquisitionConfig.load("my-acquisition/acquisition.yaml")
config = PipelineConfig.load("configs/inference/portable-torch.yaml")
pipeline = HoloD3Pipeline(config)
result = pipeline.run(acquisition=acquisition, run_dir="runs/separate-configs")
```

The subprocess runner requires an `AcquisitionConfig` loaded from a YAML file because relative transform and data paths need a stable base directory.

## Programmatic overrides

Configuration dataclasses are frozen. Use the explicit helpers to create changed copies:

```python
config = (
    PipelineConfig.preset("portable-torch")
    .with_models(
        yolo="weights/detector.pt",
        depth_primary="weights/depth-primary.pt",
        depth_fallback="weights/depth-fallback.pt",
        diameter="weights/diameter.pt",
    )
    .with_fallbacks(
        minip_bbox=True,
        depth_router=False,
        diameter_underprediction=True,
    )
    .with_runtime(
        device="cuda:0",
        yolo_device="0",
        depth_model_backend="torch",
        diameter_model_backend="torch",
        strict_backend=False,
    )
)
```

Model paths may be repository-relative or absolute. HoloD3 validates every checkpoint required by the enabled routes before building a command; the fallback depth checkpoint is not required when `depth_router=False`.

## Inspect the exact command

```python
command = pipeline.build_command(
    acquisition="my-acquisition/acquisition.yaml",
    run_dir="runs/inspect",
    limit=1,
)
print(command)
```

This performs acquisition/model validation but does not run inference or create the run directory.

## Capture logs

```python
from pathlib import Path

log_path = Path("runs/integration.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("w", encoding="utf-8") as stream:
    result = pipeline.run(
        acquisition="my-acquisition/acquisition.yaml",
        run_dir="runs/integration-with-log",
        limit=1,
        output_stream=stream,
    )
```

## Detector-only API

```python
from holod3.detector import ParticleDetector

detector = ParticleDetector(
    weights="models/production/detector.pt",
    device="0",
)
detections = detector.predict("path/to/minip.png")
for detection in detections:
    print(detection.to_dict())
```

Detector-only inference uses the validated MinIP contrast transform but does not reconstruct depth or diameter.

## Custom preprocessing plugin packaging

For a reusable pipeline, package transforms in an importable Python module and reference `package.module:function` from acquisition YAML. File-based `file.py:function` references are convenient for one acquisition and resolve relative to the YAML.

Transform plugins run inside the HoloD3 process. Treat them as trusted code, pin their source/version alongside the acquisition, and never load plugins from untrusted users.

## Output consumption

Use `row_id` as the stable row key for one run. `track_id` is retained for compatibility but is unique per detection and does not represent a trajectory. Use `frame` and `file` for frame grouping, `x_um/y_um/z_um` for the relative 3D scatter coordinate, `depth_um` for absolute configured axial depth, and `final_diameter_um` for the post-fallback estimate.

Downstream code should preserve these provenance columns:

- `diameter_method`
- `depth_router_source`
- `depth_primary_checkpoint`
- `depth_fallback_checkpoint`
- `slice_diam_model`
- `final_diameter_source`
- `final_diameter_rule`

They distinguish learned estimates from optional fallback routes.

## Concurrency and run directories

Use a new run directory per invocation. HoloD3 refuses many existing outputs unless overwrite is explicitly enabled. Avoid two processes writing the same run directory. Model checkpoints may be shared read-only across processes; GPU memory and framework caches are process-specific.
