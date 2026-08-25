"""Run one included full-pipeline frame through the public Python API."""

from holod3 import HoloD3Pipeline

pipeline = HoloD3Pipeline("portable-torch")
result = pipeline.run(
    acquisition="data/demo/experimental/acquisition.yaml",
    run_dir="runs/examples/full_pipeline",
    limit=1,
)
print(result.particles().head())
