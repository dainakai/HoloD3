# Third-party notices

HoloD3 depends on third-party Python packages. Direct and transitive versions are recorded in `pyproject.toml` and `uv.lock`; each package remains subject to its upstream license.

The production detector and its two training initializers are Ultralytics YOLO checkpoints. Embedded checkpoint metadata reports Ultralytics versions and applicable upstream terms. HoloD3 does not replace or relax those terms.

The private experimental annotations and training/evaluation datasets do not have a public redistribution grant in this repository. Repository or Hugging Face access does not imply permission to redistribute them.

Review upstream notices for Torch, Torch-TensorRT, Ultralytics, OpenCV, Flask, Plotly, NumPy, pandas, Pillow, SciPy, Matplotlib, PyYAML, Hugging Face Hub, zstandard, and other packages resolved in `uv.lock` before redistribution or deployment.
