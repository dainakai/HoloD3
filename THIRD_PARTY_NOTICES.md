# Third-party notices

HoloD3 depends on third-party Python packages. Direct and transitive versions are recorded in `pyproject.toml` and `uv.lock`; each package remains subject to its upstream license.

The Git repository and its source archives do not contain model checkpoints. The production detector and its two training initializers are Ultralytics YOLO checkpoints downloaded separately from a private model repository. Embedded checkpoint metadata reports Ultralytics versions and applicable upstream terms. The HoloD3 MIT License does not replace or relax those terms.

The Git-tracked experimental demo, private experimental annotations, and training/evaluation datasets are not covered by the HoloD3 MIT License and do not have a public redistribution grant in this repository. Repository or Hugging Face access does not imply permission to redistribute them.

Review upstream notices for Torch, Torch-TensorRT, Ultralytics, OpenCV, Flask, Plotly, NumPy, pandas, Pillow, SciPy, Matplotlib, PyYAML, Hugging Face Hub, zstandard, and other packages resolved in `uv.lock` before redistribution or deployment.
