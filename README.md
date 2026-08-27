# HoloD3

HoloD3 detects particles in hologram minimum-intensity projections (MinIP), estimates each particle's depth, and estimates its diameter. It supports:

- validated dual-camera phase-retrieval inference;
- single-hologram inline Gabor reconstruction;
- optional depth and diameter fallback rules;
- direct checkpoint overrides for custom models;
- CSV output and a self-contained animated 3D scatter plot;
- CLI, Python, and trusted-local Web UI workflows; and
- checksum-verified training reproduction from private Hugging Face bundles.

Measurements are independent per frame. HoloD3 does not perform temporal tracking.

## One-frame quick start

HoloD3 targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for a reproducible environment.

The quick start selects CUDA automatically when available. CPU execution is supported by the portable preset, but the complete 1,024-plane reconstruction is computationally expensive without a GPU.

```bash
uv sync --frozen
uv run hf auth login
uv run holod3 fetch-models --scope production
uv run holod3 verify
uv run holod3 demo --limit 1
```

The included acquisition contains real experimental raw holograms. The command writes:

```text
runs/demo/
├── particles_3d.csv       # one row per particle measurement
├── particles_3d.html      # animated, interactive 3D scatter
├── pipeline_summary.json  # optics, fallback policy, timing, and paths
├── frame_stats.csv
└── _inputs/minip/         # exact detector inputs used by this run
```

Open `runs/demo/particles_3d.html` in a browser. Process all six included frames with:

```bash
uv run holod3 demo --limit 0 --run-dir runs/experimental-demo
```

Use `--overwrite` only when intentionally replacing an existing run.

## Web UI

```bash
uv run holod3 web
```

Open <http://127.0.0.1:7860>, leave the included acquisition selected, choose a frame limit, and select **Start full pipeline**. When the run finishes, the page exposes CSV, summary, log, and 3D-animation links.

The UI has no authentication and can read server-side paths. Keep it bound to `127.0.0.1`; do not expose it to an untrusted network.

## Your own acquisition

Copy the matching template and edit every value marked for the user:

```bash
cp configs/acquisitions/dual-camera-template.yaml my-acquisition/acquisition.yaml
uv run holod3 validate-acquisition my-acquisition/acquisition.yaml
uv run holod3 infer \
  --acquisition my-acquisition/acquisition.yaml \
  --run-dir runs/my-acquisition \
  --preset portable-torch \
  --limit 0
```

An acquisition file specifies relative image directories, holography mode, wavelength, sensor pixel pitch, image size, reconstruction origin, slice spacing/count, phase-retrieval distance, calibration, and optional ordered transforms. Matching file stems define synchronized frames; camera-specific directory names are not required.

If your images need an instrument-specific conversion, reference either a built-in transform or a Python function:

```yaml
transforms:
  primary:
    - function: examples/custom_transform.py:subtract_dark_offset
      kwargs:
        offset: 0.04
```

Transform functions receive a two-dimensional `float32` grayscale array in `[0, 1]` and return an array with the same contract. See [Acquisition configuration](docs/acquisitions.md).

## Single-hologram Gabor mode

```bash
cp configs/acquisitions/single-gabor-template.yaml my-gabor-data/acquisition.yaml
uv run holod3 validate-acquisition my-gabor-data/acquisition.yaml
uv run holod3 infer \
  --acquisition my-gabor-data/acquisition.yaml \
  --run-dir runs/my-gabor-data \
  --preset portable-torch \
  --limit 0
```

`single_gabor` uses one intensity image per frame and initializes a zero-phase wavefront as `sqrt(intensity)`. A second camera and phase retrieval are not used.

The packaged depth and diameter checkpoints were trained on dual-camera phase-retrieval crops. Gabor execution is implemented and tested, but its scientific accuracy is not validated with those checkpoints. Use Gabor-domain checkpoints through the four weight override flags before treating measurements as calibrated results.

## Direct model use and pipeline integration

Detector only:

```bash
uv run holod3 detect path/to/minip.png \
  --output runs/detect/detections.json \
  --annotated-output runs/detect/detections.png
```

Custom checkpoints:

```bash
uv run holod3 infer \
  --acquisition my-acquisition/acquisition.yaml \
  --run-dir runs/custom-models \
  --yolo-weights path/to/detector.pt \
  --depth-primary-weights path/to/depth-primary.pt \
  --depth-fallback-weights path/to/depth-fallback.pt \
  --diameter-weights path/to/diameter.pt
```

Python API:

```python
from holod3 import HoloD3Pipeline

result = HoloD3Pipeline("portable-torch").run(
    acquisition="my-acquisition/acquisition.yaml",
    run_dir="runs/python-api",
    limit=0,
)
particles = result.particles()
print(result.particles_csv, result.visualization_html)
```

See [integration.md](docs/integration.md) for configuration overrides and the CSV contract.

## Reproduce training

Training images and crops are not Git objects. They are grouped by purpose in a private dataset repository and installed only when requested:

```bash
uv sync --frozen --extra train
uv run hf auth login
uv run holod3 fetch-models --scope reproduction
uv run holod3 fetch-data --scope training
uv run holod3 fetch-data --scope evaluation
uv run holod3 reproduce --stage check
uv run holod3 reproduce --stage all --dry-run
```

Run individual stages after reviewing GPU device values and storage requirements. The complete immutable command ledger is [configs/training/reproduce-production.json](configs/training/reproduce-production.json); [training.md](docs/training.md) explains every stage and expected output.

## Documentation map

- [Acquisition configuration](docs/acquisitions.md): directory layout, optics, transforms, and Gabor mode.
- [Inference](docs/inference.md): backends, fallbacks, output columns, and troubleshooting.
- [Web UI](docs/web-ui.md): exact browser workflow and local security model.
- [Training](docs/training.md): bundle contents and complete reproduction.
- [Data](docs/data.md): what stays in Git and what is fetched from Hugging Face.
- [Architecture](docs/architecture.md): processing flow and extension points.
- [Model card](docs/model-card.md): model domains, metrics, and limitations.
- [Validation](docs/validation.md): tests and recorded smoke results.
- [Repository layout](docs/repository-layout.md): directory ownership, downloaded assets, and run outputs.

## Access and licensing

The HoloD3 project-specific source code and documentation are available under the MIT License. The Git-tracked experimental demo, separately downloaded model checkpoints, and private training/evaluation datasets are not covered by that license. A Hugging Face read token grants access only to approved accounts and is not redistribution permission. Never commit tokens. Review [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.
