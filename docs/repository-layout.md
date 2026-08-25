# Repository layout

The Git checkout stays small: public code, configuration contracts, manifests, tests, documentation, and a six-frame experimental demo are tracked. Production checkpoints and full training/evaluation bundles are installed on demand from private Hugging Face repositories and remain ignored Git objects.

```text
HoloD3/
├── holod3/                    # Supported Python package, CLI, Web UI, and public API
│   ├── acquisition.py        # Portable acquisition schema and input validation
│   ├── artifacts.py          # Model manifest download and SHA-256 verification
│   ├── checkpoints.py        # Restricted state-dictionary checkpoint loader
│   ├── config.py             # Typed inference/model/fallback/runtime policy
│   ├── datasets.py           # Safe dataset-bundle download and atomic installation
│   ├── detector.py           # Direct MinIP detector API
│   ├── pipeline.py           # HoloD3Pipeline and PipelineResult facade
│   ├── reconstruction.py     # Dual phase retrieval, single Gabor, propagation, MinIP
│   ├── transforms.py         # Built-in and user-supplied image transforms
│   ├── visualization.py      # Self-contained animated 3D HTML
│   ├── web.py                # Trusted-local Flask application
│   ├── templates/            # Web page markup
│   └── static/               # Web JavaScript and CSS
├── configs/
│   ├── acquisitions/         # Dual-camera and single-Gabor YAML templates
│   ├── inference/            # Portable Torch and strict TensorRT presets
│   └── training/             # Immutable production reproduction command ledger
├── data/
│   ├── demo/experimental/    # Six synchronized raw frames, MinIP, calibration, YAML
│   ├── remote_manifest.json  # Pinned private dataset bundles, hashes, and contents
│   ├── downloaded/           # Ignored bundle install location, created on demand
│   ├── .downloads/           # Ignored compressed archive cache
│   └── README.md             # Data roles and access policy
├── models/
│   ├── production/           # Manifest plus ignored semantic checkpoint paths
│   ├── initializers/         # Ignored detector reproduction initializers
│   ├── remote_manifest.json  # Pinned model revision, byte sizes, and SHA-256 values
│   └── README.md             # Model role and download map
├── src/
│   ├── detection/            # Executable detector, ROI, learned-depth/diameter core
│   ├── depth/                # Primary and fallback depth training programs
│   ├── diam/                 # Canonical diameter model, training, and inference helpers
│   ├── evaluation/           # Detector, diameter, and end-to-end scoring
│   ├── pipeline/             # Fused executable orchestrator and runtime helpers
│   └── yolo/                 # Detector dataset and exact legacy-compatible trainer
├── scripts/
│   ├── run_reproduction.py   # Safe staged command-ledger runner
│   ├── run_reproduced_pipeline.py
│   ├── select_checkpoint.py
│   └── verify_assets.py      # Full model/data/crop/holdout integrity check
├── examples/                 # Minimal detector, pipeline, and custom-transform code
├── tests/                    # Fast unit/contract tests plus marked CUDA integration tests
├── docs/                     # Acquisition, inference, Web, training, validation, model docs
├── runs/                     # Ignored run outputs; only .gitkeep is tracked
├── pyproject.toml            # Package, dependencies, CLI entry point, test/lint policy
├── uv.lock                   # Frozen Python dependency graph
├── LICENSE.md
└── THIRD_PARTY_NOTICES.md
```

## Ownership boundaries

- Put instrument geometry and paths in one acquisition YAML; do not add host-specific paths to inference presets.
- Put model selection, thresholds, fallbacks, device, and backend policy in an inference YAML or `PipelineConfig` override.
- Treat `holod3/` as the supported integration surface. `src/` contains executable numerical/training stages used by that facade.
- Treat `data/downloaded/`, `models/**/*.pt`, and `runs/` as reproducible local materializations, not source files.
- Durable coding-agent session state belongs in a sibling `session/` directory outside this repository.

## Generated run directory

A complete run writes a self-contained measurement result:

```text
run-directory/
├── particles_3d.csv
├── particles_3d.html
├── pipeline_summary.json
├── raw_detections.csv
├── raw_detection_summary.json
├── frame_stats.csv
├── fused_depth_slice_metrics.json
├── hybrid_diameter_metrics.json
└── _inputs/minip/
```

Intermediate depth/diameter CSVs are deleted after fusion by default. Set `runtime.keep_intermediate_csv: true` in a copied inference configuration only when they are needed for debugging.
