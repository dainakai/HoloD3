# Data distribution

The Git repository contains only the small experimental demo needed for immediate inference. Training crops and the synthetic evaluation benchmark are immutable private Hugging Face bundles downloaded on demand.

## Data kept in Git

`data/demo/experimental` contains six synchronized frames:

- six primary raw holograms;
- six secondary raw holograms;
- six precomputed MinIP images;
- one secondary-camera distortion calibration; and
- a complete acquisition configuration.

This is real experimental input intended for workflow testing. It is not a statistically representative benchmark and has no independent particle-level depth/diameter truth.

## Downloadable bundles

| Bundle ID | Purpose | Main content |
| --- | --- | --- |
| `detector-experimental` | Detector base stage | 33 training and 9 validation experimental MinIP images with YOLO labels. |
| `detector-mixed` | Detector fine-tuning | 132 experimental and 96 synthetic training presentations, plus 9 experimental validation images. The repeated 4:1 schedule is explicit. |
| `depth-primary` | Primary depth scorer | 104,172 reconstructed crops representing 5,200 particles. |
| `depth-fallback` | Small-particle fallback scorer | 30,600 reconstructed crops representing 1,600 particles. |
| `diameter-combined` | Diameter regressor | 14,080 focused/dense crops; train/validation/test counts are 11,691/1,181/1,208. |
| `evaluation-benchmark` | End-to-end evaluation | 12 held-out synthetic frames with paired holograms, MinIP, calibration, acquisition YAML, and particle truth. |

The benchmark is held out from the packaged training manifests but comes from the same simulator family. It does not replace evaluation on an independently acquired experimental dataset with physical 3D truth.

## Fetch commands

```bash
uv run hf auth login
uv run holod3 fetch-data --scope training
uv run holod3 fetch-data --scope evaluation
```

Fetch one unit when only one stage is needed:

```bash
uv run holod3 fetch-data --bundle depth-primary
```

Archives download to `data/.downloads/` and install atomically under `data/downloaded/<bundle-id>/`. Both paths are ignored by Git. A marker stores the pinned remote revision and archive checksum.

The downloader:

1. selects bundles from [remote_manifest.json](../data/remote_manifest.json);
2. downloads from the pinned private repository revision;
3. verifies byte size and SHA-256;
4. rejects absolute paths, traversal, links, and device files in archives;
5. extracts into a temporary staging directory;
6. checks required paths; and
7. atomically installs the verified bundle.

Valid installed bundles are not downloaded or extracted again unless `--force` is passed.

## Bundle portability

Every text manifest uses semantic IDs and repository-relative paths. Archives contain no source-workstation mount paths, user names, host names, storage-volume labels, or sibling-workspace names.

YOLO `data.yaml` files intentionally point to `data/downloaded/<bundle-id>`, matching the standard install location. Depth crop paths are relative to their manifest. The diameter loader accepts the packaged repository-relative image references while constraining every resolved image to the explicitly selected bundle root.

## Integrity and access

The repository pins each archive by immutable Hugging Face commit, byte size, and SHA-256. `uv run holod3 reproduce --stage check` additionally resolves every detector image/label and every training crop through the same loaders used for training.

These datasets are private. Authentication is access control, not redistribution permission. Never embed an `HF_TOKEN` in a config, command committed to version control, notebook, log, or Web UI field.
