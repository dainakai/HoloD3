# Model artifacts

Checkpoint binaries are not Git objects. Download them from the pinned private Hugging Face revision:

```bash
uv run hf auth login
uv run holod3 fetch-models --scope production
uv run holod3 verify
```

## Semantic paths

| Role | Local path | Download scope |
| --- | --- | --- |
| MinIP detector | `models/production/detector.pt` | production |
| Primary depth scorer | `models/production/depth-primary.pt` | production |
| Small-particle depth fallback | `models/production/depth-fallback.pt` | production |
| Diameter regressor | `models/production/diameter.pt` | production |
| Detector architecture initializer | `models/initializers/detector-architecture.pt` | reproduction |
| Detector fine-tuning initializer | `models/initializers/detector-baseline.pt` | reproduction |

`models/remote_manifest.json` pins remote paths, revision, byte sizes, SHA-256, and scopes. `models/production/manifest.json` records model roles, frameworks, training bundle IDs, selection notes, and training-input hashes.

`production` downloads four inference weights. `reproduction` downloads those four plus two detector initializers. `all` is an alias for every artifact listed in the clean manifest; it no longer exposes unrelated historical experiments.

Private repository access is not redistribution permission. Never commit an access token or model binary.
