# Training reproduction

HoloD3 records every packaged model's data, initializer, hyperparameters, selected checkpoint, and evaluation command in [reproduce-production.json](../configs/training/reproduce-production.json). The runner executes only repository-relative Python commands, writes only below `runs/`, never overwrites a stage output, and rejects network/destructive arguments in the ledger.

## Prepare a fresh checkout

```bash
uv sync --frozen --extra gpu --extra train
uv run hf auth login
uv run holod3 fetch-models --scope reproduction
uv run holod3 fetch-data --scope training
uv run holod3 fetch-data --scope evaluation
uv run holod3 verify
uv run holod3 reproduce --stage check
```

`fetch-models --scope reproduction` installs four production weights plus two detector initializers. `fetch-data --scope training` installs five training units. The evaluation scope installs the 12-frame synthetic benchmark.

The full asset check resolves every detector image/label, 104,172 primary-depth crops, 30,600 fallback-depth crops, 14,080 diameter crops, and 12 synchronized benchmark frames. It verifies pinned hashes and row counts before training.

## Inspect before executing

```bash
uv run holod3 reproduce --stage all --dry-run
```

The dry run prints all commands and reports missing inputs or reserved outputs without writing anything. Copy the JSON ledger to a new filename if GPU device indices, batch sizes, or output roots must differ on your machine; keep the canonical file unchanged for comparison.

## Stage order

```text
check
  ↓
yolo-baseline
  ↓
yolo-production
  ↓
depth-fallback
  ↓
depth-primary
  ↓
diameter
  ↓
evaluate
```

Stages may run independently after their inputs exist:

```bash
uv run holod3 reproduce --stage depth-primary
```

`--stage all` executes the order above. Outputs are rooted at `runs/reproduction/production/`.

## Detector stages

### Experimental base

- Data: 33 experimental training MinIP images and 9 experimental validation images.
- Initializer: `models/initializers/detector-architecture.pt` (YOLO26l architecture weights).
- Maximum epochs: 2,000; patience: 300; batch: 2; image size: 1,280.
- Multi-scale: 0.2; affine scale: 0.5; detector maximum: 400.
- Optimizer: Ultralytics automatic choice; initial/final learning-rate factors: 0.01/0.01.
- Deterministic seed: 0.

### Production fine-tuning

- Data: explicit 228-presentation training schedule (132 experimental + 96 synthetic) and 9 experimental validation images.
- Exact packaged initializer: `models/initializers/detector-baseline.pt`.
- Epochs: 20; batch: 2; image size: 1,280; detector maximum: 600.
- Optimizer: AdamW; initial learning rate `2e-5`; final factor `0.1`; no warmup.
- BatchNorm running statistics remain frozen; each epoch is saved.
- The production artifact is explicitly epoch 19, copied to `selected/detector.pt`.

For direct packaged-lineage parity, keep the default `--yolo-baseline-source packaged`. In that mode, `--stage all` still trains the experimental base as an independently inspectable stage, but production fine-tuning deliberately consumes the checksum-pinned packaged baseline. To train the detector chain from the architecture initializer and feed that newly trained base into fine-tuning:

```bash
uv run holod3 reproduce --stage all --yolo-baseline-source reproduced
```

With `--stage all`, the latter is the complete from-scratch path: the detector base and fine-tune are trained in sequence, while both depth scorers and the diameter regressor initialize their architectures without pretrained checkpoints in either mode. The historical detector base-stage selected epoch was not retained as a named immutable artifact. Consequently, a newly selected base checkpoint is not guaranteed to be byte-identical to the packaged fine-tuning input.

## Depth models

Both models learn pairwise focus-nearness scores from 64×64 reconstructed crops and use 8 epochs, batch 256, Adam-style learning rate `0.002`, weight decay `0.0001`, width 16, and diameter scalar conditioning.

### Fallback depth scorer

- Data: 30,600 crop rows, 1,600 particles, 20 frames.
- Pairs per epoch: 8,000; validation pairs: 3,000.
- Pixel/outer-region/neighbor/near-far augmentation disabled.
- Fixed preserved random seed: `260617`. This is a training RNG value in the immutable ledger, not a directory, release, or user-facing version.

### Primary depth scorer

- Data: 104,172 crop rows, 5,200 particles, 16 frames.
- Leakage-safe frame-hash split: 12 train frames / 3,893 particles and 4 validation frames / 1,307 particles.
- Pairs per epoch: 30,000; validation pairs: 5,000.
- Pixel shifts, outer-region swaps, neighbour mixing, diameter perturb/dropout/overestimate augmentation, balanced diameter sampling, and near/far pair sampling are enabled exactly as listed in the JSON ledger.
- Fixed preserved random seed: `26071101`; deterministic Torch algorithms are required.
- Checkpoint selection penalizes worst-bin error, catastrophic depth errors, and pair errors.

## Diameter model

- Data: 14,080 reconstructed crops (7,680 focused-domain and 6,400 dense-domain rows).
- Split: 11,691 train, 1,181 validation, 1,208 test rows.
- Epochs: 70; patience: 16; batch: 512.
- Learning rate: `0.0018`; weight decay: `0.0001`.
- Fixed preserved random seed: `260624`; deterministic Torch algorithms enabled.
- Centre shifts up to 4 pixels and geometric/photometric augmentation enabled.
- Selection score weights overall, large, and very-large validation MAE by 0.50/0.30/0.20.

## Evaluation stage

The evaluation stage:

1. compares reproduced and packaged detectors on the same experimental validation split;
2. evaluates the reproduced diameter checkpoint on the held-out crop split;
3. runs the complete Torch pipeline on all 12 benchmark frames; and
4. scores detection, depth, and diameter against benchmark truth.

The benchmark belongs to the same simulator family as synthetic training data. Treat it as a deterministic regression benchmark, not independent experimental validation.

## Reproducibility limits

The ledger reproduces inputs, code, random seeds, stage order, hyperparameters, and checkpoint-selection rules. Exact floating-point/byte equality is not promised across GPU models, CUDA, cuDNN, Torch, TensorRT, or Ultralytics versions. Record the environment emitted by each trainer and compare metrics plus checkpoint hashes rather than assuming bitwise equality.

Full training requires a CUDA GPU and substantial temporary storage. The data install is about 738 MB after extraction and about 495 MB of compressed cache; training outputs and framework caches require additional space. GPU memory depends on the device and configured batch sizes, so review the dry run before execution.

`uv.lock` pins the Python dependency graph; use `uv sync --frozen` so it is not re-resolved. If a managed or read-only home directory prevents uv from writing its default cache, use a repository-local ignored cache for that shell, for example `UV_CACHE_DIR=.uv-cache uv sync --frozen --extra gpu --extra train`.
