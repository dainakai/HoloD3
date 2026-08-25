from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch.nn as nn
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "runs" / "training" / "detector"


def repo_relative(path: Path) -> Path:
    path = Path(path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"Path must stay inside the repository: {resolved}")
    return resolved


def default_dataset_yaml() -> Path:
    return REPO_ROOT / "data" / "downloaded" / "detector-experimental" / "data.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on an exported dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset data.yaml. Defaults to the downloaded experimental detector bundle.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "models" / "initializers" / "detector-architecture.pt",
        help="Initial model weights. Relative paths are resolved from this repo.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--multi-scale", type=float, default=0.2)
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Ultralytics affine scale augmentation. Leave unset to use the "
            "Ultralytics default; set 0 to disable scale augmentation."
        ),
    )
    parser.add_argument("--max-det", type=int, default=400)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--box", type=float, default=None, help="Override the Ultralytics box loss weight.")
    parser.add_argument("--cls", type=float, default=None, help="Override the Ultralytics classification loss weight.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=None,
        help="Override Ultralytics warmup_epochs; leave unset to preserve its default.",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=None,
        help="Override Ultralytics warmup_bias_lr; leave unset to preserve its default.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save an additional checkpoint every N epochs; -1 disables periodic saves.",
    )
    parser.add_argument(
        "--freeze-bn-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep pretrained BatchNorm running statistics fixed during fine-tuning.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N model modules using the Ultralytics freeze argument.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_DIR,
        help="Training output directory. Default: runs/detect in this repo.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name. Default: dataset directory name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Training device, e.g. "0", "1", or "cpu". Default: Ultralytics auto.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = repo_relative(args.data) if args.data else default_dataset_yaml()
    weights = repo_relative(args.weights)
    project = repo_relative(args.project)
    name = args.name or data_yaml.parent.name

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset data.yaml does not exist: {data_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights}")

    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "multi_scale": args.multi_scale,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "max_det": args.max_det,
        "patience": args.patience,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "seed": args.seed,
        "deterministic": True,
        "amp": args.amp,
        "save_period": args.save_period,
        "plots": True,
    }
    if args.device is not None:
        train_args["device"] = args.device
    if args.scale is not None:
        train_args["scale"] = args.scale
    if args.warmup_epochs is not None:
        train_args["warmup_epochs"] = args.warmup_epochs
    if args.warmup_bias_lr is not None:
        train_args["warmup_bias_lr"] = args.warmup_bias_lr
    if args.freeze is not None:
        train_args["freeze"] = args.freeze
    if args.box is not None:
        train_args["box"] = args.box
    if args.cls is not None:
        train_args["cls"] = args.cls

    run_dir = project / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "ultralytics_version": importlib.metadata.version("ultralytics"),
        "weights": str(weights.relative_to(REPO_ROOT)),
        "weights_sha256": sha256(weights),
        "data_yaml": str(data_yaml.relative_to(REPO_ROOT)),
        "data_yaml_sha256": sha256(data_yaml),
        "workspace_git_state": "unavailable: workspace is not a Git repository",
        "freeze_bn_stats": args.freeze_bn_stats,
        "train_args": train_args,
        "status": "configured",
    }
    dataset_manifest = data_yaml.parent / "manifest.csv"
    if dataset_manifest.is_file():
        config["dataset_manifest"] = str(dataset_manifest.relative_to(REPO_ROOT))
        config["dataset_manifest_sha256"] = sha256(dataset_manifest)
    (run_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model = YOLO(weights)
    if args.freeze_bn_stats:
        def freeze_batch_norm_stats(trainer: object) -> None:
            for module in trainer.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

        # Ultralytics calls model.train() after on_train_epoch_start, so enforce
        # eval mode after that transition at the first and every subsequent batch.
        model.add_callback("on_train_batch_start", freeze_batch_norm_stats)
    model.train(**train_args)
    results = run_dir / "results.csv"
    best = run_dir / "weights/best.pt"
    config.update(
        {
            "status": "complete",
            "results_sha256": sha256(results),
            "best_checkpoint_sha256": sha256(best),
        }
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
