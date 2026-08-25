#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.yolo.dataset import REPO_ROOT, repo_path, repo_relative, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair Ultralytics validation comparison on a repo-local YOLO dataset.")
    parser.add_argument("--data", required=True, help="Repository-local YOLO data.yaml to evaluate.")
    parser.add_argument("--out-dir", required=True, help="Repository-local output directory below runs/.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=WEIGHTS",
        help="Repeat for every model; NAME must be a simple directory name.",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--model must be NAME=WEIGHTS: {value}")
    name, raw_path = value.split("=", 1)
    if not name or Path(name).name != name:
        raise ValueError(f"Model NAME must be a simple directory name: {name!r}")
    weights = repo_path(raw_path)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return name, weights


def native(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def write_report(path: Path, rows: list[dict[str, Any]], data_yaml: Path, args: argparse.Namespace) -> None:
    lines = [
        "# YOLO experimental-validation comparison",
        "",
        "## Conditions",
        "",
        f"- Data: `{repo_relative(data_yaml)}`",
        f"- imgsz: `{args.imgsz}`, batch: `{args.batch}`, confidence floor: `{args.conf:g}`",
        f"- NMS IoU: `{args.iou:g}`, max_det: `{args.max_det}`, half: `{args.half}`",
        "- Every model was freshly evaluated in this run with the same Ultralytics version and validation settings.",
        "",
        "## Results",
        "",
        "| model | precision | recall | mAP50 | mAP75 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['map50']:.4f} | {row['map75']:.4f} | {row['map50_95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "PR, F1, and confusion-matrix figures for each model are stored below `validation/<model>/`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_yaml = repo_path(args.data)
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    out_dir = repo_path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [parse_model(value) for value in args.model]
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names are not allowed")

    config_dir = out_dir / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.chdir(REPO_ROOT)
    from ultralytics import YOLO

    rows: list[dict[str, Any]] = []
    for name, weights in models:
        metrics = YOLO(str(weights)).val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            half=args.half,
            plots=True,
            project=str(out_dir / "validation"),
            name=name,
            exist_ok=False,
            verbose=False,
        )
        row = {
            "model": name,
            "weights": repo_relative(weights),
            "weights_sha256": sha256_file(weights),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "map50_95": float(metrics.box.map),
            "speed": native(metrics.speed),
            "results_dict": native(metrics.results_dict),
        }
        rows.append(row)
        (out_dir / f"{name}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": repo_relative(data_yaml),
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "half": args.half,
        },
        "models": rows,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_fields = [
        "model",
        "weights",
        "weights_sha256",
        "precision",
        "recall",
        "map50",
        "map75",
        "map50_95",
    ]
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in flat_fields} for row in rows)
    write_report(out_dir / "report.md", rows, data_yaml, args)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
