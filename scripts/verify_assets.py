#!/usr/bin/env python3
"""Verify production weights and downloaded training dataset bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from holod3.artifacts import load_production_manifest, sha256_file, verify_production_artifacts
from holod3.config import repository_root, resolve_asset_path


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in stream) - 1


def validate_yolo_lists(data_yaml: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = resolve_asset_path(str(payload["path"]))
    result: dict[str, Any] = {}
    for split in ("train", "val"):
        split_source = dataset_root / str(payload[split])
        if split_source.is_dir():
            images = sorted(
                path.resolve()
                for path in split_source.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            )
            entries = [path.relative_to(dataset_root).as_posix() for path in images]
        else:
            entries = [line.strip() for line in split_source.read_text(encoding="utf-8").splitlines() if line.strip()]
            images = [(dataset_root / value).resolve() for value in entries]
        missing_images: list[str] = []
        missing_labels: list[str] = []
        for value, image in zip(entries, images, strict=True):
            if not image.is_file():
                missing_images.append(value)
                continue
            label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
            if not label.is_file():
                missing_labels.append(str(label))
        result[split] = {
            "presentations": len(entries),
            "missing_images": missing_images,
            "missing_labels": missing_labels,
        }
    return result


def validate_crop_references(root: Path) -> dict[str, Any]:
    """Resolve every training crop through the same loaders used for training."""

    from src.depth.train_depth_compare import load_manifest
    from src.diam.train_slice_diammodel import load_rows

    primary = load_manifest(root / "data/downloaded/depth-primary/manifest.csv", repo_root=root)
    fallback = load_manifest(root / "data/downloaded/depth-fallback/manifest.csv", repo_root=root)
    diameter_root = root / "data/downloaded/diameter-combined"
    diameter = load_rows(diameter_root / "crops.csv", data_root=diameter_root)
    return {
        "depth_primary": {
            "rows": len(primary),
            "samples": int(primary["sample_id"].nunique()),
            "resolved_crop_files": int(primary["_crop_abs_path"].nunique()),
            "ok": len(primary) == 104_172 and primary["_crop_abs_path"].nunique() == 104_172,
        },
        "depth_fallback": {
            "rows": len(fallback),
            "samples": int(fallback["sample_id"].nunique()),
            "resolved_crop_files": int(fallback["_crop_abs_path"].nunique()),
            "ok": len(fallback) == 30_600 and fallback["_crop_abs_path"].nunique() == 30_600,
        },
        "diameter": {
            "rows": len(diameter),
            "resolved_crop_files": len({row.resolved_image_path for row in diameter}),
            "source_frames": len({row.source_frame_id for row in diameter}),
            "ok": len(diameter) == 14_080 and len({row.resolved_image_path for row in diameter}) == 14_080,
        },
    }


def validate_holdout(root: Path) -> dict[str, Any]:
    """Check the fixed holdout contract and synchronized frame names."""

    from holod3.acquisition import AcquisitionConfig

    benchmark = root / "data/downloaded/evaluation-benchmark"
    required = [
        benchmark / "acquisition.yaml",
        benchmark / "calibration/secondary_distortion_coefficients.txt",
        benchmark / "truth/particles.csv",
        benchmark / "truth/xy_rois.csv",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    records = AcquisitionConfig.load(benchmark / "acquisition.yaml").frame_records() if not missing else []
    frame_count = len(records)
    return {
        "path": str(benchmark.relative_to(root)),
        "frame_count": frame_count,
        "frame_names_aligned": bool(records),
        "missing": missing,
        "particle_truth_rows": csv_rows(benchmark / "truth/particles.csv") if not missing else None,
        "ok": not missing and frame_count == 12,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all model and dataset inputs needed for production reproduction.")
    parser.add_argument("--full", action="store_true", help="Resolve every YOLO image and label reference.")
    args = parser.parse_args()
    root = repository_root()
    manifest = load_production_manifest()
    report: dict[str, Any] = {
        "models": [],
        "training_initializers": [],
        "training_inputs": [],
        "yolo": None,
        "crop_references": None,
        "holdout": None,
        "ok": True,
    }
    for status in verify_production_artifacts(root=root, check_hash=True):
        report["models"].append(
            {
                "id": status.artifact_id,
                "path": str(status.path.relative_to(root)),
                "exists": status.exists,
                "size_matches": status.size_matches,
                "hash_matches": status.hash_matches,
                "ok": status.ok,
            }
        )
        report["ok"] &= status.ok
    for expected in manifest.get("training_initializers", []):
        path = resolve_asset_path(expected["path"], root)
        exists = path.is_file()
        size_matches = exists and path.stat().st_size == int(expected["bytes"])
        hash_matches = exists and sha256_file(path) == expected["sha256"]
        item = {
            "id": expected["id"],
            "path": expected["path"],
            "exists": exists,
            "size_matches": size_matches,
            "hash_matches": hash_matches,
            "ok": bool(exists and size_matches and hash_matches),
        }
        report["training_initializers"].append(item)
        report["ok"] &= item["ok"]
    for expected in manifest["training_inputs"]:
        path = resolve_asset_path(expected["path"], root)
        exists = path.is_file()
        hash_matches = exists and sha256_file(path) == expected["sha256"]
        actual_rows = csv_rows(path) if exists and "rows" in expected else None
        rows_match = actual_rows == expected.get("rows") if "rows" in expected else True
        item = {
            "id": expected["id"],
            "path": expected["path"],
            "exists": exists,
            "hash_matches": hash_matches,
            "rows": actual_rows,
            "rows_match": rows_match,
            "ok": bool(exists and hash_matches and rows_match),
        }
        report["training_inputs"].append(item)
        report["ok"] &= item["ok"]
    if args.full:
        report["yolo"] = {
            "experimental_base": validate_yolo_lists(root / "data/downloaded/detector-experimental/data.yaml"),
            "mixed_training": validate_yolo_lists(root / "data/downloaded/detector-mixed/data.yaml"),
        }
        for dataset in report["yolo"].values():
            for split in dataset.values():
                report["ok"] &= not split["missing_images"] and not split["missing_labels"]
        report["crop_references"] = validate_crop_references(root)
        for group in report["crop_references"].values():
            report["ok"] &= group["ok"]
        report["holdout"] = validate_holdout(root)
        report["ok"] &= report["holdout"]["ok"]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
