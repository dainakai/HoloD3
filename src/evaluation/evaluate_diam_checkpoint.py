#!/usr/bin/env python3
"""Evaluate a SliceDiamModel checkpoint on a repository-local truth-z crop split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diam.model import load_model  # noqa: E402
from src.diam.train_slice_diammodel import (  # noqa: E402
    audit_frame_splits,
    load_rows,
    portable_path,
    predict_rows,
    repo_path,
    resolve_data_root,
    sha256_file,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crops_csv = repo_path(args.crops_csv)
    data_root = resolve_data_root(args.data_root)
    weights = repo_path(args.weights)
    out_dir = repo_path(args.out_dir)
    allowed = (ROOT / "runs").resolve()
    if not out_dir.is_relative_to(allowed) or out_dir == allowed:
        raise ValueError("--out-dir must be a new child of runs/")
    if out_dir.exists():
        raise FileExistsError(f"Output exists: {out_dir}")
    rows = load_rows(crops_csv, data_root=data_root)
    split_audit = audit_frame_splits(rows)
    selected = rows if args.split == "all" else [row for row in rows if row.split == args.split]
    if not selected:
        raise ValueError(f"No rows for split={args.split}")
    device = torch.device(args.device)
    model, norm, calibration_scale, checkpoint = load_model(weights, device)
    out_dir.mkdir(parents=True)
    predictions = predict_rows(
        model,
        selected,
        device=device,
        batch_size=args.batch_size,
        norm=norm,
        calibration_scale=calibration_scale,
    )
    metrics = write_outputs(predictions, out_dir, args.split)
    metadata = {
        "status": "complete",
        "mode": "truth-z SliceDiamModel checkpoint evaluation",
        "weights": portable_path(weights),
        "weights_sha256": sha256_file(weights),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "crops_csv": portable_path(crops_csv),
        "crops_csv_sha256": sha256_file(crops_csv),
        "data_root": portable_path(data_root),
        "split": args.split,
        "rows": len(selected),
        "device": str(device),
        "split_audit": split_audit,
        "metrics": metrics,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
