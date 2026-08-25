#!/usr/bin/env python3
"""Apply the production diameter-underprediction fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the validated large-particle diameter underprediction fallback."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--ratio-threshold", type=float, default=0.35)
    parser.add_argument("--min-bbox-side-um", type=float, default=250.0)
    parser.add_argument("--pixel-pitch-um", type=float, default=10.0)
    parser.add_argument("--slice-spacing-um", type=float, default=100.0)
    parser.add_argument("--reconstruction-start-um", type=float, default=0.0)
    parser.add_argument(
        "--diameter-underprediction-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    frame = pd.read_csv(args.input)
    required = {"w", "h", "diameter_um", "slice_diam_pred_um", "seg_xc", "seg_yc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{args.input} is missing required columns: {missing}")

    if "diameter_px" in frame.columns:
        diameter_px = frame["diameter_px"].to_numpy(float)
        pitch = np.where(
            diameter_px > 0,
            frame["diameter_um"].to_numpy(float) / diameter_px,
            args.pixel_pitch_um,
        )
    else:
        pitch = np.full(len(frame), args.pixel_pitch_um, dtype=np.float64)
    bbox_average_um = 0.5 * (frame["w"].to_numpy(float) + frame["h"].to_numpy(float)) * pitch
    prediction = frame["slice_diam_pred_um"].to_numpy(float)
    minip_diameter = frame["diameter_um"].to_numpy(float)
    fallback = (
        args.diameter_underprediction_fallback
        & (bbox_average_um >= args.min_bbox_side_um)
        & (prediction < args.ratio_threshold * bbox_average_um)
    )

    output = frame.copy()
    output["bbox_avg_side_um"] = bbox_average_um
    output["diameter_fallback_to_contrast_area"] = fallback.astype(np.int8)
    output["final_diameter_um"] = np.where(fallback, minip_diameter, prediction)
    output["final_diameter_source"] = np.where(fallback, "contrast_area_fallback", "slice_diammodel")
    output["final_diameter_rule"] = (
        (
            f"bbox_avg_side_um >= {args.min_bbox_side_um:g} and "
            f"slice_pred < {args.ratio_threshold:g} * bbox_avg_side_um"
        )
        if args.diameter_underprediction_fallback
        else "disabled"
    )
    output["x_um"] = output["seg_xc"].astype(float) * args.pixel_pitch_um
    output["y_um"] = output["seg_yc"].astype(float) * args.pixel_pitch_um
    if "slice" in output.columns:
        output["z_um"] = (output["slice"].astype(float) - 1.0) * args.slice_spacing_um
        output["depth_um"] = args.reconstruction_start_um + output["z_um"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metrics = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(output)),
        "enabled": bool(args.diameter_underprediction_fallback),
        "ratio_threshold": float(args.ratio_threshold),
        "min_bbox_side_um": float(args.min_bbox_side_um),
        "pixel_pitch_um": float(args.pixel_pitch_um),
        "slice_spacing_um": float(args.slice_spacing_um),
        "reconstruction_start_um": float(args.reconstruction_start_um),
        "fallback_count": int(fallback.sum()),
        "fallback_fraction": float(fallback.mean()) if len(fallback) else 0.0,
    }
    metrics_path = args.metrics_output or args.output.with_suffix(".hybrid_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
