#!/usr/bin/env python3
"""Evaluate YOLO -> DepthModel -> SliceDiamModel output against synthetic truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402


BINS = ((25.0, 60.0), (60.0, 160.0), (160.0, 250.0), (250.0, 400.0), (400.0, 500.000001))
BIN_NAMES = ("25-60", "60-160", "160-250", "250-400", "400-500")


def summarize_errors(values: np.ndarray, *, threshold: float | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": int(values.size),
        "mae": float(np.mean(np.abs(values))),
        "median_ae": float(np.median(np.abs(values))),
        "p90_ae": float(np.percentile(np.abs(values), 90)),
        "p95_ae": float(np.percentile(np.abs(values), 95)),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }
    if threshold is not None:
        out[f"within_{threshold:g}"] = float(np.mean(np.abs(values) <= threshold))
    return out


def intersection_metrics(truth: pd.Series, pred: pd.Series) -> tuple[float, float, float]:
    tx0, ty0, tx1, ty1 = (float(truth[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
    px0, py0, px1, py1 = (float(pred[k]) for k in ("x1", "y1", "x2", "y2"))
    iw = max(0.0, min(tx1, px1) - max(tx0, px0))
    ih = max(0.0, min(ty1, py1) - max(ty0, py0))
    inter = iw * ih
    truth_area = max((tx1 - tx0) * (ty1 - ty0), 1e-9)
    pred_area = max((px1 - px0) * (py1 - py0), 1e-9)
    union = truth_area + pred_area - inter
    return inter / max(union, 1e-9), inter / truth_area, inter / pred_area


def associate_counts(truth: pd.DataFrame, predictions: pd.DataFrame, gate_px: float) -> np.ndarray:
    """Count predictions associated with every truth without Python row-pair loops."""
    if truth.empty or predictions.empty:
        return np.zeros(len(truth), dtype=np.int64)
    txy = truth[["x_px", "y_px"]].to_numpy(np.float64)
    pxy = predictions[["xc", "yc"]].to_numpy(np.float64)
    distance = np.linalg.norm(txy[:, None, :] - pxy[None, :, :], axis=2)
    radius_gate = np.maximum(gate_px, 0.5 * truth["diameter_px"].to_numpy(np.float64))
    associated = distance <= radius_gate[:, None]

    tx0 = truth["bbox_x0"].to_numpy(np.float64)[:, None]
    ty0 = truth["bbox_y0"].to_numpy(np.float64)[:, None]
    tx1 = truth["bbox_x1"].to_numpy(np.float64)[:, None]
    ty1 = truth["bbox_y1"].to_numpy(np.float64)[:, None]
    px0 = predictions["x1"].to_numpy(np.float64)[None, :]
    py0 = predictions["y1"].to_numpy(np.float64)[None, :]
    px1 = predictions["x2"].to_numpy(np.float64)[None, :]
    py1 = predictions["y2"].to_numpy(np.float64)[None, :]
    intersection = np.maximum(0.0, np.minimum(tx1, px1) - np.maximum(tx0, px0)) * np.maximum(
        0.0, np.minimum(ty1, py1) - np.maximum(ty0, py0)
    )
    prediction_area = np.maximum((px1 - px0) * (py1 - py0), 1e-9)
    associated |= intersection / prediction_area >= 0.25
    return np.count_nonzero(associated, axis=1).astype(np.int64)


def match_frame(truth: pd.DataFrame, predictions: pd.DataFrame, gate_px: float) -> list[tuple[int, int, float]]:
    if truth.empty or predictions.empty:
        return []
    txy = truth[["x_px", "y_px"]].to_numpy(np.float64)
    pxy = predictions[["xc", "yc"]].to_numpy(np.float64)
    dist = np.linalg.norm(txy[:, None, :] - pxy[None, :, :], axis=2)
    cost = dist.copy()
    cost[cost > gate_px] = 1e9
    ti, pi = linear_sum_assignment(cost)
    return [(int(t), int(p), float(dist[t, p])) for t, p in zip(ti, pi, strict=True) if dist[t, p] <= gate_px]


def load_truth(dataset: Path) -> pd.DataFrame:
    truth_dir = dataset / "truth" if (dataset / "truth").is_dir() else dataset / "labels"
    particles = pd.read_csv(truth_dir / "particles.csv")
    rois = pd.read_csv(truth_dir / "xy_rois.csv")
    keys = ["frame", "particle_id"]
    keep = [
        *keys,
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "bbox_w",
        "bbox_h",
    ]
    truth = particles.merge(rois[keep], on=keys, how="inner", validate="one_to_one")
    if "center_in_volume" in truth:
        truth = truth.loc[truth["center_in_volume"].astype(int) == 1].copy()
    return truth.sort_values(["frame", "particle_id"]).reset_index(drop=True)


def evaluate(
    dataset: Path,
    pipeline_csv: Path,
    gate_px: float,
    dz_um: float,
    frame_stats_csv: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = load_truth(dataset)
    predictions = pd.read_csv(pipeline_csv)
    if frame_stats_csv is None:
        candidate = pipeline_csv.parent / "frame_stats.csv"
        frame_stats_csv = candidate if candidate.is_file() else None
    evaluated_frames: list[int] | None = None
    if frame_stats_csv is not None:
        frame_stats = pd.read_csv(frame_stats_csv)
        if "frame" not in frame_stats:
            raise ValueError(f"frame column is missing from {frame_stats_csv}")
        evaluated_frames = sorted({int(value) for value in frame_stats["frame"]})
        truth = truth.loc[truth["frame"].astype(int).isin(evaluated_frames)].copy()
        predictions = predictions.loc[
            predictions["frame"].astype(int).isin(evaluated_frames)
        ].copy()
    rows: list[dict[str, Any]] = []
    duplicate_truths = 0
    association_counts: list[int] = []
    for frame, frame_truth in truth.groupby("frame", sort=True):
        frame_pred = predictions.loc[predictions["frame"].astype(int) == int(frame)].reset_index(drop=True)
        frame_truth = frame_truth.reset_index(drop=True)
        frame_association_counts = associate_counts(frame_truth, frame_pred, gate_px)
        association_counts.extend(int(count) for count in frame_association_counts)
        duplicate_truths += int(np.count_nonzero(frame_association_counts > 1))
        for ti, pi, distance in match_frame(frame_truth, frame_pred, gate_px):
            t = frame_truth.iloc[ti]
            p = frame_pred.iloc[pi]
            iou, truth_coverage, pred_coverage = intersection_metrics(t, p)
            truth_side = max(float(t["bbox_w"]), float(t["bbox_h"]))
            pred_side = max(float(p["w"]), float(p["h"]))
            pred_slice = float(p["slice"])
            rows.append(
                {
                    "frame": int(frame),
                    "file": t["file"],
                    "particle_id": int(t["particle_id"]),
                    "pred_row_id": int(p.get("row_id", pi)),
                    "conf": float(p["conf"]),
                    "center_error_px": distance,
                    "bbox_iou": iou,
                    "truth_box_coverage": truth_coverage,
                    "pred_box_coverage": pred_coverage,
                    "pred_truth_side_ratio": pred_side / max(truth_side, 1e-9),
                    "truth_x_px": float(t["x_px"]),
                    "truth_y_px": float(t["y_px"]),
                    "truth_z_um": float(t["z_um"]),
                    "truth_z_slice": float(t["z_slice"]),
                    "truth_diameter_um": float(t["diameter_um"]),
                    "truth_diameter_px": float(t["diameter_px"]),
                    "pred_slice": pred_slice,
                    "depth_error_um": (pred_slice - float(t["z_slice"])) * dz_um,
                    "contrast_diameter_um": float(p.get("diameter_um", np.nan)),
                    "contrast_diameter_error_um": float(p.get("diameter_um", np.nan)) - float(t["diameter_um"]),
                    "slice_diameter_um": float(p.get("slice_diam_pred_um", np.nan)),
                    "slice_diameter_error_um": float(p.get("slice_diam_pred_um", np.nan)) - float(t["diameter_um"]),
                    "final_diameter_um": float(p.get("final_diameter_um", np.nan)),
                    "final_diameter_error_um": float(p.get("final_diameter_um", np.nan)) - float(t["diameter_um"]),
                }
            )
    matched = pd.DataFrame(rows)
    truth_count = len(truth)
    pred_count = len(predictions)
    match_count = len(matched)
    recall = match_count / truth_count if truth_count else 0.0
    precision = match_count / pred_count if pred_count else 0.0
    summary: dict[str, Any] = {
        "dataset": str(dataset.resolve()),
        "pipeline_csv": str(pipeline_csv.resolve()),
        "frame_stats_csv": str(frame_stats_csv.resolve()) if frame_stats_csv is not None else None,
        "evaluated_frames": evaluated_frames,
        "evaluated_frame_count": len(evaluated_frames) if evaluated_frames is not None else None,
        "center_gate_px": gate_px,
        "truth_count": truth_count,
        "prediction_count": pred_count,
        "matched_count": match_count,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "duplicate_truth_count": duplicate_truths,
        "duplicate_truth_fraction": duplicate_truths / truth_count if truth_count else 0.0,
        "mean_predictions_associated_per_truth": float(np.mean(association_counts)) if association_counts else 0.0,
    }
    if not matched.empty:
        summary["roi"] = {
            "mean_iou": float(matched["bbox_iou"].mean()),
            "median_iou": float(matched["bbox_iou"].median()),
            "mean_truth_coverage": float(matched["truth_box_coverage"].mean()),
            "median_side_ratio": float(matched["pred_truth_side_ratio"].median()),
            "under_box_fraction_ratio_lt_0p8": float((matched["pred_truth_side_ratio"] < 0.8).mean()),
            "over_box_fraction_ratio_gt_1p2": float((matched["pred_truth_side_ratio"] > 1.2).mean()),
        }
        depth_error = matched["depth_error_um"].to_numpy(float)
        summary["depth"] = summarize_errors(depth_error, threshold=1000.0)
        summary["depth"]["catastrophic_gt_5000um"] = float(np.mean(np.abs(depth_error) > 5000.0))
        summary["diameter"] = {
            "contrast": summarize_errors(matched["contrast_diameter_error_um"].to_numpy(float)),
            "slice": summarize_errors(matched["slice_diameter_error_um"].to_numpy(float)),
            "final": summarize_errors(matched["final_diameter_error_um"].to_numpy(float)),
        }
    by_bin: list[dict[str, Any]] = []
    for name, (lo, hi) in zip(BIN_NAMES, BINS, strict=True):
        truth_bin = truth.loc[(truth["diameter_um"] >= lo) & (truth["diameter_um"] < hi)]
        matched_bin = matched.loc[
            (matched["truth_diameter_um"] >= lo) & (matched["truth_diameter_um"] < hi)
        ] if not matched.empty else matched
        row: dict[str, Any] = {
            "diameter_bin_um": name,
            "truth_count": int(len(truth_bin)),
            "matched_count": int(len(matched_bin)),
            "recall": float(len(matched_bin) / len(truth_bin)) if len(truth_bin) else math.nan,
        }
        if len(matched_bin):
            row.update(
                {
                    "mean_iou": float(matched_bin["bbox_iou"].mean()),
                    "median_side_ratio": float(matched_bin["pred_truth_side_ratio"].median()),
                    "under_box_fraction": float((matched_bin["pred_truth_side_ratio"] < 0.8).mean()),
                    "depth_mae_um": float(np.abs(matched_bin["depth_error_um"]).mean()),
                    "depth_gt5mm_fraction": float((np.abs(matched_bin["depth_error_um"]) > 5000).mean()),
                    "slice_diam_mae_um": float(np.abs(matched_bin["slice_diameter_error_um"]).mean()),
                    "final_diam_mae_um": float(np.abs(matched_bin["final_diameter_error_um"]).mean()),
                }
            )
        by_bin.append(row)
    summary["by_diameter"] = by_bin
    return matched, summary


def plot_by_bin(by_bin: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(by_bin))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(x, by_bin["recall"])
    axes[0].set_ylabel("YOLO recall")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, by_bin.get("depth_mae_um", np.nan))
    axes[1].set_ylabel("Depth MAE [um]")
    axes[2].plot(x, by_bin.get("slice_diam_mae_um", np.nan), "o-", label="slice")
    axes[2].plot(x, by_bin.get("final_diam_mae_um", np.nan), "s-", label="final")
    axes[2].set_ylabel("Diameter MAE [um]")
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(x, by_bin["diameter_bin_um"], rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_metrics_by_diameter.png", dpi=170)
    plt.close(fig)


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    depth = summary.get("depth", {})
    diam = summary.get("diameter", {})
    lines = [
        "# Three-model end-to-end benchmark",
        "",
        f"- truth: {summary['truth_count']:,}",
        f"- prediction: {summary['prediction_count']:,}",
        f"- YOLO recall / precision / F1: {summary['recall']:.4f} / {summary['precision']:.4f} / {summary['f1']:.4f}",
        f"- Fraction of truth particles associated with multiple predictions: {summary['duplicate_truth_fraction']:.4f}",
        f"- matched ROI mean IoU: {summary.get('roi', {}).get('mean_iou', float('nan')):.4f}",
        f"- Depth MAE: {depth.get('mae', float('nan')):.2f} um",
        f"- Depth error over 5 mm: {depth.get('catastrophic_gt_5000um', float('nan')):.4f}",
        f"- SliceDiam MAE: {diam.get('slice', {}).get('mae', float('nan')):.2f} um",
        f"- final diameter MAE: {diam.get('final', {}).get('mae', float('nan')):.2f} um",
        "",
        "![End-to-end metrics by diameter](pipeline_metrics_by_diameter.png)",
        "",
        "## By diameter",
        "",
        "| bin [um] | truth | recall | IoU | side ratio | under-box | depth MAE [um] | depth >5mm | slice diam MAE [um] | final diam MAE [um] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_diameter"]:
        lines.append(
            "| {diameter_bin_um} | {truth_count} | {recall:.4f} | {mean_iou:.4f} | "
            "{median_side_ratio:.4f} | {under_box_fraction:.4f} | {depth_mae_um:.1f} | "
            "{depth_gt5mm_fraction:.4f} | {slice_diam_mae_um:.2f} | {final_diam_mae_um:.2f} |".format(
                **{key: row.get(key, float("nan")) for key in (
                    "diameter_bin_um", "truth_count", "recall", "mean_iou", "median_side_ratio",
                    "under_box_fraction", "depth_mae_um", "depth_gt5mm_fraction", "slice_diam_mae_um",
                    "final_diam_mae_um",
                )}
            )
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pipeline-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--center-gate-px", type=float, default=8.0)
    parser.add_argument("--dz-um", type=float, default=100.0)
    parser.add_argument(
        "--frame-stats-csv",
        type=Path,
        default=None,
        help="Processed-frame manifest. Defaults to frame_stats.csv beside --pipeline-csv when present.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matched, summary = evaluate(
        args.dataset,
        args.pipeline_csv,
        args.center_gate_px,
        args.dz_um,
        args.frame_stats_csv,
    )
    matched.to_csv(args.out_dir / "matched_predictions.csv", index=False)
    by_bin = pd.DataFrame(summary["by_diameter"])
    by_bin.to_csv(args.out_dir / "metrics_by_diameter.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_by_bin(by_bin, args.out_dir)
    write_report(summary, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
