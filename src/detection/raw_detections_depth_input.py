#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from holod3.checkpoints import load_state_checkpoint

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass


def natural_key(path: str | Path) -> list[Any]:
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def build_image_map(image_dir: Path) -> dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return {p.name: p for p in sorted(paths, key=natural_key)}


def load_gray_image(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return gray


class AsyncCsvRowsWriter:
    def __init__(self, path: Path, fieldnames: list[str], *, max_queue: int) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.queue: queue.Queue[list[dict[str, Any]] | None] = queue.Queue(maxsize=max(1, max_queue))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="roi-csv-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                while True:
                    rows = self.queue.get()
                    try:
                        if rows is None:
                            break
                        writer.writerows(rows)
                    finally:
                        self.queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self.error is not None:
            raise RuntimeError("ROI CSV writer thread failed") from self.error
        self.queue.put(rows)

    def close(self) -> float:
        started = time.perf_counter()
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("ROI CSV writer thread failed") from self.error
        return time.perf_counter() - started


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoiFeatNet(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            DSConv(16, 24, stride=2),
            DSConv(24, 32, stride=2),
            DSConv(32, 48, stride=2),
            DSConv(48, 64, stride=2),
            DSConv(64, 96, stride=2),
        )
        self.proj = nn.Sequential(
            nn.Linear(96, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.08),
        )
        self.embedding_head = nn.Linear(96, embedding_dim)
        self.area_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).mean(dim=(2, 3))
        hidden = self.proj(feat)
        emb = F.normalize(self.embedding_head(hidden), dim=1)
        log_area_pred = self.area_head(hidden).squeeze(1)
        return emb, log_area_pred


def load_roifeat_model(weights: Path, device: torch.device) -> RoiFeatNet:
    ckpt = load_state_checkpoint(weights, map_location=device)
    state = ckpt["model"]
    embedding_dim = int(state["embedding_head.weight"].shape[0])
    model = RoiFeatNet(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def fixed_crop_array(arr: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    h, w = arr.shape
    x0 = int(round(cx - crop_size / 2))
    y0 = int(round(cy - crop_size / 2))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    return arr[y0:y1, x0:x1]


def focus_center_array(crop: np.ndarray, bw: float, bh: float, crop_size: int, pad_ratio: float) -> np.ndarray:
    pad = max(4.0, pad_ratio * max(bw, bh))
    x0 = max(0, int(round(crop_size / 2 - bw / 2 - pad)))
    y0 = max(0, int(round(crop_size / 2 - bh / 2 - pad)))
    x1 = min(crop_size, int(round(crop_size / 2 + bw / 2 + pad)))
    y1 = min(crop_size, int(round(crop_size / 2 + bh / 2 + pad)))
    out = np.full_like(crop, int(np.median(crop)))
    out[y0:y1, x0:x1] = crop[y0:y1, x0:x1]
    return out


CONTRAST_AREA_FRACTION = 0.70
CONTRAST_AREA_BG_PERCENTILE = 90.0
CONTRAST_AREA_PEAK_PERCENTILE = 95.0
CONTRAST_AREA_CORE_RADIUS_SCALE = 0.30
CONTRAST_AREA_ROI_SCALE = 1.8
CONTRAST_AREA_ROI_MARGIN_PX = 4.0
CONTRAST_AREA_MIN_ROI_RADIUS_PX = 7.0
CONTRAST_AREA_FULL_GATE_PX = 2.0


def component_for_particle(
    binary: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    max_area_fraction: float = 0.70,
):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if nlabels <= 1:
        return None

    cx = int(round(center_x))
    cy = int(round(center_y))
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        label = int(labels[cy, cx])
        if label > 0:
            area = float(stats[label, cv2.CC_STAT_AREA])
            if 2 <= area <= binary.size * max_area_fraction:
                return label, stats[label], centroids[label]

    best = None
    best_score = None
    roi_area = binary.shape[0] * binary.shape[1]
    for label in range(1, nlabels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > roi_area * max_area_fraction:
            continue
        mx, my = centroids[label]
        dist2 = (mx - center_x) ** 2 + (my - center_y) ** 2
        score = dist2 / max(area, 1.0)
        if best_score is None or score < best_score:
            best = (label, stats[label], centroids[label])
            best_score = score
    return best


def measure_particle(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float, xc: float, yc: float):
    h, w = gray.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    bbox_diameter_px = max(bw, bh)
    radius = max(
        CONTRAST_AREA_MIN_ROI_RADIUS_PX,
        0.5 * bbox_diameter_px * CONTRAST_AREA_ROI_SCALE + CONTRAST_AREA_ROI_MARGIN_PX,
    )
    crop_radius = int(math.ceil(radius))

    rx1 = max(0, int(math.floor(xc)) - crop_radius)
    ry1 = max(0, int(math.floor(yc)) - crop_radius)
    rx2 = min(w, int(math.floor(xc)) + crop_radius + 1)
    ry2 = min(h, int(math.floor(yc)) + crop_radius + 1)

    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_empty_roi", np.nan, 0

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    local_cx = xc - rx1
    local_cy = yc - ry1

    yy, xx = np.indices(blur.shape)
    rr = np.hypot(xx - local_cx, yy - local_cy)
    background = float(np.percentile(blur, CONTRAST_AREA_BG_PERCENTILE))
    darkness = np.clip(background - blur.astype(np.float32), 0.0, None)
    core = darkness[rr <= max(2.0, bbox_diameter_px * CONTRAST_AREA_CORE_RADIUS_SCALE)]
    if core.size == 0:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_no_core", np.nan, 0

    peak = float(np.percentile(core, CONTRAST_AREA_PEAK_PERCENTILE))
    if peak <= 1e-6:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_no_contrast", np.nan, 0

    binary = (darkness >= CONTRAST_AREA_FRACTION * peak).astype(np.uint8) * 255
    selected = component_for_particle(binary, local_cx, local_cy)
    if selected is None:
        return bbox_diameter_px, xc, yc, rx1, ry1, rx2, ry2, "bbox_fallback_no_component", np.nan, 0

    _label, stats, centroid = selected
    area = float(stats[cv2.CC_STAT_AREA])
    diameter_px = 2.0 * math.sqrt(area / math.pi)
    seg_x = rx1 + float(centroid[0])
    seg_y = ry1 + float(centroid[1])
    if math.hypot(seg_x - xc, seg_y - yc) > CONTRAST_AREA_FULL_GATE_PX:
        return (
            bbox_diameter_px,
            xc,
            yc,
            rx1,
            ry1,
            rx2,
            ry2,
            "contrast_area_f70_bg90_full_gate2px_bbox_fallback",
            np.nan,
            0,
        )
    return diameter_px, seg_x, seg_y, rx1, ry1, rx2, ry2, "contrast_area_f70_bg90_component", np.nan, int(area)


def flush_roifeat_batch(
    model: RoiFeatNet,
    batch: list[np.ndarray],
    features: list[np.ndarray],
    device: torch.device,
) -> None:
    if not batch:
        return
    xb = torch.from_numpy(np.stack(batch, axis=0)).to(device, non_blocking=True)
    with torch.inference_mode():
        z, _ = model(xb)
    features.append(z.detach().cpu().numpy().astype(np.float32))
    batch.clear()


def build_depth_input(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    out_csv = args.out_csv
    ensure_dir(out_csv.parent)
    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} exists. Use --overwrite.")
    if args.roifeat_output and args.roifeat_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.roifeat_output} exists. Use --overwrite.")

    df = pd.read_csv(args.raw_csv)
    required = {"frame", "file", "conf", "x1", "y1", "x2", "y2", "xc", "yc", "w", "h", "img_w", "img_h"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{args.raw_csv} is missing required columns: {missing}")
    df = df[df["conf"] >= args.min_conf].copy()
    if args.start_frame is not None:
        df = df[df["frame"] >= args.start_frame].copy()
    if args.end_frame is not None:
        df = df[df["frame"] <= args.end_frame].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering")

    if "raw_det_id" in df.columns:
        order_col = "raw_det_id"
    elif "raw_row_id" in df.columns:
        order_col = "raw_row_id"
    elif "row_id" in df.columns:
        order_col = "row_id"
    else:
        df["_input_order"] = np.arange(len(df), dtype=np.int64)
        order_col = "_input_order"
    df = df.sort_values(["frame", order_col]).reset_index(drop=True)

    image_map = build_image_map(args.image_dir)
    model = None
    features: list[np.ndarray] = []
    feature_batch: list[np.ndarray] = []
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if args.roifeat_output:
        if args.roifeat_weights is None:
            raise ValueError("--roifeat-weights is required when --roifeat-output is set.")
        model = load_roifeat_model(args.roifeat_weights, device)

    source_cols = [c for c in df.columns if c != "row_id"]
    extra_cols = [
        "row_id",
        "diameter_px",
        "diameter_um",
        "seg_xc",
        "seg_yc",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "diameter_method",
        "otsu_threshold",
        "seg_area_px",
    ]
    if args.assign_track_ids:
        fieldnames = source_cols[:2] + ["track_id"] + source_cols[2:] + extra_cols
    else:
        fieldnames = source_cols + extra_cols
    row_count = 0
    bbox_fallback_count = 0
    rejected_bbox_fallback_count = 0
    csv_writer_close_sec = 0.0
    writer = AsyncCsvRowsWriter(out_csv, fieldnames, max_queue=args.stream_write_queue_size) if args.stream_csv_writes else None
    direct_csv = None
    direct_writer = None
    if writer is None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        direct_csv = out_csv.open("w", newline="", encoding="utf-8")
        direct_writer = csv.DictWriter(direct_csv, fieldnames=fieldnames)
        direct_writer.writeheader()

    groups = list(df.groupby("file", sort=False))
    prefetch_workers = max(0, int(args.image_prefetch_workers))
    prefetch_frames = max(1, int(args.image_prefetch_frames))
    executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="roi-image-prefetch") if prefetch_workers > 0 else None
    futures: dict[int, Future[np.ndarray]] = {}

    def image_path_for(file_name: str) -> Path:
        img_path = image_map.get(str(file_name))
        if img_path is None:
            raise FileNotFoundError(f"Image not found for file={file_name} in {args.image_dir}")
        return img_path

    def submit_until(next_index: int) -> None:
        if executor is None:
            return
        stop = min(len(groups), next_index + prefetch_frames)
        for idx in range(next_index, stop):
            if idx not in futures:
                futures[idx] = executor.submit(load_gray_image, image_path_for(str(groups[idx][0])))

    try:
        submit_until(0)
        for group_index, (file_name, group) in enumerate(
            tqdm(groups, desc="raw ROI measure", unit="frame")
        ):
            submit_until(group_index + 1)
            if executor is None:
                gray = load_gray_image(image_path_for(str(file_name)))
            else:
                gray = futures.pop(group_index).result()

            frame_rows: list[dict[str, Any]] = []
            for row in group.itertuples(index=False):
                raw = row._asdict()
                diameter_px, seg_x, seg_y, rx1, ry1, rx2, ry2, method, otsu_threshold, area = measure_particle(
                    gray,
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                    float(raw["xc"]),
                    float(raw["yc"]),
                )
                is_bbox_fallback = "bbox_fallback" in method.lower()
                if is_bbox_fallback:
                    bbox_fallback_count += 1
                if is_bbox_fallback and not args.bbox_fallback:
                    rejected_bbox_fallback_count += 1
                    continue
                out = {col: raw[col] for col in source_cols}
                out["row_id"] = row_count
                if args.assign_track_ids:
                    out["track_id"] = row_count + 1
                    out["file"] = f"{Path(str(out['file'])).stem}.png"
                out["diameter_px"] = f"{diameter_px:.6f}"
                out["diameter_um"] = f"{diameter_px * args.pixel_pitch_um:.6f}"
                out["seg_xc"] = f"{seg_x:.6f}"
                out["seg_yc"] = f"{seg_y:.6f}"
                out["roi_x1"] = int(rx1)
                out["roi_y1"] = int(ry1)
                out["roi_x2"] = int(rx2)
                out["roi_y2"] = int(ry2)
                out["diameter_method"] = method
                out["otsu_threshold"] = f"{otsu_threshold:.6f}" if np.isfinite(otsu_threshold) else ""
                out["seg_area_px"] = int(area)
                frame_rows.append(out)
                row_count += 1

                if model is not None:
                    crop = fixed_crop_array(gray, float(raw["xc"]), float(raw["yc"]), args.crop_size)
                    crop = focus_center_array(crop, float(raw["w"]), float(raw["h"]), args.crop_size, args.focus_pad_ratio)
                    feature_batch.append((crop.astype(np.float32) / 255.0)[None, :, :])
                    if len(feature_batch) >= args.feature_batch_size:
                        flush_roifeat_batch(model, feature_batch, features, device)

            if writer is not None:
                writer.write_rows(frame_rows)
            elif direct_writer is not None:
                direct_writer.writerows(frame_rows)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if writer is not None:
        csv_writer_close_sec = writer.close()
    if direct_csv is not None:
        direct_csv.close()

    if model is not None:
        flush_roifeat_batch(model, feature_batch, features, device)
        emb = l2_normalize(np.concatenate(features, axis=0).astype(np.float32))
        if len(emb) != row_count:
            raise RuntimeError(f"ROIFeat row count mismatch: {len(emb)} features for {row_count} rows")
        ensure_dir(args.roifeat_output.parent)
        np.save(args.roifeat_output, emb)

    summary = {
        "raw_csv": str(args.raw_csv),
        "image_dir": str(args.image_dir),
        "out_csv": str(out_csv),
        "roifeat_output": str(args.roifeat_output) if args.roifeat_output else None,
        "rows": int(row_count),
        "frame_min": int(df["frame"].min()),
        "frame_max": int(df["frame"].max()),
        "frame_count": int(df["frame"].nunique()),
        "min_conf": float(args.min_conf),
        "bbox_fallback_enabled": bool(args.bbox_fallback),
        "bbox_fallback_rows": int(bbox_fallback_count),
        "rejected_bbox_fallback_rows": int(rejected_bbox_fallback_count),
        "assign_track_ids": bool(args.assign_track_ids),
        "image_prefetch_workers": int(args.image_prefetch_workers),
        "image_prefetch_frames": int(args.image_prefetch_frames),
        "stream_csv_writes": bool(args.stream_csv_writes),
        "csv_writer_close_sec": float(csv_writer_close_sec),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    summary_path = args.summary_json or out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fused learned-depth input directly from YOLO raw detections.")
    p.add_argument("--raw-csv", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--min-conf", type=float, default=0.10)
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--pixel-pitch-um", type=float, default=10.0)
    p.add_argument(
        "--bbox-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep YOLO bbox measurements when contrast-area measurement fails. Disabling this rejects those rows.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--roifeat-output", type=Path, default=None)
    p.add_argument("--roifeat-weights", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--feature-batch-size", type=int, default=4096)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--focus-pad-ratio", type=float, default=0.30)
    p.add_argument("--assign-track-ids", action="store_true", help="Emit the same per-detection track_id columns as add_detection_track_ids.py.")
    p.add_argument("--image-prefetch-workers", type=int, default=0, help="Thread workers for grayscale image prefetch during ROI measurement.")
    p.add_argument("--image-prefetch-frames", type=int, default=16, help="Maximum ROI image frames to keep queued ahead.")
    p.add_argument("--stream-csv-writes", action="store_true", help="Write ROI rows from a background thread.")
    p.add_argument("--stream-write-queue-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    build_depth_input(parse_args())


if __name__ == "__main__":
    main()
