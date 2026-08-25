from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def natural_key(path: str | Path) -> list[Any]:
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def image_stem_number(path: str | Path) -> int | None:
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else None


def resolve_image_selector(selector: str, img_paths: list[Path], label: str) -> int:
    value = os.path.basename(str(selector))
    stem, ext = os.path.splitext(value)

    matches: list[int] = []
    for idx, path in enumerate(img_paths):
        if value == path.name or (not ext and value == path.stem):
            matches.append(idx)

    if not matches and stem.isdigit():
        want = int(stem)
        matches = [idx for idx, path in enumerate(img_paths) if image_stem_number(path) == want]

    if not matches:
        raise ValueError(f"{label} image not found in sequence: {selector}")
    if len(matches) > 1:
        names = ", ".join(img_paths[i].name for i in matches[:5])
        raise ValueError(f"{label} image selector is ambiguous: {selector} matched {names}")
    return matches[0]


def selected_image_items(
    img_paths: list[Path],
    *,
    start: str | None,
    end: str | None,
    start_index: int | None,
    end_index: int | None,
    limit: int | None,
) -> list[tuple[int, Path]]:
    if start is not None and start_index is not None:
        raise ValueError("--start and --start-index cannot be used together")
    if end is not None and end_index is not None:
        raise ValueError("--end and --end-index cannot be used together")

    first = 0 if start_index is None else start_index
    last = len(img_paths) - 1 if end_index is None else end_index

    if start is not None:
        first = resolve_image_selector(start, img_paths, "--start")
    if end is not None:
        last = resolve_image_selector(end, img_paths, "--end")

    first = max(0, first)
    last = min(last, len(img_paths) - 1)
    if last < first:
        raise ValueError(f"Invalid image range: {first}..{last}")

    items = list(enumerate(img_paths))[first : last + 1]
    if limit is not None:
        items = items[:limit]
    return items


def contrast_stretch_0_75_to_255(img_u8: np.ndarray, in_max: int = 75) -> np.ndarray:
    x = img_u8.astype(np.float32)
    x = np.clip(x, 0, float(in_max)) * (255.0 / float(in_max))
    return x.astype(np.uint8)


def apply_contrast(img_bgr: np.ndarray, in_max: int = 75) -> np.ndarray:
    if img_bgr.ndim == 2:
        gray = img_bgr
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    stretched = contrast_stretch_0_75_to_255(gray, in_max=in_max)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def batched(items: list[tuple[int, Path]], batch_size: int) -> list[list[tuple[int, Path]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def load_and_prepare_image(item: tuple[int, Path], contrast_in_max: int) -> tuple[int, Path, np.ndarray, tuple[int, int]]:
    frame, img_path = item
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img
    shape = (int(img_bgr.shape[0]), int(img_bgr.shape[1]))
    return frame, img_path, apply_contrast(img_bgr, in_max=contrast_in_max), shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw YOLO detections with production-checkpoint-compatible preprocessing. "
            "Particle center and diameter are measured by the downstream HoloD3 stages."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--weights", default="models/production/detector.pt")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--frame-stats-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--start", default=None, help="First image filename/stem, inclusive.")
    parser.add_argument("--end", default=None, help="Last image filename/stem, inclusive.")
    parser.add_argument("--start-index", type=int, default=None, help="First zero-based sorted image index, inclusive.")
    parser.add_argument("--end-index", type=int, default=None, help="Last zero-based sorted image index, inclusive.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, cuda:0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--image-load-workers",
        type=int,
        default=0,
        help="Thread workers for image read + contrast preprocessing. 0 keeps sequential loading.",
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--contrast-in-max", type=int, default=75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    image_dir = Path(args.image_dir)
    weights = Path(args.weights)
    out_csv = Path(args.out_csv)
    stats_csv = Path(args.frame_stats_csv) if args.frame_stats_csv else out_csv.with_name("frame_stats.csv")
    summary_json = Path(args.summary_json) if args.summary_json else out_csv.with_name("raw_detection_summary.json")

    for path in (out_csv, stats_csv, summary_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=natural_key,
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    items = selected_image_items(
        img_paths,
        start=args.start,
        end=args.end,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )
    if not items:
        raise ValueError("No images selected.")

    device = args.device
    if str(device).startswith("cuda:"):
        device = str(device).split(":", 1)[1]
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(
        "Images: "
        f"{len(items)} selected from {len(img_paths)} "
        f"({items[0][0]}:{items[0][1].name} .. {items[-1][0]}:{items[-1][1].name})"
    )
    print(f"weights: {weights}")
    print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {device}")
    if torch.cuda.is_available() and str(device) != "cpu":
        dev_index = int(device) if str(device).isdigit() else 0
        print(f"GPU: {torch.cuda.get_device_name(dev_index)}")

    model = YOLO(str(weights))

    fieldnames = [
        "frame", "file",
        "raw_row_id", "raw_det_id",
        "class_id", "conf",
        "x1", "y1", "x2", "y2",
        "xc", "yc", "w", "h",
        "x1n", "y1n", "x2n", "y2n",
        "xcn", "ycn", "wn", "hn",
        "img_w", "img_h",
    ]
    stats_fieldnames = ["frame", "file", "img_w", "img_h", "n_detections", "elapsed_sec"]

    raw_row_id = 0
    total_dets = 0
    frame_counts: list[int] = []
    frame_elapsed: list[float] = []
    image_load_sec_total = 0.0
    predict_sec_total = 0.0
    csv_write_sec_total = 0.0

    with out_csv.open("w", newline="", encoding="utf-8") as det_f, stats_csv.open("w", newline="", encoding="utf-8") as stat_f:
        det_writer = csv.DictWriter(det_f, fieldnames=fieldnames)
        stat_writer = csv.DictWriter(stat_f, fieldnames=stats_fieldnames)
        det_writer.writeheader()
        stat_writer.writeheader()

        load_workers = max(0, int(args.image_load_workers))
        executor = ThreadPoolExecutor(max_workers=load_workers) if load_workers > 0 else None
        try:
            for chunk in tqdm(batched(items, max(1, args.batch_size)), desc="YOLO raw detect", unit="batch"):
                infer_images: list[np.ndarray] = []
                orig_shapes: list[tuple[int, int]] = []
                chunk_items: list[tuple[int, Path]] = []
                read_started = time.perf_counter()
                if executor is None:
                    prepared = [load_and_prepare_image(item, args.contrast_in_max) for item in chunk]
                else:
                    prepared = list(executor.map(lambda item: load_and_prepare_image(item, args.contrast_in_max), chunk))
                read_elapsed = time.perf_counter() - read_started
                image_load_sec_total += read_elapsed
                for frame, img_path, infer_img, shape in prepared:
                    chunk_items.append((frame, img_path))
                    orig_shapes.append(shape)
                    infer_images.append(infer_img)

                pred_started = time.perf_counter()
                results = model.predict(
                    source=infer_images,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    max_det=args.max_det,
                    half=args.half,
                    verbose=False,
                    save=False,
                    stream=False,
                    end2end=False,
                )
                pred_elapsed = time.perf_counter() - pred_started
                predict_sec_total += pred_elapsed
                write_started = time.perf_counter()
                per_frame_elapsed = (read_elapsed + pred_elapsed) / max(1, len(chunk))

                for (frame, img_path), result, (img_h, img_w) in zip(chunk_items, results, orig_shapes):
                    boxes = result.boxes
                    n_det = 0 if boxes is None else len(boxes)
                    frame_counts.append(int(n_det))
                    frame_elapsed.append(per_frame_elapsed)
                    stat_writer.writerow(
                        {
                            "frame": frame,
                            "file": img_path.name,
                            "img_w": img_w,
                            "img_h": img_h,
                            "n_detections": int(n_det),
                            "elapsed_sec": f"{per_frame_elapsed:.6f}",
                        }
                    )
                    if n_det == 0 or boxes is None:
                        continue

                    xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
                    xywh = boxes.xywh.detach().cpu().numpy().astype(float)
                    confs = boxes.conf.detach().cpu().numpy().astype(float)
                    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                    for raw_det_id, (bb_xyxy, bb_xywh, conf, cls_id) in enumerate(zip(xyxy, xywh, confs, cls_ids)):
                        x1, y1, x2, y2 = bb_xyxy.tolist()
                        xc, yc, bw, bh = bb_xywh.tolist()
                        det_writer.writerow(
                            {
                                "frame": int(frame),
                                "file": img_path.name,
                                "raw_row_id": raw_row_id,
                                "raw_det_id": int(raw_det_id),
                                "class_id": int(cls_id),
                                "conf": float(conf),
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "xc": float(xc),
                                "yc": float(yc),
                                "w": float(bw),
                                "h": float(bh),
                                "x1n": float(x1 / img_w),
                                "y1n": float(y1 / img_h),
                                "x2n": float(x2 / img_w),
                                "y2n": float(y2 / img_h),
                                "xcn": float(xc / img_w),
                                "ycn": float(yc / img_h),
                                "wn": float(bw / img_w),
                                "hn": float(bh / img_h),
                                "img_w": int(img_w),
                                "img_h": int(img_h),
                            }
                        )
                        raw_row_id += 1
                    total_dets += int(n_det)

                # Keep the progress bar responsive even when stdout is buffered.
                det_f.flush()
                stat_f.flush()
                csv_write_sec_total += time.perf_counter() - write_started
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    counts = np.asarray(frame_counts, dtype=np.int32)
    elapsed = time.perf_counter() - t0
    summary = {
        "weights": str(weights),
        "image_dir": str(image_dir),
        "out_csv": str(out_csv),
        "frame_stats_csv": str(stats_csv),
        "n_frames": int(len(items)),
        "frame_first": int(items[0][0]),
        "frame_last": int(items[-1][0]),
        "file_first": items[0][1].name,
        "file_last": items[-1][1].name,
        "total_detections": int(total_dets),
        "detections_per_frame": {
            "min": int(counts.min()) if len(counts) else 0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "mean": float(np.mean(counts)) if len(counts) else 0.0,
            "max": int(counts.max()) if len(counts) else 0,
        },
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "image_load_workers": int(args.image_load_workers),
        "half": bool(args.half),
        "contrast_in_max": int(args.contrast_in_max),
        "image_load_sec_total": float(image_load_sec_total),
        "predict_sec_total": float(predict_sec_total),
        "csv_write_sec_total": float(csv_write_sec_total),
        "image_load_sec_per_frame": float(image_load_sec_total / max(1, len(items))),
        "predict_sec_per_frame": float(predict_sec_total / max(1, len(items))),
        "csv_write_sec_per_frame": float(csv_write_sec_total / max(1, len(items))),
        "elapsed_sec": float(elapsed),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
