#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT_FOR_IMPORT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from holod3.acquisition import AcquisitionConfig  # noqa: E402
from holod3.reconstruction import prepare_minip_images  # noqa: E402
from src.pipeline.runtime import (  # noqa: E402
    DETECTION_DIR,
    REPO_ROOT,
    artifact_fingerprint,
    portable_path,
    repo_path,
    run_step,
    sanitize_json_paths,
)


def mark_intermediate_metrics(path: Path, *, retained: bool) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["intermediate_csv_retained"] = retained
    if not retained:
        for key in ("input", "depth_output", "slice_output"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[f"{key}_name"] = Path(value).name
                payload[key] = None
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_checkpoint_provenance(csv_path: Path, artifacts: dict[str, dict[str, object]]) -> None:
    frame = pd.read_csv(csv_path)
    for role, record in artifacts.items():
        prefix = f"{role}_checkpoint"
        frame[f"{prefix}_id"] = str(record["id"])
        frame[f"{prefix}_sha256"] = str(record["sha256"])
        frame[f"{prefix}_bytes"] = int(record["bytes"])
    temporary = csv_path.with_name(f".{csv_path.name}.provenance.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run HoloD3 detection, depth, and diameter inference from a validated acquisition. "
            "Measurements are independent per frame; this command does not track trajectories."
        )
    )
    p.add_argument("--acquisition-config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=30, help="Maximum frames to process. Use 0 for all selected frames.")
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--yolo-device", default=None)
    p.add_argument("--yolo-weights", type=Path, default=Path("models/production/detector.pt"))
    p.add_argument("--depth-checkpoint", type=Path, default=Path("models/production/depth-primary.pt"))
    p.add_argument("--depth-fallback-checkpoint", type=Path, default=Path("models/production/depth-fallback.pt"))
    p.add_argument("--slice-diam-weights", type=Path, default=Path("models/production/diameter.pt"))
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.15)
    p.add_argument("--max-det", type=int, default=600)
    p.add_argument("--contrast-in-max", type=int, default=75)
    p.add_argument("--yolo-imgsz", type=int, default=1024)
    p.add_argument("--yolo-batch-size", type=int, default=8)
    p.add_argument("--yolo-image-load-workers", type=int, default=4)
    p.add_argument("--depth-model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--depth-batch-size", type=int, default=256)
    p.add_argument("--depth-slice-block", type=int, default=8)
    p.add_argument("--depth-crop-size", type=int, default=64)
    p.add_argument("--depth-slice-start", type=int, default=1)
    p.add_argument("--depth-slice-end", type=int, default=0)
    p.add_argument("--depth-slice-step", type=int, default=1)
    p.add_argument("--diam-batch-size", type=int, default=512)
    p.add_argument("--diam-model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--bbox-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-router", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-router-max-diameter-um", type=float, default=75.0)
    p.add_argument(
        "--diameter-underprediction-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--hybrid-ratio-threshold", type=float, default=0.35)
    p.add_argument("--hybrid-min-bbox-side-um", type=float, default=250.0)
    p.add_argument("--direct-roi-track-ids", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--roi-stream-csv-writes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--roi-stream-write-queue-size", type=int, default=8)
    p.add_argument("--roi-image-prefetch-workers", type=int, default=4)
    p.add_argument("--roi-image-prefetch-frames", type=int, default=32)
    p.add_argument("--stream-csv-writes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stream-write-chunk-frames", type=int, default=50)
    p.add_argument("--stream-write-queue-size", type=int, default=4)
    p.add_argument("--prefetch-holo-workers", type=int, default=4)
    p.add_argument("--prefetch-holo-frames", type=int, default=16)
    p.add_argument("--slice-diam-frame-batch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-buffer-reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-inplace-propagation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-final-hybrid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--recenter-on-slice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine particle centers on the estimated reconstructed slice inside the fused step.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--stop-after-preprocessing",
        action="store_true",
        help="Stop after YOLO detection and ROI measurement, retaining their staging CSV.",
    )
    p.add_argument(
        "--keep-intermediate-csv",
        action="store_true",
        help="Keep large Depth/Diam staging CSVs for debugging. By default they are deleted after particles_3d.csv is complete.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    acquisition_path = repo_path(args.acquisition_config)
    acquisition = AcquisitionConfig.load(acquisition_path)
    selected_records = acquisition.selected_records(
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    run_dir = repo_path(args.run_dir)
    yolo_weights = repo_path(args.yolo_weights)
    depth_checkpoint = repo_path(args.depth_checkpoint)
    depth_fallback_checkpoint = repo_path(args.depth_fallback_checkpoint)
    slice_diam_weights = repo_path(args.slice_diam_weights)
    py = sys.executable

    model_artifacts = {
        role: artifact_fingerprint(path, run_dir=run_dir)
        for role, path in {
            "detector": yolo_weights,
            "depth_primary": depth_checkpoint,
            "depth_fallback": depth_fallback_checkpoint,
            "diameter": slice_diam_weights,
        }.items()
        if path.is_file()
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir = run_dir / "_inputs" / "minip"
    work_dir = run_dir if args.keep_intermediate_csv else run_dir / "_work" / "depth_diam"
    retain_preprocessing = bool(args.keep_intermediate_csv or args.stop_after_preprocessing)
    if args.overwrite and not retain_preprocessing and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = run_dir / "raw_detections.csv"
    frame_stats_csv = run_dir / "frame_stats.csv"
    raw_summary_json = run_dir / "raw_detection_summary.json"
    measured_preids_csv = work_dir / "tracks_with_diameter_preids.csv"
    measured_csv = work_dir / "tracks_with_diameter.csv"
    measured_summary_json = work_dir / "tracks_with_diameter.summary.json"
    depth_csv = work_dir / "tracks_with_diam_depth.csv"
    slice_csv = work_dir / "tracks_with_estdepth_slice_diameter.csv"
    fused_metrics_json = run_dir / "fused_depth_slice_metrics.json"
    final_csv = run_dir / "particles_3d.csv"
    hybrid_metrics_json = run_dir / "hybrid_diameter_metrics.json"
    pipeline_summary_json = run_dir / "pipeline_summary.json"

    protected_outputs = [
        raw_csv,
        frame_stats_csv,
        raw_summary_json,
        fused_metrics_json,
        final_csv,
        hybrid_metrics_json,
        pipeline_summary_json,
    ]
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Run outputs already exist; choose a new --run-dir or pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    overwrite = ["--overwrite"] if args.overwrite else []
    yolo_device = args.yolo_device if args.yolo_device is not None else args.device
    if str(yolo_device).startswith("cuda:"):
        yolo_device = str(yolo_device).split(":", 1)[1]

    steps: list[dict[str, object]] = []
    started = time.perf_counter()

    prepare_started = time.perf_counter()
    minip_summary = prepare_minip_images(
        acquisition,
        selected_records,
        image_dir,
        device=str(args.device),
        overwrite=args.overwrite,
    )
    steps.append(
        {
            "step": "prepare_model_minip_images",
            "elapsed_sec": time.perf_counter() - prepare_started,
            "outputs": [str(image_dir)],
            "summary": minip_summary,
        }
    )

    roi_output_csv = measured_csv if args.direct_roi_track_ids else measured_preids_csv
    yolo_cmd = [
        py,
        str(DETECTION_DIR / "raw_detect_best.py"),
        "--image-dir",
        str(image_dir),
        "--weights",
        str(yolo_weights),
        "--out-csv",
        str(raw_csv),
        "--frame-stats-csv",
        str(frame_stats_csv),
        "--summary-json",
        str(raw_summary_json),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--contrast-in-max",
        str(args.contrast_in_max),
        "--imgsz",
        str(args.yolo_imgsz),
        "--device",
        str(yolo_device),
        "--batch-size",
        str(args.yolo_batch_size),
        "--image-load-workers",
        str(args.yolo_image_load_workers),
    ]
    yolo_cmd += overwrite
    run_step("yolo_raw_minip_detection", yolo_cmd, steps)

    roi_cmd = [
        py,
        str(DETECTION_DIR / "raw_detections_depth_input.py"),
        "--raw-csv",
        str(raw_csv),
        "--image-dir",
        str(image_dir),
        "--out-csv",
        str(roi_output_csv),
        "--summary-json",
        str(measured_summary_json),
        "--min-conf",
        str(args.conf),
        "--pixel-pitch-um",
        str(acquisition.optics.pixel_pitch_um),
        "--device",
        str(args.device),
        "--stream-write-queue-size",
        str(args.roi_stream_write_queue_size),
        "--image-prefetch-workers",
        str(args.roi_image_prefetch_workers),
        "--image-prefetch-frames",
        str(args.roi_image_prefetch_frames),
        *overwrite,
    ]
    roi_cmd.append("--bbox-fallback" if args.bbox_fallback else "--no-bbox-fallback")
    if args.direct_roi_track_ids:
        roi_cmd.append("--assign-track-ids")
    if args.roi_stream_csv_writes:
        roi_cmd.append("--stream-csv-writes")
    run_step(
        "measure_detection_roi_for_depth_input",
        roi_cmd,
        steps,
    )

    if not args.direct_roi_track_ids:
        run_step(
            "assign_detection_ids_no_tracking",
            [
                py,
                str(DETECTION_DIR / "add_detection_track_ids.py"),
                "--input",
                str(measured_preids_csv),
                "--output",
                str(measured_csv),
                *overwrite,
            ],
            steps,
        )

    if args.stop_after_preprocessing:
        summary = {
            "event": "pipeline_preprocessing_done",
            "acquisition": acquisition.name,
            "acquisition_config": portable_path(acquisition_path, acquisition_dir=acquisition.base_dir),
            "acquisition_config_sha256": artifact_fingerprint(acquisition_path)["sha256"],
            "prepared_minip_dir": portable_path(image_dir, run_dir=run_dir),
            "run_dir": "run:.",
            "raw_detections_csv": portable_path(raw_csv, run_dir=run_dir),
            "measured_roi_csv": portable_path(measured_csv, run_dir=run_dir),
            "model_artifacts": model_artifacts,
            "steps": steps,
            "elapsed_sec": time.perf_counter() - started,
        }
        preprocessing_summary = run_dir / "pipeline_preprocessing_summary.json"
        preprocessing_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for metadata_path in run_dir.rglob("*.json"):
            sanitize_json_paths(metadata_path, run_dir=run_dir, acquisition_dir=acquisition.base_dir)
        shareable_summary = json.loads(preprocessing_summary.read_text(encoding="utf-8"))
        print(json.dumps(shareable_summary, ensure_ascii=False, indent=2), flush=True)
        return

    fused_cmd = [
        py,
        str(DETECTION_DIR / "depth_and_slice_fused.py"),
        "--run-dir",
        str(run_dir),
        "--acquisition-config",
        str(acquisition_path),
        "--input",
        str(measured_csv),
        "--depth-output",
        str(depth_csv),
        "--slice-output",
        str(slice_csv),
        "--metrics-output",
        str(fused_metrics_json),
        "--depth-checkpoint",
        str(depth_checkpoint),
        "--depth-fallback-checkpoint",
        str(depth_fallback_checkpoint),
        "--depth-router-max-diameter-um",
        str(args.depth_router_max_diameter_um),
        "--slice-diam-weights",
        str(slice_diam_weights),
        "--device",
        str(args.device),
        "--model-backend",
        str(args.depth_model_backend),
        "--depth-batch-size",
        str(args.depth_batch_size),
        "--slice-block",
        str(args.depth_slice_block),
        "--crop-size",
        str(args.depth_crop_size),
        "--slice-start",
        str(args.depth_slice_start),
        "--slice-end",
        str(args.depth_slice_end),
        "--slice-step",
        str(args.depth_slice_step),
        "--diam-batch-size",
        str(args.diam_batch_size),
        "--diam-model-backend",
        str(args.diam_model_backend),
        "--stream-write-chunk-frames",
        str(args.stream_write_chunk_frames),
        "--stream-write-queue-size",
        str(args.stream_write_queue_size),
        "--prefetch-holo-workers",
        str(args.prefetch_holo_workers),
        "--prefetch-holo-frames",
        str(args.prefetch_holo_frames),
        "--hybrid-ratio-threshold",
        str(args.hybrid_ratio_threshold),
        "--hybrid-min-bbox-side-um",
        str(args.hybrid_min_bbox_side_um),
    ]
    fused_cmd.append("--depth-router" if args.depth_router else "--no-depth-router")
    fused_cmd.append(
        "--diameter-underprediction-fallback"
        if args.diameter_underprediction_fallback
        else "--no-diameter-underprediction-fallback"
    )
    if args.fused_final_hybrid:
        fused_cmd += [
            "--final-output",
            str(final_csv),
            "--hybrid-metrics-output",
            str(hybrid_metrics_json),
        ]
    if args.stream_csv_writes:
        fused_cmd.append("--stream-csv-writes")
    fused_cmd.append("--slice-diam-frame-batch" if args.slice_diam_frame_batch else "--no-slice-diam-frame-batch")
    fused_cmd.append("--depth-buffer-reuse" if args.depth_buffer_reuse else "--no-depth-buffer-reuse")
    fused_cmd.append(
        "--depth-inplace-propagation"
        if args.depth_inplace_propagation
        else "--no-depth-inplace-propagation"
    )
    if args.recenter_on_slice:
        fused_cmd.append("--recenter-on-slice")
    run_step("estimate_depth_and_slice_diameter_fused", fused_cmd, steps)

    if not args.fused_final_hybrid:
        run_step(
            "apply_hybrid_diameter_rule",
            [
                py,
                str(DETECTION_DIR / "apply_hybrid_diameter.py"),
                "--input",
                str(slice_csv),
                "--output",
                str(final_csv),
                "--metrics-output",
                str(hybrid_metrics_json),
                "--pixel-pitch-um",
                str(acquisition.optics.pixel_pitch_um),
                "--slice-spacing-um",
                str(acquisition.optics.slice_spacing_um),
                "--reconstruction-start-um",
                str(acquisition.optics.reconstruction_start_um),
                "--ratio-threshold",
                str(args.hybrid_ratio_threshold),
                "--min-bbox-side-um",
                str(args.hybrid_min_bbox_side_um),
                *overwrite,
            ],
            steps,
        )

    mark_intermediate_metrics(fused_metrics_json, retained=args.keep_intermediate_csv)
    mark_intermediate_metrics(hybrid_metrics_json, retained=args.keep_intermediate_csv)
    add_checkpoint_provenance(final_csv, model_artifacts)

    total_elapsed = time.perf_counter() - started
    summary = {
        "method": "hologram_detection_depth_and_diameter",
        "acquisition": acquisition.name,
        "acquisition_config": portable_path(acquisition_path, acquisition_dir=acquisition.base_dir),
        "acquisition_config_sha256": artifact_fingerprint(acquisition_path)["sha256"],
        "holography_mode": acquisition.mode,
        "prepared_minip_dir": portable_path(image_dir, run_dir=run_dir),
        "run_dir": "run:.",
        "selected_frames": len(selected_records),
        "device": str(args.device),
        "optics": {
            "wavelength_um": float(acquisition.optics.wavelength_um),
            "pixel_pitch_um": float(acquisition.optics.pixel_pitch_um),
            "reconstruction_start_um": float(acquisition.optics.reconstruction_start_um),
            "slice_spacing_um": float(acquisition.optics.slice_spacing_um),
            "slice_count": int(acquisition.optics.slice_count),
        },
        "recenter_on_slice": bool(args.recenter_on_slice),
        "fallbacks": {
            "minip_bbox": bool(args.bbox_fallback),
            "depth_router": bool(args.depth_router),
            "depth_router_max_diameter_um": float(args.depth_router_max_diameter_um),
            "diameter_underprediction": bool(args.diameter_underprediction_fallback),
            "diameter_ratio_threshold": float(args.hybrid_ratio_threshold),
            "diameter_min_bbox_side_um": float(args.hybrid_min_bbox_side_um),
        },
        "model_artifacts": model_artifacts,
        "performance_options": {
            "yolo_batch_size": int(args.yolo_batch_size),
            "yolo_image_size": int(args.yolo_imgsz),
            "yolo_image_load_workers": int(args.yolo_image_load_workers),
            "roi_image_prefetch_workers": int(args.roi_image_prefetch_workers),
            "roi_image_prefetch_frames": int(args.roi_image_prefetch_frames),
            "prefetch_holo_workers": int(args.prefetch_holo_workers),
            "prefetch_holo_frames": int(args.prefetch_holo_frames),
            "fft_padding_side": int(acquisition.reconstruction.fft_padding_side),
            "slice_diam_frame_batch": bool(args.slice_diam_frame_batch),
            "depth_buffer_reuse": bool(args.depth_buffer_reuse),
            "depth_inplace_propagation": bool(args.depth_inplace_propagation),
        },
        "outputs": {
            "raw_detections_csv": portable_path(raw_csv, run_dir=run_dir),
            "particles_3d_csv": portable_path(final_csv, run_dir=run_dir),
            "fused_depth_slice_metrics_json": portable_path(fused_metrics_json, run_dir=run_dir),
            "hybrid_metrics_json": portable_path(hybrid_metrics_json, run_dir=run_dir),
        },
        "intermediate_csv_retained": bool(args.keep_intermediate_csv),
        "intermediate_csv_dir": portable_path(work_dir, run_dir=run_dir) if args.keep_intermediate_csv else None,
        "steps": steps,
        "elapsed_sec": total_elapsed,
    }
    pipeline_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for metadata_path in run_dir.rglob("*.json"):
        sanitize_json_paths(metadata_path, run_dir=run_dir, acquisition_dir=acquisition.base_dir)
    if not args.keep_intermediate_csv and work_dir.exists():
        shutil.rmtree(work_dir)
        if work_dir.parent.is_dir() and not any(work_dir.parent.iterdir()):
            work_dir.parent.rmdir()
    shareable_summary = json.loads(pipeline_summary_json.read_text(encoding="utf-8"))
    print(json.dumps({"event": "pipeline_done", **shareable_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
