#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holod3.acquisition import AcquisitionConfig, FrameRecord  # noqa: E402
from holod3.reconstruction import (  # noqa: E402
    PropagationSetup,
    build_propagation_setup,
    load_frame_arrays,
    wavefront_from_arrays,
)
from src.detection.depth_runtime import (  # noqa: E402
    CudaRoiAbs2Cropper,
    TransferSetup,
    centered_crops_with_mean_padding,
    crop_start_tensors,
    crop_start_xy_tensors,
    fft2,
    ifft2,
    load_model as load_depth_model,
    normalized_image_name,
    optimize_model_backend,
    run_model_batches,
    slice_values,
)
from src.diam.inference import (  # noqa: E402
    optimize_slice_model_backend,
    predict_diameter_batch,
)
from src.diam.model import load_model as load_slice_model  # noqa: E402


DEFAULT_DEPTH_CHECKPOINT = REPO_ROOT / "models/production/depth-primary.pt"
DEFAULT_DEPTH_FALLBACK_CHECKPOINT = REPO_ROOT / "models/production/depth-fallback.pt"
DEFAULT_SLICE_WEIGHTS = REPO_ROOT / "models/production/diameter.pt"

CONTRAST_AREA_FRACTION = 0.70
CONTRAST_AREA_BG_PERCENTILE = 90.0
CONTRAST_AREA_PEAK_PERCENTILE = 95.0
CONTRAST_AREA_CORE_RADIUS_SCALE = 0.30
CONTRAST_AREA_ROI_SCALE = 1.8
CONTRAST_AREA_ROI_MARGIN_PX = 4.0
CONTRAST_AREA_MIN_ROI_RADIUS_PX = 7.0
CONTRAST_AREA_FULL_GATE_PX = 2.0


def portable_model_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external:{path.name}"


def repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate depth and SliceDiamModel diameter in one per-frame pass without changing model semantics."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--acquisition-config", type=Path, required=True)
    p.add_argument("--input", type=Path, default=None, help="Default: <run-dir>/tracks_with_diameter.csv")
    p.add_argument("--depth-output", type=Path, default=None, help="Default: <run-dir>/tracks_with_diam_depth.csv")
    p.add_argument("--slice-output", type=Path, default=None, help="Default: <run-dir>/tracks_with_estdepth_slice_diameter.csv")
    p.add_argument("--final-output", type=Path, default=None, help="Optional final particles_3d.csv with hybrid diameter columns.")
    p.add_argument("--metrics-output", type=Path, default=None, help="Default: <run-dir>/fused_depth_slice_metrics.json")
    p.add_argument("--hybrid-metrics-output", type=Path, default=None, help="Optional hybrid diameter metrics when --final-output is used.")
    p.add_argument("--depth-checkpoint", type=Path, default=DEFAULT_DEPTH_CHECKPOINT)
    p.add_argument("--depth-fallback-checkpoint", type=Path, default=DEFAULT_DEPTH_FALLBACK_CHECKPOINT)
    p.add_argument(
        "--depth-router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route small bbox-fallback measurements to the baseline DepthModel.",
    )
    p.add_argument("--depth-router-max-diameter-um", type=float, default=75.0)
    p.add_argument("--slice-diam-weights", type=Path, default=DEFAULT_SLICE_WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--frames", type=int, default=0, help="0 means all frames.")
    p.add_argument("--row-limit", type=int, default=0, help="0 means all rows.")
    p.add_argument("--only-file", default="")
    p.add_argument("--slices", type=int, default=0, help="0 means the acquisition slice count.")
    p.add_argument("--slice-start", type=int, default=1)
    p.add_argument("--slice-end", type=int, default=0, help="0 means final slice.")
    p.add_argument("--slice-step", type=int, default=1)
    p.add_argument("--slice-block", type=int, default=8)
    p.add_argument(
        "--padlen",
        type=int,
        default=0,
        help="Optional FFT-side override; 0 uses acquisition.reconstruction.fft_padding_side.",
    )
    p.add_argument("--depth-batch-size", type=int, default=256)
    p.add_argument("--diam-batch-size", type=int, default=512)
    p.add_argument("--diam-model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--amp", action="store_true", help="Deprecated for TensorRT backend; kept for torch fallback.")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile-model", action="store_true", help="Only used with --model-backend torch.")
    p.add_argument("--model-backend", choices=["tensorrt", "torch"], default="tensorrt")
    p.add_argument("--fft-backend", choices=["torch"], default="torch")
    p.add_argument("--mask-radius-diam-scale", type=float, default=None)
    p.add_argument("--mask-min-radius-px", type=float, default=None)
    p.add_argument("--mask-softness-px", type=float, default=None)
    p.add_argument("--stream-csv-writes", action="store_true", help="Write output CSV chunks from a background thread.")
    p.add_argument("--stream-write-chunk-frames", type=int, default=50)
    p.add_argument("--stream-write-queue-size", type=int, default=4)
    p.add_argument("--prefetch-holo-workers", type=int, default=0)
    p.add_argument("--prefetch-holo-frames", type=int, default=4)
    p.add_argument(
        "--slice-diam-frame-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate reconstructed crops for one frame and run SliceDiam in large batches.",
    )
    p.add_argument(
        "--depth-buffer-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-frame padded/crop/diameter buffers and avoid block torch.cat.",
    )
    p.add_argument(
        "--depth-inplace-propagation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clone base frequency once and propagate with in-place multiply.",
    )
    p.add_argument(
        "--recenter-on-slice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine seg_xc/seg_yc with contrast-area centroid on the estimated reconstructed slice crop.",
    )
    p.add_argument("--hybrid-ratio-threshold", type=float, default=0.35)
    p.add_argument("--hybrid-min-bbox-side-um", type=float, default=250.0)
    p.add_argument(
        "--diameter-underprediction-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the MinIP diameter for large boxes with severe SliceDiam underprediction.",
    )
    return p.parse_args()


def transfer_setup_for_padlen(
    acquisition: AcquisitionConfig,
    device: torch.device,
    requested_slices: int,
    requested_padlen: int,
) -> tuple[TransferSetup, PropagationSetup, int]:
    propagation = build_propagation_setup(acquisition, device)
    slices = (
        min(int(requested_slices), propagation.slice_count)
        if int(requested_slices) > 0
        else propagation.slice_count
    )
    padlen = int(requested_padlen) if int(requested_padlen) > 0 else propagation.padding_side
    if padlen < propagation.image_size:
        raise ValueError(f"padlen must be >= image size: {padlen} < {propagation.image_size}")
    if (padlen - propagation.image_size) % 2 != 0:
        raise ValueError(f"padlen - image size must be even: {padlen} - {propagation.image_size}")
    if padlen != propagation.padding_side:
        raise ValueError(
            "--padlen must match acquisition.reconstruction.fft_padding_side; edit acquisition.yaml to change it."
        )
    placeholder = torch.empty(0, device=device, dtype=torch.complex64)
    setup = TransferSetup(
        variables={
            "lambda": float(acquisition.optics.wavelength_um),
            "dx": float(acquisition.optics.pixel_pitch_um),
            "dz": float(acquisition.optics.slice_spacing_um),
            "datlen": float(acquisition.optics.image_size_px),
            "slices": float(slices),
        },
        coeffs=(
            propagation.distortion_coefficients
            if propagation.distortion_coefficients is not None
            else np.empty((0,), dtype=np.float64)
        ),
        dz_um=float(propagation.slice_spacing_um),
        datlen=int(propagation.image_size),
        slices=int(slices),
        z0_um=float(propagation.reconstruction_start_um),
        d_pr=propagation.phase_forward if propagation.phase_forward is not None else placeholder,
        d_pr_inv=propagation.phase_inverse if propagation.phase_inverse is not None else placeholder,
        d_tf_unshifted=propagation.initial_transfer,
        d_slice_unshifted=propagation.slice_transfer,
    )
    return setup, propagation, padlen


def frame_groups(df: pd.DataFrame, frames: int, only_file: str) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, np.ndarray]] = []
    for file_name, group in df.groupby("file", sort=True):
        norm = normalized_image_name(file_name)
        if only_file and norm != normalized_image_name(only_file):
            continue
        groups.append((norm, group.index.to_numpy(np.int64)))
        if frames > 0 and len(groups) >= frames:
            break
    return groups


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


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


def recenter_particle_on_crop(
    crop: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    xc: float,
    yc: float,
) -> tuple[float, float, str, int]:
    h, w = crop.shape[:2]
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

    roi = crop[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return xc, yc, "recenter_fallback_empty_roi", 0

    blur = cv2.GaussianBlur(roi.astype(np.float32, copy=False), (3, 3), 0)
    local_cx = xc - rx1
    local_cy = yc - ry1

    yy, xx = np.indices(blur.shape)
    rr = np.hypot(xx - local_cx, yy - local_cy)
    background = float(np.percentile(blur, CONTRAST_AREA_BG_PERCENTILE))
    darkness = np.clip(background - blur.astype(np.float32), 0.0, None)
    core = darkness[rr <= max(2.0, bbox_diameter_px * CONTRAST_AREA_CORE_RADIUS_SCALE)]
    if core.size == 0:
        return xc, yc, "recenter_fallback_no_core", 0

    peak = float(np.percentile(core, CONTRAST_AREA_PEAK_PERCENTILE))
    if peak <= 1e-6:
        return xc, yc, "recenter_fallback_no_contrast", 0

    binary = (darkness >= CONTRAST_AREA_FRACTION * peak).astype(np.uint8) * 255
    selected = component_for_particle(binary, local_cx, local_cy)
    if selected is None:
        return xc, yc, "recenter_fallback_no_component", 0

    _label, stats, centroid = selected
    seg_x = rx1 + float(centroid[0])
    seg_y = ry1 + float(centroid[1])
    if math.hypot(seg_x - xc, seg_y - yc) > CONTRAST_AREA_FULL_GATE_PX:
        return xc, yc, "recenter_fallback_gate2px", 0
    return seg_x, seg_y, "reconstructed_contrast_area_f70_bg90_component", int(stats[cv2.CC_STAT_AREA])


def add_recenter_columns(
    frame: pd.DataFrame,
    indices: np.ndarray,
    recenter_x: np.ndarray,
    recenter_y: np.ndarray,
    recenter_method: list[str],
    recenter_area: np.ndarray,
) -> pd.DataFrame:
    out = frame.copy()
    out["orig_seg_xc"] = out["seg_xc"]
    out["orig_seg_yc"] = out["seg_yc"]
    out["recenter_x"] = recenter_x[indices]
    out["recenter_y"] = recenter_y[indices]
    out["recenter_method"] = [recenter_method[int(i)] for i in indices]
    out["recenter_area_px"] = recenter_area[indices]
    out["recenter_applied"] = (
        out["recenter_method"].to_numpy() == "reconstructed_contrast_area_f70_bg90_component"
    ).astype(np.int8)
    out["seg_xc"] = out["recenter_x"]
    out["seg_yc"] = out["recenter_y"]
    return out


class AsyncCsvWriter:
    def __init__(self, paths: dict[str, Path], *, max_queue: int) -> None:
        self.paths = paths
        self.queue: queue.Queue[dict[str, pd.DataFrame] | None] = queue.Queue(maxsize=max(1, max_queue))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="fused-csv-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            files = {}
            wrote_header = {name: False for name in self.paths}
            try:
                for name, path in self.paths.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    files[name] = path.open("w", encoding="utf-8", newline="")
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            break
                        for name, frame in item.items():
                            if frame.empty:
                                continue
                            frame.to_csv(files[name], index=False, header=not wrote_header[name])
                            wrote_header[name] = True
                    finally:
                        self.queue.task_done()
            finally:
                for f in files.values():
                    f.close()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc

    def write(self, item: dict[str, pd.DataFrame]) -> None:
        if self.error is not None:
            raise RuntimeError("CSV writer thread failed") from self.error
        self.queue.put(item)

    def close(self) -> float:
        started = time.perf_counter()
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("CSV writer thread failed") from self.error
        return time.perf_counter() - started


class HoloPrefetcher:
    def __init__(
        self,
        groups: list[tuple[str, np.ndarray]],
        acquisition: AcquisitionConfig,
        records: dict[str, FrameRecord],
        propagation: PropagationSetup,
        *,
        workers: int,
        lookahead: int,
    ) -> None:
        self.groups = groups
        self.acquisition = acquisition
        self.records = records
        self.propagation = propagation
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="holo-prefetch")
        self.lookahead = max(1, lookahead)
        self.next_submit = 0
        self.futures: dict[int, Future[tuple[np.ndarray, np.ndarray | None]]] = {}
        self._submit_until(0)

    def _load_frame(self, file_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        stem = Path(file_name).stem
        try:
            record = self.records[stem]
        except KeyError as exc:
            raise FileNotFoundError(f"No raw hologram matches detected MinIP frame {file_name!r}.") from exc
        return load_frame_arrays(self.acquisition, record, self.propagation)

    def _submit_until(self, current_index: int) -> None:
        target = min(len(self.groups), current_index + self.lookahead)
        while self.next_submit < target:
            file_name, _ = self.groups[self.next_submit]
            self.futures[self.next_submit] = self.executor.submit(self._load_frame, file_name)
            self.next_submit += 1

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        self._submit_until(index + 1)
        future = self.futures.pop(index)
        return future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)


def compute_holo_from_arrays(
    primary_np: np.ndarray,
    secondary_np: np.ndarray | None,
    acquisition: AcquisitionConfig,
    propagation: PropagationSetup,
    device: torch.device,
) -> torch.Tensor:
    return wavefront_from_arrays(acquisition, primary_np, secondary_np, propagation, device)


def add_hybrid_columns(
    slice_frame: pd.DataFrame,
    *,
    ratio_threshold: float,
    min_bbox_side_um: float,
    pixel_pitch_um: float,
    reconstruction_start_um: float,
    slice_spacing_um: float,
    enabled: bool = True,
) -> tuple[pd.DataFrame, int]:
    if "diameter_px" in slice_frame.columns:
        diam_px = slice_frame["diameter_px"].to_numpy(float)
        pitch = np.where(diam_px > 0, slice_frame["diameter_um"].to_numpy(float) / diam_px, pixel_pitch_um)
    else:
        pitch = np.full(len(slice_frame), pixel_pitch_um, dtype=np.float64)
    bbox_avg_um = 0.5 * (slice_frame["w"].to_numpy(float) + slice_frame["h"].to_numpy(float)) * pitch
    slice_pred = slice_frame["slice_diam_pred_um"].to_numpy(float)
    current = slice_frame["diameter_um"].to_numpy(float)
    fallback = enabled & (bbox_avg_um >= min_bbox_side_um) & (slice_pred < ratio_threshold * bbox_avg_um)
    final = np.where(fallback, current, slice_pred)

    out = slice_frame.copy()
    out["bbox_avg_side_um"] = bbox_avg_um
    out["diameter_fallback_to_contrast_area"] = fallback.astype(np.int8)
    out["final_diameter_um"] = final
    out["final_diameter_source"] = np.where(fallback, "contrast_area_fallback", "slice_diammodel")
    out["final_diameter_rule"] = (
        (
            f"bbox_avg_side_um >= {min_bbox_side_um:g} and "
            f"slice_pred < {ratio_threshold:g} * bbox_avg_side_um"
        )
        if enabled
        else "disabled"
    )
    out["x_um"] = out["seg_xc"].astype(float) * pixel_pitch_um
    out["y_um"] = out["seg_yc"].astype(float) * pixel_pitch_um
    out["z_um"] = (out["slice"].astype(float) - 1.0) * slice_spacing_um
    out["depth_um"] = reconstruction_start_um + out["z_um"]
    return out, int(fallback.sum())


def main() -> None:
    args = parse_args()
    run_dir = repo_path(args.run_dir)
    acquisition = AcquisitionConfig.load(repo_path(args.acquisition_config))
    frame_record_list = acquisition.frame_records()
    frame_records_by_stem = {record.stem: record for record in frame_record_list}
    input_path = repo_path(args.input) if args.input else run_dir / "tracks_with_diameter.csv"
    depth_output = repo_path(args.depth_output) if args.depth_output else run_dir / "tracks_with_diam_depth.csv"
    slice_output = repo_path(args.slice_output) if args.slice_output else run_dir / "tracks_with_estdepth_slice_diameter.csv"
    final_output = repo_path(args.final_output) if args.final_output else None
    metrics_output = repo_path(args.metrics_output) if args.metrics_output else run_dir / "fused_depth_slice_metrics.json"
    hybrid_metrics_output = (
        repo_path(args.hybrid_metrics_output)
        if args.hybrid_metrics_output
        else (run_dir / "hybrid_diameter_metrics.json" if final_output is not None else None)
    )
    depth_checkpoint = repo_path(args.depth_checkpoint)
    depth_fallback_checkpoint = repo_path(args.depth_fallback_checkpoint)
    slice_weights = repo_path(args.slice_diam_weights)
    depth_checkpoint_id = portable_model_id(depth_checkpoint)
    depth_fallback_checkpoint_id = portable_model_id(depth_fallback_checkpoint)
    slice_weights_id = portable_model_id(slice_weights)

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    setup, propagation, padlen = transfer_setup_for_padlen(acquisition, device, args.slices, args.padlen)

    depth_model, depth_info = load_depth_model(depth_checkpoint, device, False, args.channels_last)
    if args.mask_radius_diam_scale is not None:
        depth_info["mask_radius_diam_scale"] = float(args.mask_radius_diam_scale)
    if args.mask_min_radius_px is not None:
        depth_info["mask_min_radius_px"] = float(args.mask_min_radius_px)
    if args.mask_softness_px is not None:
        depth_info["mask_softness_px"] = float(args.mask_softness_px)
    args.batch_size = int(args.depth_batch_size)
    depth_model, depth_compile_sec = optimize_model_backend(depth_model, depth_info, args, device)
    depth_fallback_model = None
    depth_fallback_info: dict[str, Any] = {
        "model_backend": "disabled",
        "inference_precision": "disabled",
    }
    depth_fallback_compile_sec = 0.0
    if args.depth_router:
        depth_fallback_model, depth_fallback_info = load_depth_model(
            depth_fallback_checkpoint, device, False, args.channels_last
        )
        depth_fallback_model, depth_fallback_compile_sec = optimize_model_backend(
            depth_fallback_model, depth_fallback_info, args, device
        )

    slice_model, slice_norm, slice_calibration_scale, slice_ckpt = load_slice_model(slice_weights, device)
    slice_model.eval()
    slice_model, slice_backend_info, slice_compile_sec = optimize_slice_model_backend(
        slice_model,
        backend=args.diam_model_backend,
        device=device,
        batch_size=int(args.diam_batch_size),
        crop_size=int(args.crop_size),
    )

    df = pd.read_csv(input_path, nrows=args.row_limit if args.row_limit > 0 else None)
    required = {"frame", "file", "seg_xc", "seg_yc", "diameter_um"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{input_path} is missing required columns: {missing}")
    df = df.reset_index(drop=True)
    df["file"] = [normalized_image_name(v) for v in df["file"]]
    if "diameter_method" not in df.columns:
        raise RuntimeError(f"{input_path} is missing required router column: diameter_method")
    router_to_fallback = (
        args.depth_router
        & df["diameter_method"].astype(str).str.contains("fallback", case=False, na=False)
        & (df["diameter_um"].astype(float) <= args.depth_router_max_diameter_um)
    ).to_numpy(bool)
    groups = frame_groups(df, args.frames, args.only_file)
    if not groups:
        raise RuntimeError("No frames selected")

    slices = slice_values(setup, args)
    offset = (padlen - setup.datlen) // 2
    missing_raw = sorted({Path(file_name).stem for file_name, _ in groups} - set(frame_records_by_stem))
    if missing_raw:
        raise FileNotFoundError(f"Detected MinIP frames have no matching raw holograms: {missing_raw[:10]}")
    prefetcher = (
        HoloPrefetcher(
            groups,
            acquisition,
            frame_records_by_stem,
            propagation,
            workers=args.prefetch_holo_workers,
            lookahead=args.prefetch_holo_frames,
        )
        if args.prefetch_holo_workers > 0
        else None
    )

    roi_cropper: CudaRoiAbs2Cropper | None = None
    roi_kernel_compile_sec = 0.0
    if device.type == "cuda":
        k0 = time.perf_counter()
        roi_cropper = CudaRoiAbs2Cropper.build(device)
        roi_kernel_compile_sec = time.perf_counter() - k0

    best_slice = np.zeros((len(df),), dtype=np.int32)
    best_score = np.full((len(df),), -np.inf, dtype=np.float32)
    pred = np.full((len(df),), np.nan, dtype=np.float32)
    sigma = np.full((len(df),), np.nan, dtype=np.float32)
    pred_log = np.full((len(df),), np.nan, dtype=np.float32)
    sigma_log = np.full((len(df),), np.nan, dtype=np.float32)
    recenter_x = df["seg_xc"].to_numpy(np.float32).copy()
    recenter_y = df["seg_yc"].to_numpy(np.float32).copy()
    recenter_area = np.zeros((len(df),), dtype=np.int32)
    recenter_method = ["not_enabled"] * len(df)

    holo_times: list[float] = []
    depth_times: list[float] = []
    slice_times: list[float] = []
    recenter_times: list[float] = []
    depth_crops_scored = 0
    slice_crops_scored = 0
    recenter_success_count = 0
    recenter_fallback_count = 0
    unique_frame_slices = 0
    final_fallback_count = 0
    csv_writer_close_sec = 0.0
    csv_writer_put_wait_sec = [0.0]
    csv_writer: AsyncCsvWriter | None = None
    depth_stream_chunks: list[pd.DataFrame] = []
    slice_stream_chunks: list[pd.DataFrame] = []
    final_stream_chunks: list[pd.DataFrame] = []
    if args.stream_csv_writes:
        writer_paths = {"depth": depth_output, "slice": slice_output}
        if final_output is not None:
            writer_paths["final"] = final_output
        csv_writer = AsyncCsvWriter(writer_paths, max_queue=args.stream_write_queue_size)

    def flush_stream_chunks(force: bool = False) -> None:
        if csv_writer is None or not depth_stream_chunks:
            return
        if not force and len(depth_stream_chunks) < max(1, args.stream_write_chunk_frames):
            return
        item = {
            "depth": pd.concat(depth_stream_chunks, ignore_index=True),
            "slice": pd.concat(slice_stream_chunks, ignore_index=True),
        }
        if final_output is not None:
            item["final"] = pd.concat(final_stream_chunks, ignore_index=True)
        q0 = time.perf_counter()
        csv_writer.write(item)
        csv_writer_put_wait_sec[0] += time.perf_counter() - q0
        depth_stream_chunks.clear()
        slice_stream_chunks.clear()
        final_stream_chunks.clear()

    started = time.perf_counter()

    print(
        json.dumps(
            {
                "event": "start",
                "frames": len(groups),
                "rows": int(sum(len(indices) for _, indices in groups)),
                "slices": len(slices),
                "device": str(device),
                "depth_checkpoint": str(depth_checkpoint),
                "slice_weights": str(slice_weights),
                "depth_batch_size": int(args.depth_batch_size),
                "slice_block": int(args.slice_block),
                "padlen": int(padlen),
                "slice_diam_frame_batch": bool(args.slice_diam_frame_batch),
                "depth_buffer_reuse": bool(args.depth_buffer_reuse),
                "depth_inplace_propagation": bool(args.depth_inplace_propagation),
                "diam_batch_size": int(args.diam_batch_size),
                "depth_backend": str(depth_info.get("model_backend", args.model_backend)),
                "depth_precision": str(depth_info.get("inference_precision", "fp32")),
                "depth_compile_sec": round(float(depth_compile_sec), 4),
                "diam_backend": str(slice_backend_info["model_backend"]),
                "diam_precision": str(slice_backend_info["inference_precision"]),
                "diam_compile_sec": round(float(slice_compile_sec), 4),
                "roi_kernel_compile_sec": round(float(roi_kernel_compile_sec), 4),
                "stream_csv_writes": bool(args.stream_csv_writes),
                "stream_write_chunk_frames": int(args.stream_write_chunk_frames),
                "prefetch_holo_workers": int(args.prefetch_holo_workers),
                "prefetch_holo_frames": int(args.prefetch_holo_frames),
                "recenter_on_slice": bool(args.recenter_on_slice),
                "final_output": str(final_output) if final_output is not None else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    padded_reuse = (
        torch.empty((padlen, padlen), device=device, dtype=torch.complex64)
        if args.depth_buffer_reuse
        else None
    )

    with torch.inference_mode():
        for frame_i, (file_name, indices) in enumerate(groups, start=1):
            rows = df.loc[indices].reset_index(drop=True)
            frame_route_to_fallback = torch.from_numpy(router_to_fallback[indices]).to(device=device)
            nrows = len(rows)
            diam_frame = torch.from_numpy(rows["diameter_um"].to_numpy(np.float32)).to(device=device)
            if roi_cropper is not None:
                x0, y0 = crop_start_xy_tensors(rows, args.crop_size, device)
                x_idx = y_idx = valid = None
            else:
                x_idx, y_idx, valid = crop_start_tensors(rows, args.crop_size, setup.datlen, device)
                x0 = y0 = None

            h0 = time.perf_counter()
            if prefetcher is None:
                record = frame_records_by_stem[Path(file_name).stem]
                primary_np, secondary_np = load_frame_arrays(acquisition, record, propagation)
            else:
                primary_np, secondary_np = prefetcher.get(frame_i - 1)
            d_holo = compute_holo_from_arrays(
                primary_np,
                secondary_np,
                acquisition,
                propagation,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            holo_times.append(time.perf_counter() - h0)

            padded = (
                padded_reuse
                if padded_reuse is not None
                else torch.empty((padlen, padlen), device=device, dtype=torch.complex64)
            )
            padded.fill_(d_holo.mean())
            padded[offset : offset + setup.datlen, offset : offset + setup.datlen] = d_holo
            base_freq = fft2(padded, args.fft_backend) * setup.d_tf_unshifted

            freq = base_freq.clone() if args.depth_inplace_propagation else base_freq
            if slices[0] > 1:
                initial_transfer = setup.d_slice_unshifted ** int(slices[0] - 1)
                if args.depth_inplace_propagation:
                    freq.mul_(initial_transfer)
                else:
                    freq = freq * initial_transfer
            current_slice = int(slices[0])
            frame_best_slice = torch.full((nrows,), int(slices[0]), device=device, dtype=torch.int32)
            frame_best_score = torch.full((nrows,), -torch.inf, device=device, dtype=torch.float32)
            step_cache: dict[int, torch.Tensor] = {}
            has_fallback = bool(frame_route_to_fallback.any().item())

            d0 = time.perf_counter()
            block_crops: list[torch.Tensor] = []
            block_slices: list[int] = []
            block_count = 0
            reuse_depth_buffers = bool(args.depth_buffer_reuse and roi_cropper is not None)
            depth_crop_buffer = None
            depth_diam_buffer = None
            if reuse_depth_buffers:
                depth_crop_buffer = torch.empty(
                    (args.slice_block, nrows, args.crop_size, args.crop_size),
                    device=device,
                    dtype=torch.float32,
                )
                depth_diam_buffer = diam_frame.repeat(args.slice_block)

            def score_depth_block() -> None:
                nonlocal block_count, depth_crops_scored
                count = block_count if reuse_depth_buffers else len(block_crops)
                if count <= 0:
                    return
                if reuse_depth_buffers:
                    if depth_crop_buffer is None or depth_diam_buffer is None:
                        raise RuntimeError("Depth reuse buffers were not initialized")
                    crops = depth_crop_buffer[:count].reshape(-1, args.crop_size, args.crop_size)
                    diam = depth_diam_buffer[: count * nrows]
                else:
                    crops = torch.cat(block_crops, dim=0)
                    diam = diam_frame.repeat(count)
                scores = run_model_batches(
                    depth_model,
                    crops,
                    diam,
                    depth_info,
                    args.depth_batch_size,
                    args.amp,
                    args.channels_last,
                    acquisition.optics.pixel_pitch_um,
                ).view(count, nrows)
                if has_fallback:
                    if depth_fallback_model is None:
                        raise RuntimeError("Depth fallback rows were selected while the fallback model is disabled")
                    fallback_crops = crops.view(
                        count, nrows, args.crop_size, args.crop_size
                    )[:, frame_route_to_fallback].reshape(-1, args.crop_size, args.crop_size)
                    fallback_diam = diam_frame[frame_route_to_fallback].repeat(count)
                    fallback_scores = run_model_batches(
                        depth_fallback_model,
                        fallback_crops,
                        fallback_diam,
                        depth_fallback_info,
                        args.depth_batch_size,
                        args.amp,
                        args.channels_last,
                        acquisition.optics.pixel_pitch_um,
                    ).view(count, -1)
                    scores[:, frame_route_to_fallback] = fallback_scores
                values, arg = scores.max(dim=0)
                improve = values > frame_best_score
                slice_tensor = torch.tensor(block_slices, device=device, dtype=torch.int32)
                candidate_slices = slice_tensor[arg]
                frame_best_slice.copy_(torch.where(improve, candidate_slices, frame_best_slice))
                frame_best_score.copy_(torch.where(improve, values, frame_best_score))
                depth_crops_scored += int(crops.shape[0])
                block_crops.clear()
                block_slices.clear()
                block_count = 0

            def collect_depth_crop(field: torch.Tensor, zi: int) -> None:
                nonlocal block_count
                if roi_cropper is not None:
                    if x0 is None or y0 is None:
                        raise RuntimeError("CUDA crop coordinates missing")
                    if not field.is_contiguous():
                        field = field.contiguous()
                    if reuse_depth_buffers:
                        if depth_crop_buffer is None:
                            raise RuntimeError("Depth crop buffer was not initialized")
                        crop_out = depth_crop_buffer[block_count]
                    else:
                        crop_out = torch.empty(
                            (nrows, args.crop_size, args.crop_size), device=device, dtype=torch.float32
                        )
                    roi_cropper.crop_abs2_meanpad(field, x0, y0, crop_out, setup.datlen, offset)
                    if not reuse_depth_buffers:
                        block_crops.append(crop_out)
                else:
                    if x_idx is None or y_idx is None or valid is None:
                        raise RuntimeError("Torch crop coordinates missing")
                    intensity = (field.real.square() + field.imag.square()).clamp_(0.0, 1.0)
                    central = intensity[offset : offset + setup.datlen, offset : offset + setup.datlen]
                    block_crops.append(centered_crops_with_mean_padding(central, x_idx, y_idx, valid))
                block_slices.append(int(zi))
                block_count += 1
                if block_count >= args.slice_block:
                    score_depth_block()

            for zi in slices:
                gap = int(zi - current_slice)
                if gap > 0:
                    step_transfer = step_cache.get(gap)
                    if step_transfer is None:
                        step_transfer = setup.d_slice_unshifted ** gap
                        step_cache[gap] = step_transfer
                    if args.depth_inplace_propagation:
                        freq.mul_(step_transfer)
                    else:
                        freq = freq * step_transfer
                    current_slice = int(zi)
                collect_depth_crop(ifft2(freq, args.fft_backend), int(zi))

            score_depth_block()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            depth_times.append(time.perf_counter() - d0)

            frame_best_slice_np = frame_best_slice.cpu().numpy().astype(np.int32)
            frame_best_score_np = frame_best_score.cpu().numpy().astype(np.float32)
            best_slice[indices] = frame_best_slice_np
            best_score[indices] = frame_best_score_np
            frame_pred = np.full(nrows, np.nan, dtype=np.float32)
            frame_sigma = np.full(nrows, np.nan, dtype=np.float32)
            frame_pred_log = np.full(nrows, np.nan, dtype=np.float32)
            frame_sigma_log = np.full(nrows, np.nan, dtype=np.float32)
            frame_recenter_crops: list[torch.Tensor] = []
            frame_recenter_indices: list[np.ndarray] = []
            frame_recenter_x0: list[np.ndarray] = []
            frame_recenter_y0: list[np.ndarray] = []
            frame_recenter_x1: list[np.ndarray] = []
            frame_recenter_y1: list[np.ndarray] = []
            frame_recenter_x2: list[np.ndarray] = []
            frame_recenter_y2: list[np.ndarray] = []
            frame_recenter_xc: list[np.ndarray] = []
            frame_recenter_yc: list[np.ndarray] = []
            frame_diam_crops: list[torch.Tensor] = []
            frame_diam_indices: list[np.ndarray] = []
            frame_diam_positions: list[np.ndarray] = []
            frame_recenter_elapsed = 0.0

            s0 = time.perf_counter()
            freq = base_freq.clone() if args.depth_inplace_propagation else base_freq
            current_slice = 1
            step_cache.clear()
            for slice_int in sorted(np.unique(frame_best_slice_np).astype(np.int32).tolist()):
                gap = int(slice_int - current_slice)
                if gap > 0:
                    step_transfer = step_cache.get(gap)
                    if step_transfer is None:
                        step_transfer = setup.d_slice_unshifted ** gap
                        step_cache[gap] = step_transfer
                    if args.depth_inplace_propagation:
                        freq.mul_(step_transfer)
                    else:
                        freq = freq * step_transfer
                    current_slice = int(slice_int)

                mask = frame_best_slice_np == int(slice_int)
                local_positions = np.flatnonzero(mask)
                local_indices = indices[local_positions]
                sub = df.loc[local_indices]
                field = ifft2(freq, args.fft_backend)
                if not field.is_contiguous():
                    field = field.contiguous()
                if roi_cropper is not None:
                    x0_t, y0_t = crop_start_xy_tensors(sub, args.crop_size, device)
                    crops = torch.empty((len(local_indices), args.crop_size, args.crop_size), device=device, dtype=torch.float32)
                    roi_cropper.crop_abs2_meanpad(field, x0_t, y0_t, crops, setup.datlen, offset)
                else:
                    intensity = (field.real.square() + field.imag.square()).clamp(0.0, 1.0)
                    central = intensity[offset : offset + setup.datlen, offset : offset + setup.datlen]
                    x_idx_s, y_idx_s, valid_s = crop_start_tensors(sub, args.crop_size, setup.datlen, device)
                    crops = centered_crops_with_mean_padding(central, x_idx_s, y_idx_s, valid_s)

                if args.recenter_on_slice:
                    sub_x = sub["seg_xc"].to_numpy(np.float64)
                    sub_y = sub["seg_yc"].to_numpy(np.float64)
                    if roi_cropper is not None:
                        x0_np = x0_t.detach().cpu().numpy().astype(np.float64)
                        y0_np = y0_t.detach().cpu().numpy().astype(np.float64)
                    else:
                        x0_np = np.rint(sub_x - args.crop_size / 2).astype(np.float64)
                        y0_np = np.rint(sub_y - args.crop_size / 2).astype(np.float64)
                    x1_np = sub["x1"].to_numpy(np.float64)
                    y1_np = sub["y1"].to_numpy(np.float64)
                    x2_np = sub["x2"].to_numpy(np.float64)
                    y2_np = sub["y2"].to_numpy(np.float64)
                    frame_recenter_crops.append(crops.detach())
                    frame_recenter_indices.append(local_indices.copy())
                    frame_recenter_x0.append(x0_np)
                    frame_recenter_y0.append(y0_np)
                    frame_recenter_x1.append(x1_np)
                    frame_recenter_y1.append(y1_np)
                    frame_recenter_x2.append(x2_np)
                    frame_recenter_y2.append(y2_np)
                    frame_recenter_xc.append(sub_x)
                    frame_recenter_yc.append(sub_y)

                if args.slice_diam_frame_batch:
                    frame_diam_crops.append(crops)
                    frame_diam_indices.append(local_indices.copy())
                    frame_diam_positions.append(local_positions.copy())
                else:
                    for start in range(0, len(local_indices), args.diam_batch_size):
                        end = min(start + args.diam_batch_size, len(local_indices))
                        p_um, s_um, p_log, s_log = predict_diameter_batch(
                            slice_model, crops[start:end], slice_norm, slice_calibration_scale
                        )
                        pred[local_indices[start:end]] = p_um.astype(np.float32)
                        sigma[local_indices[start:end]] = s_um.astype(np.float32)
                        pred_log[local_indices[start:end]] = p_log.astype(np.float32)
                        sigma_log[local_indices[start:end]] = s_log.astype(np.float32)
                        frame_positions = local_positions[start:end]
                        frame_pred[frame_positions] = p_um.astype(np.float32)
                        frame_sigma[frame_positions] = s_um.astype(np.float32)
                        frame_pred_log[frame_positions] = p_log.astype(np.float32)
                        frame_sigma_log[frame_positions] = s_log.astype(np.float32)
                unique_frame_slices += 1
                slice_crops_scored += int(len(local_indices))

            if args.slice_diam_frame_batch and frame_diam_crops:
                all_diam_crops = torch.cat(frame_diam_crops, dim=0)
                all_diam_indices = np.concatenate(frame_diam_indices)
                all_diam_positions = np.concatenate(frame_diam_positions)
                for start in range(0, len(all_diam_indices), args.diam_batch_size):
                    end = min(start + args.diam_batch_size, len(all_diam_indices))
                    p_um, s_um, p_log, s_log = predict_diameter_batch(
                        slice_model, all_diam_crops[start:end], slice_norm, slice_calibration_scale
                    )
                    target_indices = all_diam_indices[start:end]
                    target_positions = all_diam_positions[start:end]
                    pred[target_indices] = p_um.astype(np.float32)
                    sigma[target_indices] = s_um.astype(np.float32)
                    pred_log[target_indices] = p_log.astype(np.float32)
                    sigma_log[target_indices] = s_log.astype(np.float32)
                    frame_pred[target_positions] = p_um.astype(np.float32)
                    frame_sigma[target_positions] = s_um.astype(np.float32)
                    frame_pred_log[target_positions] = p_log.astype(np.float32)
                    frame_sigma_log[target_positions] = s_log.astype(np.float32)

            if args.recenter_on_slice and frame_recenter_crops:
                r0 = time.perf_counter()
                crop_np = torch.cat(frame_recenter_crops, dim=0).cpu().numpy()
                rec_indices = np.concatenate(frame_recenter_indices)
                x0_np = np.concatenate(frame_recenter_x0)
                y0_np = np.concatenate(frame_recenter_y0)
                x1_np = np.concatenate(frame_recenter_x1)
                y1_np = np.concatenate(frame_recenter_y1)
                x2_np = np.concatenate(frame_recenter_x2)
                y2_np = np.concatenate(frame_recenter_y2)
                sub_x = np.concatenate(frame_recenter_xc)
                sub_y = np.concatenate(frame_recenter_yc)
                for j, global_idx in enumerate(rec_indices):
                    lx, ly, method, area = recenter_particle_on_crop(
                        crop_np[j],
                        x1_np[j] - x0_np[j],
                        y1_np[j] - y0_np[j],
                        x2_np[j] - x0_np[j],
                        y2_np[j] - y0_np[j],
                        sub_x[j] - x0_np[j],
                        sub_y[j] - y0_np[j],
                    )
                    recenter_x[global_idx] = np.float32(lx + x0_np[j])
                    recenter_y[global_idx] = np.float32(ly + y0_np[j])
                    recenter_method[global_idx] = method
                    recenter_area[global_idx] = int(area)
                    if method == "reconstructed_contrast_area_f70_bg90_component":
                        recenter_success_count += 1
                    else:
                        recenter_fallback_count += 1
                frame_recenter_elapsed = time.perf_counter() - r0
                recenter_times.append(frame_recenter_elapsed)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            slice_times.append(time.perf_counter() - s0)

            if csv_writer is not None:
                depth_frame = df.loc[indices].copy()
                depth_frame["slice"] = frame_best_slice_np
                depth_frame["depth_um"] = setup.z0_um + (frame_best_slice_np.astype(np.float64) - 1.0) * setup.dz_um
                depth_frame["model_score"] = frame_best_score_np
                depth_frame["depth_method"] = "focus_score_model_argmax"
                depth_frame["depth_router_source"] = np.where(
                    router_to_fallback[indices], "baseline_fallback", "robust_primary"
                )
                depth_frame["depth_primary_checkpoint"] = depth_checkpoint_id
                depth_frame["depth_fallback_checkpoint"] = depth_fallback_checkpoint_id
                depth_frame["model_crop_method"] = (
                    "cuda_field_roi_abs2_centered_64px_valid_region_mean_padding"
                    if roi_cropper is not None
                    else "torch_full_intensity_centered_64px_valid_region_mean_padding"
                )
                slice_frame = depth_frame.copy()
                slice_frame["slice_diam_pred_um"] = frame_pred
                slice_frame["slice_diam_sigma_um"] = frame_sigma
                slice_frame["slice_diam_pred_log"] = frame_pred_log
                slice_frame["slice_diam_sigma_log"] = frame_sigma_log
                slice_frame["slice_diam_model"] = slice_weights_id
                slice_frame["slice_diam_input_depth"] = "depth_and_slice_fused.slice"
                if args.recenter_on_slice:
                    slice_frame = add_recenter_columns(
                        slice_frame,
                        indices,
                        recenter_x,
                        recenter_y,
                        recenter_method,
                        recenter_area,
                    )
                depth_stream_chunks.append(depth_frame)
                slice_stream_chunks.append(slice_frame)
                if final_output is not None:
                    final_frame, fallback_count = add_hybrid_columns(
                        slice_frame,
                        ratio_threshold=args.hybrid_ratio_threshold,
                        min_bbox_side_um=args.hybrid_min_bbox_side_um,
                        pixel_pitch_um=acquisition.optics.pixel_pitch_um,
                        reconstruction_start_um=acquisition.optics.reconstruction_start_um,
                        slice_spacing_um=acquisition.optics.slice_spacing_um,
                        enabled=args.diameter_underprediction_fallback,
                    )
                    final_fallback_count += fallback_count
                    final_stream_chunks.append(final_frame)
                flush_stream_chunks(force=False)

    if prefetcher is not None:
        prefetcher.close()

    if csv_writer is not None:
        flush_stream_chunks(force=True)
        csv_writer_close_sec = csv_writer.close()
    else:
        depth_out = df.copy()
        depth_out["slice"] = best_slice
        depth_out["depth_um"] = setup.z0_um + (best_slice.astype(np.float64) - 1.0) * setup.dz_um
        depth_out["model_score"] = best_score
        depth_out["depth_method"] = "focus_score_model_argmax"
        depth_out["depth_router_source"] = np.where(router_to_fallback, "baseline_fallback", "robust_primary")
        depth_out["depth_primary_checkpoint"] = depth_checkpoint_id
        depth_out["depth_fallback_checkpoint"] = depth_fallback_checkpoint_id
        depth_out["model_crop_method"] = (
            "cuda_field_roi_abs2_centered_64px_valid_region_mean_padding"
            if roi_cropper is not None
            else "torch_full_intensity_centered_64px_valid_region_mean_padding"
        )
        depth_output.parent.mkdir(parents=True, exist_ok=True)
        depth_out.to_csv(depth_output, index=False)

        slice_out = depth_out.copy()
        slice_out["slice_diam_pred_um"] = pred
        slice_out["slice_diam_sigma_um"] = sigma
        slice_out["slice_diam_pred_log"] = pred_log
        slice_out["slice_diam_sigma_log"] = sigma_log
        slice_out["slice_diam_model"] = slice_weights_id
        slice_out["slice_diam_input_depth"] = "depth_and_slice_fused.slice"
        if args.recenter_on_slice:
            slice_out = add_recenter_columns(
                slice_out,
                np.arange(len(slice_out), dtype=np.int64),
                recenter_x,
                recenter_y,
                recenter_method,
                recenter_area,
            )
        slice_output.parent.mkdir(parents=True, exist_ok=True)
        slice_out.to_csv(slice_output, index=False)

        if final_output is not None:
            final_out, final_fallback_count = add_hybrid_columns(
                slice_out,
                ratio_threshold=args.hybrid_ratio_threshold,
                min_bbox_side_um=args.hybrid_min_bbox_side_um,
                pixel_pitch_um=acquisition.optics.pixel_pitch_um,
                reconstruction_start_um=acquisition.optics.reconstruction_start_um,
                slice_spacing_um=acquisition.optics.slice_spacing_um,
                enabled=args.diameter_underprediction_fallback,
            )
            final_output.parent.mkdir(parents=True, exist_ok=True)
            final_out.to_csv(final_output, index=False)

    elapsed = time.perf_counter() - started
    metrics = {
        "method": "fused_depth_model_and_slice_diameter_same_frame_holo",
        "input": str(input_path),
        "depth_output": str(depth_output),
        "slice_output": str(slice_output),
        "final_output": str(final_output) if final_output is not None else None,
        "acquisition": acquisition.name,
        "acquisition_config": str(acquisition.source_path),
        "holography_mode": acquisition.mode,
        "model_domain_warning": (
            "Packaged depth and diameter checkpoints were trained on dual-camera phase-retrieval crops; "
            "single-Gabor accuracy is not validated."
            if acquisition.mode == "single_gabor"
            else None
        ),
        "depth_checkpoint": depth_checkpoint_id,
        "depth_fallback_checkpoint": depth_fallback_checkpoint_id,
        "depth_router_enabled": bool(args.depth_router),
        "depth_router_rule": f"diameter_method contains fallback and diameter_um <= {args.depth_router_max_diameter_um:g}",
        "depth_router_primary_count": int((~router_to_fallback).sum()),
        "depth_router_fallback_count": int(router_to_fallback.sum()),
        "slice_weights": slice_weights_id,
        "device": str(device),
        "rows": int(len(df)),
        "processed_rows": int(sum(len(indices) for _, indices in groups)),
        "frames": int(len(groups)),
        "slices": int(len(slices)),
        "slice_start": int(slices[0]),
        "slice_end": int(slices[-1]),
        "slice_step": int(args.slice_step),
        "crop_size": int(args.crop_size),
        "phase_retrieval_iterations": int(acquisition.reconstruction.phase_retrieval_iterations),
        "z0_um": float(setup.z0_um),
        "dz_um": float(setup.dz_um),
        "pixel_pitch_um": float(acquisition.optics.pixel_pitch_um),
        "depth_batch_size": int(args.depth_batch_size),
        "slice_block": int(args.slice_block),
        "padlen": int(padlen),
        "padding_offset": int(offset),
        "slice_diam_frame_batch": bool(args.slice_diam_frame_batch),
        "depth_buffer_reuse": bool(args.depth_buffer_reuse),
        "depth_inplace_propagation": bool(args.depth_inplace_propagation),
        "diam_batch_size": int(args.diam_batch_size),
        "roi_crop_backend": "cuda" if roi_cropper is not None else "torch",
        "depth_model_backend": str(depth_info.get("model_backend", args.model_backend)),
        "depth_inference_precision": str(depth_info.get("inference_precision", "fp32")),
        "depth_model_compile_sec": float(depth_compile_sec),
        "depth_fallback_model_backend": str(depth_fallback_info.get("model_backend", args.model_backend)),
        "depth_fallback_inference_precision": str(depth_fallback_info.get("inference_precision", "fp32")),
        "depth_fallback_model_compile_sec": float(depth_fallback_compile_sec),
        "slice_model_backend": str(slice_backend_info["model_backend"]),
        "slice_inference_precision": str(slice_backend_info["inference_precision"]),
        "slice_model_compile_sec": float(slice_compile_sec),
        "slice_trt_dynamic_max_batch": slice_backend_info["trt_dynamic_max_batch"],
        "roi_kernel_compile_sec": float(roi_kernel_compile_sec),
        "fft_backend": str(args.fft_backend),
        "stream_csv_writes": bool(args.stream_csv_writes),
        "stream_write_chunk_frames": int(args.stream_write_chunk_frames),
        "stream_write_queue_size": int(args.stream_write_queue_size),
        "csv_writer_close_sec": float(csv_writer_close_sec),
        "csv_writer_put_wait_sec": float(csv_writer_put_wait_sec[0]),
        "prefetch_holo_workers": int(args.prefetch_holo_workers),
        "prefetch_holo_frames": int(args.prefetch_holo_frames),
        "recenter_on_slice": bool(args.recenter_on_slice),
        "recenter_success_count": int(recenter_success_count),
        "recenter_fallback_count": int(recenter_fallback_count),
        "recenter_elapsed_sec": float(sum(recenter_times)),
        "mean_recenter_sec_per_unique_frame_slice": float(np.mean(recenter_times)) if recenter_times else None,
        "depth_model": depth_info,
        "slice_checkpoint_epoch": int(slice_ckpt.get("epoch", -1)),
        "slice_model_norm": slice_norm,
        "slice_calibration_scale": float(slice_calibration_scale),
        "depth_crops_scored": int(depth_crops_scored),
        "slice_crops_scored": int(slice_crops_scored),
        "unique_frame_slices": int(unique_frame_slices),
        "elapsed_sec": float(elapsed),
        "mean_holo_sec_per_frame": float(np.mean(holo_times)) if holo_times else None,
        "mean_depth_sec_per_frame": float(np.mean(depth_times)) if depth_times else None,
        "mean_slice_infer_sec_per_frame": float(np.mean(slice_times)) if slice_times else None,
        "depth_crops_per_sec": float(depth_crops_scored / sum(depth_times)) if sum(depth_times) > 0 else None,
        "slice_crops_per_sec_including_holo": float(slice_crops_scored / (sum(slice_times) + sum(holo_times)))
        if sum(slice_times) + sum(holo_times) > 0
        else None,
    }
    write_json(metrics_output, metrics)
    if final_output is not None and hybrid_metrics_output is not None:
        write_json(
            hybrid_metrics_output,
            {
                "input": str(slice_output),
                "output": str(final_output),
                "rows": int(len(df)),
                "ratio_threshold": float(args.hybrid_ratio_threshold),
                "min_bbox_side_um": float(args.hybrid_min_bbox_side_um),
                "enabled": bool(args.diameter_underprediction_fallback),
                "fallback_count": int(final_fallback_count),
                "fallback_fraction": float(final_fallback_count / max(1, len(df))),
                "computed_in": "depth_and_slice_fused.py",
                "stream_csv_writes": bool(args.stream_csv_writes),
            },
        )
    print(json.dumps({"event": "done", **metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
