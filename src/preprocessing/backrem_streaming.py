"""Two-pass temporal background removal with bounded image memory."""

from __future__ import annotations

import importlib.util
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.backrem_adaptive import (
    AdaptiveConfig,
    correct_with_backgrounds,
    load_gray,
    make_anchor_positions,
    save_gray,
)


def load_stack_for_paths(paths: list[Path]) -> np.ndarray:
    frames = [load_gray(path) for path in paths]
    shape = frames[0].shape
    for path, frame in zip(paths, frames, strict=True):
        if frame.shape != shape:
            raise ValueError(f"Image size mismatch in {path}: got {frame.shape}, expected {shape}")
    return np.stack(frames, axis=0)


def load_window_cached(
    paths: list[Path],
    lo: int,
    hi: int,
    cache: dict[int, np.ndarray],
    expected_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int], int]:
    for idx in list(cache):
        if idx < lo or idx >= hi:
            del cache[idx]

    loaded = 0
    shape = expected_shape
    for idx in range(lo, hi):
        if idx in cache:
            continue
        frame = load_gray(paths[idx])
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(f"Image size mismatch in {paths[idx]}: got {frame.shape}, expected {shape}")
        cache[idx] = frame
        loaded += 1

    if shape is None:
        raise ValueError("Cannot load an empty image window")
    return np.stack([cache[idx] for idx in range(lo, hi)], axis=0), shape, loaded


def anchor_path(anchor_dir: Path, anchor_idx: int, frame_pos: int) -> Path:
    return anchor_dir / f"anchor_{anchor_idx:04d}_frame_{frame_pos:06d}.npy"


def estimate_anchor_backgrounds_to_disk(
    paths: list[Path],
    config: AdaptiveConfig,
    device: torch.device,
    anchor_dir: Path,
) -> tuple[list[int], list[float], dict[str, float]]:
    n_frames = len(paths)
    anchors = make_anchor_positions(n_frames, config.anchor_stride)
    half = max(1, config.window) // 2
    means: list[float] = []
    cache: dict[int, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    source_image_reads = 0
    anchor_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, pos in enumerate(anchors):
            lo = max(0, pos - half)
            hi = min(n_frames, pos + half + 1)
            stack, expected_shape, loaded = load_window_cached(paths, lo, hi, cache, expected_shape)
            source_image_reads += loaded
            window = torch.from_numpy(stack).to(device=device, dtype=torch.float32, non_blocking=True)
            bg = torch.median(window, dim=0).values.cpu().numpy().astype(np.float16)
            np.save(anchor_path(anchor_dir, i, pos), bg)
            means.append(float(bg.mean(dtype=np.float32)))
            del stack, window, bg
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return anchors, means, {
        "estimate_background_sec": time.perf_counter() - t0,
        "anchor_source_image_reads": source_image_reads,
        "anchor_window_image_visits": sum(min(n_frames, pos + half + 1) - max(0, pos - half) for pos in anchors),
    }


def stack_range_from_cache(
    paths: list[Path],
    lo: int,
    hi: int,
    cache: dict[int, np.ndarray],
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    loaded = 0
    for idx in range(lo, hi):
        if idx in cache:
            continue
        frame = load_gray(paths[idx])
        if frame.shape != expected_shape:
            raise ValueError(f"Image size mismatch in {paths[idx]}: got {frame.shape}, expected {expected_shape}")
        cache[idx] = frame
        loaded += 1
    return np.stack([cache[idx] for idx in range(lo, hi)], axis=0), loaded


def estimate_anchor_backgrounds_cuda_to_disk(
    paths: list[Path],
    config: AdaptiveConfig,
    device: torch.device,
    anchor_dir: Path,
) -> tuple[list[int], list[float], dict[str, float]]:
    if device.type != "cuda":
        raise RuntimeError("CUDA median backend requires a CUDA device")
    from src.preprocessing.cuda_median_anchor import median_u8_time

    n_frames = len(paths)
    anchors = make_anchor_positions(n_frames, config.anchor_stride)
    half = max(1, config.window) // 2
    means: list[float] = []
    cache: dict[int, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    source_image_reads = 0
    anchor_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, pos in enumerate(anchors):
            lo = max(0, pos - half)
            hi = min(n_frames, pos + half + 1)
            stack_np, expected_shape, loaded = load_window_cached(paths, lo, hi, cache, expected_shape)
            source_image_reads += loaded
            stack = torch.from_numpy(stack_np).to(device=device, dtype=torch.uint8, non_blocking=True)
            bg_u8 = median_u8_time(stack)
            bg = bg_u8.cpu().numpy().astype(np.float16)
            np.save(anchor_path(anchor_dir, i, pos), bg)
            means.append(float(bg.mean(dtype=np.float32)))
            del stack_np, stack, bg_u8, bg
    torch.cuda.synchronize(device)
    return anchors, means, {
        "estimate_background_sec": time.perf_counter() - t0,
        "median_backend": "cuda_nvrtc_histogram",
        "anchor_source_image_reads": source_image_reads,
        "anchor_window_image_visits": sum(min(n_frames, pos + half + 1) - max(0, pos - half) for pos in anchors),
    }


def estimate_anchor_backgrounds_rolling_cuda_to_disk(
    paths: list[Path],
    config: AdaptiveConfig,
    device: torch.device,
    anchor_dir: Path,
) -> tuple[list[int], list[float], dict[str, float]]:
    if device.type != "cuda":
        raise RuntimeError("CUDA rolling median backend requires a CUDA device")
    if config.window > 255:
        raise ValueError("rolling uint8 histogram backend requires --window <= 255")
    from src.preprocessing.cuda_rolling_median import get_rolling_median_kernel

    n_frames = len(paths)
    anchors = make_anchor_positions(n_frames, config.anchor_stride)
    half = max(1, config.window) // 2
    first = load_gray(paths[0])
    height, width = first.shape
    total = height * width
    cache: dict[int, np.ndarray] = {0: first}
    source_image_reads = 1
    gpu_frame_updates = 0
    means: list[float] = []
    anchor_dir.mkdir(parents=True, exist_ok=True)

    kernel = get_rolling_median_kernel(device.index if device.index is not None else torch.cuda.current_device())
    hist = torch.zeros((total * 256,), device=device, dtype=torch.uint8)
    current_lo = 0
    current_hi = 0

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, pos in enumerate(anchors):
            target_lo = max(0, pos - half)
            target_hi = min(n_frames, pos + half + 1)

            remove_hi = min(target_lo, current_hi)
            if remove_hi > current_lo:
                remove_stack_np, _ = stack_range_from_cache(paths, current_lo, remove_hi, cache, (height, width))
                remove_stack = torch.from_numpy(remove_stack_np).to(device=device, dtype=torch.uint8, non_blocking=True)
                kernel.update_hist(remove_stack, hist, -1)
                gpu_frame_updates += remove_hi - current_lo
                for idx in range(current_lo, remove_hi):
                    cache.pop(idx, None)
                del remove_stack_np, remove_stack

            add_lo = max(current_hi, target_lo)
            if target_hi > add_lo:
                add_stack_np, loaded = stack_range_from_cache(paths, add_lo, target_hi, cache, (height, width))
                source_image_reads += loaded
                add_stack = torch.from_numpy(add_stack_np).to(device=device, dtype=torch.uint8, non_blocking=True)
                kernel.update_hist(add_stack, hist, +1)
                gpu_frame_updates += target_hi - add_lo
                del add_stack_np, add_stack
            current_lo, current_hi = target_lo, target_hi

            window_count = target_hi - target_lo
            bg_u8 = kernel.median_from_hist(hist, height, width, window_count)
            bg = bg_u8.cpu().numpy().astype(np.float16)
            np.save(anchor_path(anchor_dir, i, pos), bg)
            means.append(float(bg.mean(dtype=np.float32)))
            del bg_u8, bg
    torch.cuda.synchronize(device)
    return anchors, means, {
        "estimate_background_sec": time.perf_counter() - t0,
        "median_backend": "cuda_nvrtc_rolling_histogram",
        "anchor_source_image_reads": source_image_reads,
        "anchor_window_image_visits": sum(min(n_frames, pos + half + 1) - max(0, pos - half) for pos in anchors),
        "gpu_frame_hist_updates": gpu_frame_updates,
    }


def resolve_median_backend(requested: str, device: torch.device, window: int) -> str:
    if requested not in {"auto", "torch", "cuda-window", "cuda-rolling"}:
        raise ValueError(f"Unknown background median backend: {requested}")
    if requested != "auto":
        if requested.startswith("cuda-") and device.type != "cuda":
            raise ValueError(f"Background median backend {requested} requires CUDA")
        return requested
    if device.type != "cuda":
        return "torch"
    try:
        available = importlib.util.find_spec("cuda.bindings.nvrtc") is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        return "torch"
    if window <= 255:
        return "cuda-rolling"
    return "cuda-window"


def load_anchor(anchor_dir: Path, anchor_idx: int, frame_pos: int) -> np.ndarray:
    return np.load(anchor_path(anchor_dir, anchor_idx, frame_pos)).astype(np.float16)


def interpolated_background_batch(
    left_bg: np.ndarray,
    right_bg: np.ndarray,
    left_pos: int,
    right_pos: int,
    frame_positions: list[int],
) -> np.ndarray:
    if right_pos == left_pos:
        return np.repeat(left_bg[None, :, :], len(frame_positions), axis=0)
    out = np.empty((len(frame_positions), *left_bg.shape), dtype=np.float16)
    denom = float(right_pos - left_pos)
    left32 = left_bg.astype(np.float32)
    right32 = right_bg.astype(np.float32)
    for i, pos in enumerate(frame_positions):
        alpha = float(pos - left_pos) / denom
        out[i] = ((1.0 - alpha) * left32 + alpha * right32).astype(np.float16)
    return out


def correct_frames_streaming(
    paths: list[Path],
    output_dir: Path,
    config: AdaptiveConfig,
    device: torch.device,
    anchor_dir: Path,
    anchors: list[int],
    target_level: float,
    save_workers: int,
    *,
    selected_positions: set[int] | None = None,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_count = 0
    batch_size = max(1, config.batch_size)
    correction_config = AdaptiveConfig(
        window=config.window,
        anchor_stride=config.anchor_stride,
        batch_size=batch_size,
        lowpass=config.lowpass,
        target_level=target_level,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    futures: list[Future[None]] = []
    max_pending_saves = max(1, save_workers * 4)

    def submit_saves(executor: ThreadPoolExecutor | None, src_paths: list[Path], corrected: np.ndarray) -> None:
        nonlocal corrected_count
        for src, arr in zip(src_paths, corrected, strict=True):
            out_path = output_dir / f"{src.stem}.png"
            if executor is None:
                save_gray(out_path, arr)
            else:
                futures.append(executor.submit(save_gray, out_path, arr.copy()))
                while len(futures) >= max_pending_saves:
                    futures.pop(0).result()
            corrected_count += 1

    def wait_for_saves() -> None:
        for future in futures:
            future.result()
        futures.clear()

    executor = ThreadPoolExecutor(max_workers=save_workers) if save_workers > 1 else None

    def positions_between(lo: int, hi: int) -> list[int]:
        return [pos for pos in range(lo, hi) if selected_positions is None or pos in selected_positions]

    try:
        if len(anchors) == 1:
            bg = load_anchor(anchor_dir, 0, anchors[0])
            for lo in range(0, len(paths), batch_size):
                hi = min(len(paths), lo + batch_size)
                positions = positions_between(lo, hi)
                if not positions:
                    continue
                batch_paths = [paths[pos] for pos in positions]
                stack = load_stack_for_paths(batch_paths)
                bgs = np.repeat(bg[None, :, :], len(positions), axis=0)
                corrected, _, _ = correct_with_backgrounds(stack, bgs, correction_config, device, synchronize=False)
                submit_saves(executor, batch_paths, corrected)
            wait_for_saves()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return {"correct_sec": time.perf_counter() - t0, "written_frames": corrected_count}

        for anchor_idx in range(len(anchors) - 1):
            left_pos = anchors[anchor_idx]
            right_pos = anchors[anchor_idx + 1]
            left_bg = load_anchor(anchor_dir, anchor_idx, left_pos)
            right_bg = load_anchor(anchor_dir, anchor_idx + 1, right_pos)
            for lo in range(left_pos, right_pos, batch_size):
                hi = min(right_pos, lo + batch_size)
                frame_positions = positions_between(lo, hi)
                if not frame_positions:
                    continue
                batch_paths = [paths[pos] for pos in frame_positions]
                stack = load_stack_for_paths(batch_paths)
                bgs = interpolated_background_batch(left_bg, right_bg, left_pos, right_pos, frame_positions)
                corrected, _, _ = correct_with_backgrounds(stack, bgs, correction_config, device, synchronize=False)
                submit_saves(executor, batch_paths, corrected)
            del left_bg, right_bg

        last_pos = anchors[-1]
        if selected_positions is None or last_pos in selected_positions:
            last_bg = load_anchor(anchor_dir, len(anchors) - 1, last_pos)
            stack = load_stack_for_paths([paths[last_pos]])
            bgs = np.repeat(last_bg[None, :, :], 1, axis=0)
            corrected, _, _ = correct_with_backgrounds(stack, bgs, correction_config, device, synchronize=False)
            submit_saves(executor, [paths[last_pos]], corrected)
        wait_for_saves()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"correct_sec": time.perf_counter() - t0, "written_frames": corrected_count}
