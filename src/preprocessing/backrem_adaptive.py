"""Temporal background subtraction and residual illumination correction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class AdaptiveConfig:
    window: int = 129
    anchor_stride: int = 32
    batch_size: int = 8
    lowpass: int = 0
    target_level: float | None = None


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        with Image.open(path) as im:
            if np.asarray(im).dtype != np.uint8:
                raise ValueError(f"Background removal requires 8-bit images: {path}")
            img = np.asarray(im.convert("L"), dtype=np.uint8)
    if img.dtype != np.uint8:
        raise ValueError(f"Background removal requires 8-bit images: {path} ({img.dtype})")
    return img


def save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).save(path)


def make_anchor_positions(n_frames: int, stride: int) -> list[int]:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    positions = list(range(0, n_frames, max(1, stride)))
    if positions[-1] != n_frames - 1:
        positions.append(n_frames - 1)
    return positions


def _smooth_profiles_1d(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    if sigma <= 0 or x.shape[-1] <= 1:
        return x
    radius = min(max(2, int(round(sigma * 3))), x.shape[-1] - 1)
    idx = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(idx * idx) / (2.0 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    y = x.unsqueeze(1)
    y = F.pad(y, (radius, radius), mode="reflect")
    return F.conv1d(y, kernel).squeeze(1)


def correct_with_backgrounds(
    stack: np.ndarray,
    backgrounds: np.ndarray,
    config: AdaptiveConfig,
    device: torch.device,
    *,
    synchronize: bool = True,
) -> tuple[np.ndarray, dict[str, float], float]:
    if stack.shape != backgrounds.shape:
        raise ValueError(f"shape mismatch: stack {stack.shape}, backgrounds {backgrounds.shape}")
    n_frames, height, width = stack.shape
    corrected = np.empty_like(stack, dtype=np.uint8)
    target = float(config.target_level) if config.target_level is not None else float(np.median(backgrounds.mean(axis=(1, 2))))
    batch = max(1, config.batch_size)

    if synchronize and device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for lo in range(0, n_frames, batch):
            hi = min(n_frames, lo + batch)
            raw = torch.from_numpy(stack[lo:hi]).to(device=device, dtype=torch.float32, non_blocking=True)
            bg = torch.from_numpy(backgrounds[lo:hi].astype(np.float32)).to(device=device, dtype=torch.float32, non_blocking=True)
            resid = raw - bg

            center = resid.flatten(1).median(dim=1).values.view(-1, 1, 1)
            resid = resid - center

            row = resid.median(dim=2).values
            row = row - row.median(dim=1).values[:, None]
            row = _smooth_profiles_1d(row, sigma=3.0)
            resid = resid - row[:, :, None]

            col = resid.median(dim=1).values
            col = col - col.median(dim=1).values[:, None]
            col = _smooth_profiles_1d(col, sigma=3.0)
            resid = resid - col[:, None, :]

            if config.lowpass and config.lowpass > 1:
                k = int(config.lowpass)
                if k % 2 == 0:
                    k += 1
                pad = k // 2
                if pad >= min(height, width):
                    raise ValueError("background_removal.lowpass is too large for the input image dimensions")
                low = F.avg_pool2d(F.pad(resid[:, None, :, :], (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
                resid = resid - low[:, 0, :, :]

            out = torch.clamp(torch.round(resid + target), 0, 255).to(torch.uint8)
            corrected[lo:hi] = out.cpu().numpy()
            del raw, bg, resid, out

    if synchronize and device.type == "cuda":
        torch.cuda.synchronize(device)
    return corrected, {"correct_sec": time.perf_counter() - t0}, target
