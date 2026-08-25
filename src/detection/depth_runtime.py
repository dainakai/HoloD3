"""Shared learned-depth model and crop runtime used by the fused pipeline."""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holod3.checkpoints import load_state_checkpoint  # noqa: E402
from src.common.cuda_roi_kernels import CudaRoiAbs2Cropper  # noqa: E402


INPUT_CHANNELS = {
    "raw": 1,
    "raw_mask": 2,
    "raw_diam": 2,
    "raw_diam_scalar": 1,
    "raw_mask_diam": 3,
    "raw_mask_diam_scalar": 2,
}
FULL_DIAM_CHANNEL_MODES = {"raw_diam", "raw_mask_diam"}
SCALAR_DIAM_MODES = {"raw_diam_scalar", "raw_mask_diam_scalar"}


@dataclass(frozen=True)
class TransferSetup:
    variables: dict[str, float]
    coeffs: np.ndarray
    dz_um: float
    datlen: int
    slices: int
    z0_um: float
    d_pr: torch.Tensor
    d_pr_inv: torch.Tensor
    d_tf_unshifted: torch.Tensor
    d_slice_unshifted: torch.Tensor


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


class FocusScoreNet(nn.Module):
    def __init__(
        self,
        width: int = 24,
        in_channels: int = 1,
        cond_dim: int = 0,
        arch: str = "default",
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        if arch == "default":
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
                DSConv(width, width * 2, stride=2),
                DSConv(width * 2, width * 2, stride=1),
                DSConv(width * 2, width * 3, stride=2),
                DSConv(width * 3, width * 4, stride=2),
                DSConv(width * 4, width * 5, stride=2),
                DSConv(width * 5, width * 6, stride=2),
            )
        elif arch == "faststem":
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
                DSConv(width, width * 2, stride=1),
                DSConv(width * 2, width * 3, stride=2),
                DSConv(width * 3, width * 4, stride=2),
                DSConv(width * 4, width * 5, stride=2),
                DSConv(width * 5, width * 6, stride=2),
            )
        else:
            raise ValueError(f"Unknown arch={arch}. Use default or faststem.")
        self.head = nn.Sequential(
            nn.Linear(width * 6 + cond_dim, width * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.08),
            nn.Linear(width * 4, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x).mean(dim=(2, 3))
        if self.cond_dim:
            if cond is None:
                raise RuntimeError("Condition tensor is required for this model")
            feat = torch.cat([feat, cond.to(feat.dtype)], dim=1)
        return self.head(feat).squeeze(1)


def input_channels(input_mode: str) -> int:
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unknown input_mode={input_mode}. Use one of {sorted(INPUT_CHANNELS)}")
    return INPUT_CHANNELS[input_mode]


def condition_dim(input_mode: str) -> int:
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unknown input_mode={input_mode}. Use one of {sorted(INPUT_CHANNELS)}")
    return 1 if input_mode in SCALAR_DIAM_MODES else 0


def normalized_image_name(file_value: Any, imgext: str = "png") -> str:
    stem = Path(str(file_value)).stem
    return f"{stem}.{imgext}"


def fft2(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != "torch":
        raise ValueError(f"Unsupported fft backend: {backend}")
    return torch.fft.fft2(x)


def ifft2(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend != "torch":
        raise ValueError(f"Unsupported fft backend: {backend}")
    return torch.fft.ifft2(x)


def diameter_condition_values(diameter_um: torch.Tensor) -> torch.Tensor:
    lo = math.log(25.0)
    hi = math.log(500.0)
    values = (torch.log(torch.clamp(diameter_um, min=1e-6)) - lo) / max(hi - lo, 1e-6)
    return torch.clamp(values * 2.0 - 1.0, -1.5, 1.5)


def add_condition_channels(
    raw: torch.Tensor,
    diameter_um: torch.Tensor,
    input_mode: str,
    mask_radius_diam_scale: float,
    mask_min_radius_px: float,
    mask_softness_px: float,
    pixel_pitch_um: float = 10.0,
) -> torch.Tensor:
    if input_mode == "raw":
        return raw
    batch, _, crop_size, _ = raw.shape
    channels = [raw]
    if "mask" in input_mode:
        coords = torch.arange(crop_size, device=raw.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (crop_size - 1) / 2.0
        rr = torch.sqrt((xx - center).square() + (yy - center).square())[None, None, :, :]
        if pixel_pitch_um <= 0:
            raise ValueError("pixel_pitch_um must be positive")
        diameter_px = diameter_um.view(batch, 1, 1, 1) / float(pixel_pitch_um)
        radius = torch.maximum(
            torch.full_like(diameter_px, mask_min_radius_px),
            diameter_px * mask_radius_diam_scale,
        )
        mask = torch.sigmoid((radius - rr) / max(mask_softness_px, 1e-3))
        channels.append(mask)
    if input_mode in FULL_DIAM_CHANNEL_MODES:
        values = diameter_condition_values(diameter_um)
        channels.append(values.view(batch, 1, 1, 1).expand(batch, 1, crop_size, crop_size))
    return torch.cat(channels, dim=1)


def make_condition_tensor(diameter_um: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode not in SCALAR_DIAM_MODES:
        return torch.empty((diameter_um.numel(), 0), device=diameter_um.device, dtype=torch.float32)
    return diameter_condition_values(diameter_um).view(-1, 1)


def load_model(
    checkpoint: Path,
    device: torch.device,
    compile_model: bool,
    channels_last: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = load_state_checkpoint(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_info = ckpt.get("model", {})
    input_mode = model_info.get("input_mode") or ckpt_args.get("input_mode", "raw")
    arch = model_info.get("arch", ckpt_args.get("arch", "default"))
    width = int(model_info.get("width", ckpt_args.get("width", 24)))
    in_ch = int(model_info.get("in_channels", input_channels(input_mode)))
    cond_dim = int(model_info.get("cond_dim", condition_dim(input_mode)))
    model = FocusScoreNet(width=width, in_channels=in_ch, cond_dim=cond_dim, arch=arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    if compile_model:
        model = torch.compile(model)
    mask_radius_diam_scale = float(ckpt_args.get("mask_radius_diam_scale", 2.0))
    mask_min_radius_px = float(ckpt_args.get("mask_min_radius_px", 8.0))
    mask_softness_px = float(ckpt_args.get("mask_softness_px", 3.0))
    return model, {
        "input_mode": input_mode,
        "arch": arch,
        "width": width,
        "in_channels": in_ch,
        "cond_dim": cond_dim,
        "mask_radius_diam_scale": mask_radius_diam_scale,
        "mask_min_radius_px": mask_min_radius_px,
        "mask_softness_px": mask_softness_px,
    }


class NoCondWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, None)


def optimize_model_backend(
    model: nn.Module,
    model_info: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float]:
    compile_start = time.perf_counter()
    backend = str(args.model_backend)
    cond_dim = int(model_info.get("cond_dim", 0))
    model_info["model_backend"] = backend
    model_info["inference_precision"] = "fp16" if backend == "tensorrt" else "fp32"
    model_info["trt_dynamic_max_batch"] = None
    model_info["trt_num_inputs"] = 2 if cond_dim > 0 else 1

    if backend == "torch":
        if args.compile_model:
            model = torch.compile(model)
            model_info["model_backend"] = "torch_compile"
        return model, time.perf_counter() - compile_start

    if backend != "tensorrt":
        raise RuntimeError(f"Unknown model backend: {backend}")
    if device.type != "cuda":
        raise RuntimeError("TensorRT backend requires CUDA. Use --model-backend torch on CPU.")

    try:
        import torch_tensorrt
    except Exception as exc:
        raise RuntimeError("TensorRT backend requires torch-tensorrt. Install torch-tensorrt first.") from exc

    batch_size = int(args.batch_size)
    crop_size = int(args.crop_size)
    in_channels = int(model_info["in_channels"])
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, in_channels, crop_size, crop_size),
            opt_shape=(batch_size, in_channels, crop_size, crop_size),
            max_shape=(batch_size, in_channels, crop_size, crop_size),
            dtype=torch.float32,
        )
    ]
    if cond_dim > 0:
        trt_target: nn.Module = model
        inputs.append(
            torch_tensorrt.Input(
                min_shape=(1, cond_dim),
                opt_shape=(batch_size, cond_dim),
                max_shape=(batch_size, cond_dim),
                dtype=torch.float32,
            )
        )
        model_info["trt_num_inputs"] = 2
    else:
        trt_target = NoCondWrapper(model)
        model_info["trt_num_inputs"] = 1

    trt_model = torch_tensorrt.compile(
        trt_target,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
        truncate_double=True,
    )
    trt_model.eval()
    model_info["model_backend"] = "tensorrt"
    model_info["inference_precision"] = "fp16"
    model_info["trt_dynamic_max_batch"] = batch_size
    return trt_model, time.perf_counter() - compile_start


def crop_start_tensors(rows: pd.DataFrame, crop_size: int, datlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = rows["seg_xc"].to_numpy(np.float64)
    y = rows["seg_yc"].to_numpy(np.float64)
    x0 = np.rint(x - crop_size / 2).astype(np.int64)
    y0 = np.rint(y - crop_size / 2).astype(np.int64)
    offsets = torch.arange(crop_size, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    x_idx = torch.from_numpy(x0).to(device=device, dtype=torch.long)[:, None, None] + xx[None, :, :]
    y_idx = torch.from_numpy(y0).to(device=device, dtype=torch.long)[:, None, None] + yy[None, :, :]
    valid = (x_idx >= 0) & (x_idx < datlen) & (y_idx >= 0) & (y_idx < datlen)
    x_idx = x_idx.clamp_(0, datlen - 1)
    y_idx = y_idx.clamp_(0, datlen - 1)
    return x_idx, y_idx, valid


def crop_start_xy_tensors(rows: pd.DataFrame, crop_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = rows["seg_xc"].to_numpy(np.float64)
    y = rows["seg_yc"].to_numpy(np.float64)
    x0 = np.rint(x - crop_size / 2).astype(np.int32)
    y0 = np.rint(y - crop_size / 2).astype(np.int32)
    return (
        torch.from_numpy(x0).to(device=device, dtype=torch.int32).contiguous(),
        torch.from_numpy(y0).to(device=device, dtype=torch.int32).contiguous(),
    )


def centered_crops_with_mean_padding(
    central_intensity: torch.Tensor,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    crops = central_intensity[y_idx, x_idx]
    masked = torch.where(valid, crops, torch.zeros((), device=crops.device, dtype=crops.dtype))
    counts = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    means = masked.sum(dim=(1, 2), keepdim=True) / counts
    return torch.where(valid, crops, means).clamp_(0.0, 1.0)


def run_model_batches(
    model: nn.Module,
    crops: torch.Tensor,
    diameters_um: torch.Tensor,
    model_info: dict[str, Any],
    batch_size: int,
    amp: bool,
    channels_last: bool,
    pixel_pitch_um: float = 10.0,
) -> torch.Tensor:
    scores = torch.empty((crops.shape[0],), device=crops.device, dtype=torch.float32)
    input_mode = str(model_info["input_mode"])
    model_backend = str(model_info.get("model_backend", "torch"))
    trt_num_inputs = int(model_info.get("trt_num_inputs", 2))
    for start in range(0, crops.shape[0], batch_size):
        end = min(start + batch_size, crops.shape[0])
        raw = crops[start:end, None, :, :].to(dtype=torch.float32)
        raw = raw.sub(0.5).div(0.25)
        diam = diameters_um[start:end]
        tensor = add_condition_channels(
            raw,
            diam,
            input_mode,
            float(model_info["mask_radius_diam_scale"]),
            float(model_info["mask_min_radius_px"]),
            float(model_info["mask_softness_px"]),
            pixel_pitch_um,
        )
        cond = make_condition_tensor(diam, input_mode)
        if model_backend == "tensorrt":
            tensor = tensor.contiguous()
            if trt_num_inputs == 1:
                out = model(tensor)
            else:
                out = model(tensor, cond)
        else:
            if channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            if amp and crops.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(tensor, cond)
            else:
                out = model(tensor, cond)
        scores[start:end] = out.detach().float()
    return scores


def slice_values(setup: TransferSetup, args: argparse.Namespace) -> list[int]:
    if args.slice_step <= 0:
        raise ValueError("--slice-step must be positive")
    lo = max(1, min(setup.slices, int(args.slice_start)))
    hi_req = int(args.slice_end) if int(args.slice_end) > 0 else setup.slices
    hi = max(lo, min(setup.slices, hi_req))
    values = list(range(lo, hi + 1, int(args.slice_step)))
    if values[-1] != hi:
        values.append(hi)
    return values
