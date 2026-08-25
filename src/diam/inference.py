"""Production SliceDiamModel inference helpers shared by pipeline backends."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


@torch.inference_mode()
def predict_diameter_batch(
    model: torch.nn.Module,
    crops: torch.Tensor,
    norm: dict[str, float],
    calibration_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict calibrated diameter and uncertainty from reconstructed crops."""

    inputs = torch.round(crops.clamp(0.0, 1.0) * 255.0).div(255.0)[:, None, :, :].contiguous()
    mu_norm, sigma_norm = model(inputs)
    mu_log = mu_norm.detach().float().cpu().numpy() * norm["log_diam_std"] + norm["log_diam_mean"]
    sigma_log = sigma_norm.detach().float().cpu().numpy() * norm["log_diam_std"] * calibration_scale
    prediction_um = np.exp(mu_log)
    sigma_um = prediction_um * sigma_log
    return prediction_um, sigma_um, mu_log, sigma_log


def optimize_slice_model_backend(
    model: torch.nn.Module,
    *,
    backend: str,
    device: torch.device,
    batch_size: int,
    crop_size: int,
) -> tuple[torch.nn.Module, dict[str, Any], float]:
    """Return the Torch model directly or compile it for strict TensorRT FP16."""

    started = time.perf_counter()
    info: dict[str, Any] = {
        "model_backend": backend,
        "inference_precision": "fp16" if backend == "tensorrt" else "fp32",
        "trt_dynamic_max_batch": None,
    }
    if backend == "torch":
        return model, info, time.perf_counter() - started
    if backend != "tensorrt":
        raise RuntimeError(f"Unknown diameter model backend: {backend}")
    if device.type != "cuda":
        raise RuntimeError("The TensorRT diameter backend requires CUDA. Use the Torch backend on CPU.")

    try:
        import torch_tensorrt
    except Exception as exc:
        raise RuntimeError(
            "The TensorRT diameter backend requires torch-tensorrt. Install with `uv sync --extra gpu`."
        ) from exc

    optimized = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 1, crop_size, crop_size),
                opt_shape=(batch_size, 1, crop_size, crop_size),
                max_shape=(batch_size, 1, crop_size, crop_size),
                dtype=torch.float32,
            )
        ],
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        min_block_size=1,
        truncate_double=True,
    )
    optimized.eval()
    info["trt_dynamic_max_batch"] = int(batch_size)
    return optimized, info, time.perf_counter() - started


__all__ = ["optimize_slice_model_backend", "predict_diameter_batch"]
