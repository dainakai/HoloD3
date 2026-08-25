from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from holod3.checkpoints import load_state_checkpoint


CHECKPOINT_FORMAT = "slice_diammodel_v1"


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.bn1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.silu(y + self.skip(x), inplace=True)


class DiameterNet(nn.Module):
    """Canonical SliceDiamModel architecture used by training and production."""

    def __init__(self, min_sigma_norm: float = 0.025, dropout: float = 0.10) -> None:
        super().__init__()
        self.min_sigma_norm = min_sigma_norm
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(32, 32, stride=1, dropout=0.02),
            ResBlock(32, 48, stride=2, dropout=0.02),
            ResBlock(48, 64, stride=1, dropout=0.03),
            ResBlock(64, 96, stride=2, dropout=0.04),
            ResBlock(96, 128, stride=2, dropout=0.05),
            ResBlock(128, 160, stride=2, dropout=0.06),
            ResBlock(160, 192, stride=2, dropout=0.06),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.SiLU(inplace=True),
        )
        self.mu_head = nn.Linear(96, 1)
        self.sigma_head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.blocks(self.stem(x)).mean(dim=(2, 3))
        h = self.head(feat)
        mu = self.mu_head(h).squeeze(1)
        sigma = F.softplus(self.sigma_head(h).squeeze(1)) + self.min_sigma_norm
        return mu, sigma


def _serialized_args(args: Namespace | Mapping[str, Any]) -> dict[str, Any]:
    values = vars(args) if isinstance(args, Namespace) else dict(args)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def save_model_checkpoint(
    path: Path,
    model: DiameterNet,
    args: Namespace | Mapping[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    norm: dict[str, float],
    calibration_scale: float,
) -> None:
    path = checkpoint_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model_class": "src.diam.model.DiameterNet",
            "model": model.state_dict(),
            "args": _serialized_args(args),
            "epoch": epoch,
            "metrics": metrics,
            "norm": norm,
            "calibration_scale": calibration_scale,
        },
        path,
    )


def load_model(
    weights: Path,
    device: torch.device,
) -> tuple[DiameterNet, dict[str, float], float, dict[str, Any]]:
    """Load both historical checkpoints and checkpoints saved by this workspace."""
    weights = checkpoint_path(weights)
    ckpt = load_state_checkpoint(weights, map_location=device)
    if "model" not in ckpt or "norm" not in ckpt:
        raise RuntimeError(f"Invalid SliceDiamModel checkpoint (missing model/norm): {weights}")
    model = DiameterNet()
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    norm = ckpt["norm"]
    return model, norm, float(ckpt.get("calibration_scale", 1.0)), ckpt


__all__ = ["CHECKPOINT_FORMAT", "DiameterNet", "ResBlock", "load_model", "save_model_checkpoint"]
