from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


@dataclass(frozen=True)
class PairBatchStats:
    loss: float
    pair_loss: float
    score_reg_loss: float
    acc: float
    n: int


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


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def input_channels(input_mode: str) -> int:
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unknown input_mode={input_mode}. Use one of {sorted(INPUT_CHANNELS)}")
    return INPUT_CHANNELS[input_mode]


def condition_dim(input_mode: str) -> int:
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unknown input_mode={input_mode}. Use one of {sorted(INPUT_CHANNELS)}")
    return 1 if input_mode in SCALAR_DIAM_MODES else 0


def diameter_value(diameter_um: float, min_um: float = 25.0, max_um: float = 500.0) -> float:
    lo = math.log(min_um)
    hi = math.log(max_um)
    v = (math.log(max(float(diameter_um), 1e-6)) - lo) / max(hi - lo, 1e-6)
    return float(np.clip(v * 2.0 - 1.0, -1.5, 1.5))


def soft_center_mask(
    crop_size: int,
    diameter_um: float,
    radius_diam_scale: float,
    min_radius_px: float,
    softness_px: float,
) -> np.ndarray:
    diameter_px = float(diameter_um) / 10.0
    radius = max(min_radius_px, radius_diam_scale * diameter_px)
    yy, xx = np.mgrid[0:crop_size, 0:crop_size].astype(np.float32)
    c = (crop_size - 1) / 2.0
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    mask = 1.0 / (1.0 + np.exp((rr - radius) / max(softness_px, 1e-3)))
    return mask.astype(np.float32)


def make_input_tensor(
    arr: np.ndarray,
    diameter_um: float,
    input_mode: str,
    mask_radius_diam_scale: float,
    mask_min_radius_px: float,
    mask_softness_px: float,
) -> np.ndarray:
    raw = ((arr.astype(np.float32) - 0.5) / 0.25)[None, :, :]
    channels = [raw]
    if "mask" in input_mode:
        mask = soft_center_mask(
            arr.shape[0],
            diameter_um,
            mask_radius_diam_scale,
            mask_min_radius_px,
            mask_softness_px,
        )
        channels.append(mask[None, :, :])
    if input_mode in FULL_DIAM_CHANNEL_MODES:
        value = diameter_value(diameter_um)
        channels.append(np.full((1, arr.shape[0], arr.shape[1]), value, dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(channels, axis=0), dtype=np.float32)


def make_condition_tensor(diameter_um: float, input_mode: str) -> np.ndarray:
    if input_mode in SCALAR_DIAM_MODES:
        return np.asarray([diameter_value(diameter_um)], dtype=np.float32)
    return np.zeros((0,), dtype=np.float32)


def shift_edge(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return arr
    h, w = arr.shape
    pad_x = abs(dx)
    pad_y = abs(dy)
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[y0 : y0 + h, x0 : x0 + w]


def augment_pixels(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        arr = np.flip(arr, axis=1)
    if rng.random() < 0.5:
        arr = np.flip(arr, axis=0)
    if rng.random() < 0.35:
        arr = np.rot90(arr, rng.randint(0, 3))
    if rng.random() < 0.8:
        arr = shift_edge(arr, rng.randint(-2, 2), rng.randint(-2, 2))

    scale = rng.uniform(0.88, 1.12)
    offset = rng.uniform(-0.045, 0.045)
    gamma = rng.uniform(0.88, 1.12)
    arr = np.clip(arr * scale + offset, 0.0, 1.0)
    arr = np.clip(arr, 1e-4, 1.0) ** gamma

    if rng.random() < 0.75:
        noise = np.random.normal(0.0, rng.uniform(0.002, 0.014), size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)

    if rng.random() < 0.12:
        h, w = arr.shape
        side = rng.randint(4, 9)
        x0 = rng.randint(0, max(0, w - side))
        y0 = rng.randint(0, max(0, h - side))
        arr = arr.copy()
        arr[y0 : y0 + side, x0 : x0 + side] = float(np.median(arr))

    return np.ascontiguousarray(arr, dtype=np.float32)


def blend_outer_with_distractor(
    arr: np.ndarray,
    distractor: np.ndarray,
    diameter_um: float,
    rng: random.Random,
    radius_diam_scale: float,
    min_radius_px: float,
    softness_px: float,
) -> np.ndarray:
    mask = soft_center_mask(arr.shape[0], diameter_um, radius_diam_scale, min_radius_px, softness_px)
    alpha = rng.uniform(0.15, 0.55)
    outer = alpha * arr + (1.0 - alpha) * distractor
    return np.clip(arr * mask + outer * (1.0 - mask), 0.0, 1.0).astype(np.float32)


def shifted_neighbor_residual(
    arr: np.ndarray,
    neighbor: np.ndarray,
    diameter_um: float,
    rng: random.Random,
    radius_diam_scale: float,
    min_radius_px: float,
    softness_px: float,
) -> np.ndarray:
    crop_size = arr.shape[0]
    diameter_px = float(diameter_um) / 10.0
    min_shift = max(8.0, 1.7 * diameter_px)
    max_shift = max(min_shift + 1.0, crop_size * 0.48)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    radius = rng.uniform(min_shift, max_shift)
    dx = int(round(math.cos(angle) * radius))
    dy = int(round(math.sin(angle) * radius))
    residual = neighbor.astype(np.float32) - float(np.median(neighbor))
    residual = shift_edge(residual, dx, dy)
    center_keep = soft_center_mask(arr.shape[0], diameter_um, radius_diam_scale, min_radius_px, softness_px)
    beta = rng.uniform(0.15, 0.65)
    return np.clip(arr + beta * residual * (1.0 - center_keep), 0.0, 1.0).astype(np.float32)


class DepthPairDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        manifest_dir: Path,
        split: str,
        pairs_per_epoch: int,
        min_delta_um: float,
        training: bool,
        seed: int,
        input_mode: str,
        mask_radius_diam_scale: float,
        mask_min_radius_px: float,
        mask_softness_px: float,
        outer_swap_prob: float,
        outer_swap_radius_diam_scale: float,
        neighbor_mix_prob: float,
        near_far_prob: float,
        cache_images: bool,
    ) -> None:
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.manifest_dir = manifest_dir
        self.pairs_per_epoch = pairs_per_epoch
        self.min_delta_um = min_delta_um
        self.training = training
        self.rng = random.Random(seed)
        self.cache: dict[int, np.ndarray] = {}
        self.input_mode = input_mode
        self.mask_radius_diam_scale = mask_radius_diam_scale
        self.mask_min_radius_px = mask_min_radius_px
        self.mask_softness_px = mask_softness_px
        self.outer_swap_prob = outer_swap_prob
        self.outer_swap_radius_diam_scale = outer_swap_radius_diam_scale
        self.neighbor_mix_prob = neighbor_mix_prob
        self.near_far_prob = near_far_prob

        groups: list[list[int]] = []
        for _, group in self.rows.groupby("sample_id", sort=False):
            if len(group) >= 2:
                groups.append(group.index.to_list())
        if not groups:
            raise RuntimeError(f"No usable groups for split={split}")
        self.groups = groups
        if cache_images:
            for row_idx, row in self.rows.iterrows():
                self.cache[int(row_idx)] = read_gray(self.manifest_dir / str(row["crop_path"]))

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def load_crop(self, row_idx: int) -> np.ndarray:
        row = self.rows.loc[row_idx]
        if row_idx not in self.cache:
            path = self.manifest_dir / str(row["crop_path"])
            self.cache[row_idx] = read_gray(path)
        arr = self.cache[row_idx]
        if self.training:
            if self.outer_swap_prob > 0.0 and self.rng.random() < self.outer_swap_prob:
                other_idx = self.rng.randrange(len(self.rows))
                if other_idx not in self.cache:
                    self.cache[other_idx] = read_gray(self.manifest_dir / str(self.rows.loc[other_idx, "crop_path"]))
                arr = blend_outer_with_distractor(
                    arr,
                    self.cache[other_idx],
                    float(row["diameter_um"]),
                    self.rng,
                    self.outer_swap_radius_diam_scale,
                    self.mask_min_radius_px,
                    self.mask_softness_px,
                )
            if self.neighbor_mix_prob > 0.0 and self.rng.random() < self.neighbor_mix_prob:
                other_idx = self.rng.randrange(len(self.rows))
                if other_idx not in self.cache:
                    self.cache[other_idx] = read_gray(self.manifest_dir / str(self.rows.loc[other_idx, "crop_path"]))
                arr = shifted_neighbor_residual(
                    arr,
                    self.cache[other_idx],
                    float(row["diameter_um"]),
                    self.rng,
                    self.outer_swap_radius_diam_scale,
                    self.mask_min_radius_px,
                    self.mask_softness_px,
                )
            arr = augment_pixels(arr, self.rng)
        else:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        return make_input_tensor(
            arr,
            float(row["diameter_um"]),
            self.input_mode,
            self.mask_radius_diam_scale,
            self.mask_min_radius_px,
            self.mask_softness_px,
        )

    def choose_pair(self) -> tuple[int, int]:
        if self.training and self.near_far_prob > 0.0 and self.rng.random() < self.near_far_prob:
            for _ in range(32):
                group = self.rng.choice(self.groups)
                ordered = sorted(group, key=lambda idx: float(self.rows.loc[idx, "focus_dist_um"]))
                near_n = max(1, len(ordered) // 4)
                far_n = max(1, len(ordered) // 2)
                near = self.rng.choice(ordered[:near_n])
                far = self.rng.choice(ordered[-far_n:])
                if abs(float(self.rows.loc[near, "focus_dist_um"]) - float(self.rows.loc[far, "focus_dist_um"])) >= self.min_delta_um:
                    if self.rng.random() < 0.5:
                        return near, far
                    return far, near
        for _ in range(32):
            group = self.rng.choice(self.groups)
            a, b = self.rng.sample(group, 2)
            da = float(self.rows.loc[a, "focus_dist_um"])
            db = float(self.rows.loc[b, "focus_dist_um"])
            if abs(da - db) >= self.min_delta_um:
                return a, b
        group = self.rng.choice(self.groups)
        return tuple(self.rng.sample(group, 2))  # type: ignore[return-value]

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        a, b = self.choose_pair()
        row_a = self.rows.loc[a]
        row_b = self.rows.loc[b]
        da = float(row_a["focus_dist_um"])
        db = float(row_b["focus_dist_um"])
        target = 1.0 if da < db else 0.0
        return {
            "left": torch.from_numpy(self.load_crop(a)),
            "right": torch.from_numpy(self.load_crop(b)),
            "left_cond": torch.from_numpy(make_condition_tensor(float(row_a["diameter_um"]), self.input_mode)),
            "right_cond": torch.from_numpy(make_condition_tensor(float(row_b["diameter_um"]), self.input_mode)),
            "target": torch.tensor(target, dtype=torch.float32),
            "left_dist_um": torch.tensor(da, dtype=torch.float32),
            "right_dist_um": torch.tensor(db, dtype=torch.float32),
        }


class DepthCropEvalDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        manifest_dir: Path,
        split: str,
        input_mode: str,
        mask_radius_diam_scale: float,
        mask_min_radius_px: float,
        mask_softness_px: float,
        cache_images: bool,
    ) -> None:
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.manifest_dir = manifest_dir
        self.input_mode = input_mode
        self.mask_radius_diam_scale = mask_radius_diam_scale
        self.mask_min_radius_px = mask_min_radius_px
        self.mask_softness_px = mask_softness_px
        self.cache: dict[int, np.ndarray] = {}
        if cache_images:
            for row_idx, row in self.rows.iterrows():
                self.cache[int(row_idx)] = read_gray(self.manifest_dir / str(row["crop_path"]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        row = self.rows.loc[index]
        arr = self.cache.get(index)
        if arr is None:
            arr = read_gray(self.manifest_dir / str(row["crop_path"]))
        arr = make_input_tensor(
            arr,
            float(row["diameter_um"]),
            self.input_mode,
            self.mask_radius_diam_scale,
            self.mask_min_radius_px,
            self.mask_softness_px,
        )
        return {
            "image": torch.from_numpy(arr),
            "cond": torch.from_numpy(make_condition_tensor(float(row["diameter_um"]), self.input_mode)),
            "sample_id": str(row["sample_id"]),
            "slice": int(row["slice"]),
            "z_rel_um": float(row["z_rel_um"]),
            "z_true_rel_um": float(row["z_true_rel_um"]),
            "focus_dist_um": float(row["focus_dist_um"]),
        }


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
        self.arch = arch
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


def closeness_target(dist_um: torch.Tensor, focus_scale_um: float) -> torch.Tensor:
    return torch.exp(-dist_um / focus_scale_um).clamp(0.0, 1.0)


def run_epoch(
    model: FocusScoreNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pair_temperature: float,
    score_reg_weight: float,
    focus_scale_um: float,
) -> PairBatchStats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pair_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total_n = 0

    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        left_cond = batch["left_cond"].to(device, non_blocking=True)
        right_cond = batch["right_cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        left_dist = batch["left_dist_um"].to(device, non_blocking=True)
        right_dist = batch["right_dist_um"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            left_score = model(left, left_cond)
            right_score = model(right, right_cond)
            logits = (left_score - right_score) / pair_temperature
            pair_loss = F.binary_cross_entropy_with_logits(logits, target)
            reg_left = F.mse_loss(torch.sigmoid(left_score), closeness_target(left_dist, focus_scale_um))
            reg_right = F.mse_loss(torch.sigmoid(right_score), closeness_target(right_dist, focus_scale_um))
            reg_loss = 0.5 * (reg_left + reg_right)
            loss = pair_loss + score_reg_weight * reg_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        pred = (logits.detach() > 0).float()
        total_correct += int((pred == target).sum().item())
        n = int(target.numel())
        total_n += n
        total_loss += float(loss.detach().item()) * n
        total_pair_loss += float(pair_loss.detach().item()) * n
        total_reg_loss += float(reg_loss.detach().item()) * n

    return PairBatchStats(
        loss=total_loss / max(total_n, 1),
        pair_loss=total_pair_loss / max(total_n, 1),
        score_reg_loss=total_reg_loss / max(total_n, 1),
        acc=total_correct / max(total_n, 1),
        n=total_n,
    )


@torch.no_grad()
def evaluate_argmax(
    model: FocusScoreNet,
    manifest: pd.DataFrame,
    manifest_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    input_mode: str,
    mask_radius_diam_scale: float,
    mask_min_radius_px: float,
    mask_softness_px: float,
    cache_images: bool,
) -> dict[str, float]:
    ds = DepthCropEvalDataset(
        manifest,
        manifest_dir,
        split,
        input_mode,
        mask_radius_diam_scale,
        mask_min_radius_px,
        mask_softness_px,
        cache_images,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)
        scores = model(image, cond).detach().cpu().numpy()
        for i, score in enumerate(scores):
            rows.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "slice": int(batch["slice"][i]),
                    "z_rel_um": float(batch["z_rel_um"][i]),
                    "z_true_rel_um": float(batch["z_true_rel_um"][i]),
                    "focus_dist_um": float(batch["focus_dist_um"][i]),
                    "score": float(score),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"argmax_mae_um": math.nan, "argmax_median_abs_um": math.nan, "argmax_samples": 0}

    pred_rows = df.loc[df.groupby("sample_id")["score"].idxmax()].copy()
    abs_err = (pred_rows["z_rel_um"] - pred_rows["z_true_rel_um"]).abs().to_numpy()
    oracle_rows = df.loc[df.groupby("sample_id")["focus_dist_um"].idxmin()].copy()
    oracle_err = oracle_rows["focus_dist_um"].to_numpy()
    return {
        "argmax_mae_um": float(abs_err.mean()),
        "argmax_median_abs_um": float(np.median(abs_err)),
        "argmax_p90_abs_um": float(np.percentile(abs_err, 90)),
        "argmax_samples": int(len(abs_err)),
        "oracle_grid_mae_um": float(oracle_err.mean()),
        "oracle_grid_median_abs_um": float(np.median(oracle_err)),
    }


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "split",
        "sample_id",
        "crop_path",
        "focus_dist_um",
        "z_rel_um",
        "z_true_rel_um",
        "slice",
        "diameter_um",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {missing}")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the fallback focus-nearness model from depth crop pairs.")
    p.add_argument("--manifest", type=Path, default=Path("data/downloaded/depth-fallback/manifest.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/training/depth-fallback"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pairs-per-epoch", type=int, default=20000)
    p.add_argument("--valid-pairs", type=int, default=6000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--arch", choices=["default", "faststem"], default="default")
    p.add_argument("--input-mode", choices=sorted(INPUT_CHANNELS), default="raw")
    p.add_argument("--mask-radius-diam-scale", type=float, default=2.0)
    p.add_argument("--mask-min-radius-px", type=float, default=8.0)
    p.add_argument("--mask-softness-px", type=float, default=3.0)
    p.add_argument("--outer-swap-prob", type=float, default=0.0)
    p.add_argument("--outer-swap-radius-diam-scale", type=float, default=2.5)
    p.add_argument("--neighbor-mix-prob", type=float, default=0.0)
    p.add_argument("--near-far-prob", type=float, default=0.0)
    p.add_argument("--min-delta-um", type=float, default=50.0)
    p.add_argument("--pair-temperature", type=float, default=0.35)
    p.add_argument("--score-reg-weight", type=float, default=0.15)
    p.add_argument("--focus-scale-um", type=float, default=1200.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-images", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    manifest = load_manifest(args.manifest)
    manifest_dir = args.manifest.resolve().parent

    train_ds = DepthPairDataset(
        manifest,
        manifest_dir,
        split="train",
        pairs_per_epoch=args.pairs_per_epoch,
        min_delta_um=args.min_delta_um,
        training=True,
        seed=args.seed,
        input_mode=args.input_mode,
        mask_radius_diam_scale=args.mask_radius_diam_scale,
        mask_min_radius_px=args.mask_min_radius_px,
        mask_softness_px=args.mask_softness_px,
        outer_swap_prob=args.outer_swap_prob,
        outer_swap_radius_diam_scale=args.outer_swap_radius_diam_scale,
        neighbor_mix_prob=args.neighbor_mix_prob,
        near_far_prob=args.near_far_prob,
        cache_images=args.cache_images,
    )
    valid_ds = DepthPairDataset(
        manifest,
        manifest_dir,
        split="valid",
        pairs_per_epoch=args.valid_pairs,
        min_delta_um=args.min_delta_um,
        training=False,
        seed=args.seed + 1,
        input_mode=args.input_mode,
        mask_radius_diam_scale=args.mask_radius_diam_scale,
        mask_min_radius_px=args.mask_min_radius_px,
        mask_softness_px=args.mask_softness_px,
        outer_swap_prob=0.0,
        outer_swap_radius_diam_scale=args.outer_swap_radius_diam_scale,
        neighbor_mix_prob=0.0,
        near_far_prob=0.0,
        cache_images=args.cache_images,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    in_ch = input_channels(args.input_mode)
    cond_dim = condition_dim(args.input_mode)
    model = FocusScoreNet(width=args.width, in_channels=in_ch, cond_dim=cond_dim, arch=args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_meta = vars(args).copy()
    run_meta["manifest"] = str(args.manifest.resolve())
    run_meta["out_dir"] = str(args.out_dir.resolve())
    run_meta["device_resolved"] = str(device)
    run_meta["rows"] = int(len(manifest))
    run_meta["train_groups"] = int(len(train_ds.groups))
    run_meta["valid_groups"] = int(len(valid_ds.groups))
    write_json(args.out_dir / "run_meta.json", run_meta)

    history_path = args.out_dir / "history.csv"
    best_acc = -1.0
    best_metrics: dict[str, Any] = {}
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_pair_loss",
                "train_score_reg_loss",
                "train_acc",
                "valid_loss",
                "valid_pair_loss",
                "valid_score_reg_loss",
                "valid_acc",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_stats = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                args.pair_temperature,
                args.score_reg_weight,
                args.focus_scale_um,
            )
            valid_stats = run_epoch(
                model,
                valid_loader,
                device,
                None,
                args.pair_temperature,
                args.score_reg_weight,
                args.focus_scale_um,
            )
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_stats.loss,
                "train_pair_loss": train_stats.pair_loss,
                "train_score_reg_loss": train_stats.score_reg_loss,
                "train_acc": train_stats.acc,
                "valid_loss": valid_stats.loss,
                "valid_pair_loss": valid_stats.pair_loss,
                "valid_score_reg_loss": valid_stats.score_reg_loss,
                "valid_acc": valid_stats.acc,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} "
                f"valid_loss={valid_stats.loss:.4f} valid_acc={valid_stats.acc:.4f}"
            )

            if valid_stats.acc > best_acc:
                best_acc = valid_stats.acc
                best_metrics = row
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model": {
                            "width": args.width,
                            "in_channels": in_ch,
                            "cond_dim": cond_dim,
                            "input_mode": args.input_mode,
                            "arch": args.arch,
                        },
                        "args": run_meta,
                        "metrics": best_metrics,
                    },
                    args.out_dir / "depth_compare_best.pt",
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "model": {
                "width": args.width,
                "in_channels": in_ch,
                "cond_dim": cond_dim,
                "input_mode": args.input_mode,
                "arch": args.arch,
            },
            "args": run_meta,
            "metrics": best_metrics,
        },
        args.out_dir / "depth_compare_last.pt",
    )

    argmax_metrics = evaluate_argmax(
        model,
        manifest,
        manifest_dir,
        "valid",
        device,
        args.batch_size,
        args.input_mode,
        args.mask_radius_diam_scale,
        args.mask_min_radius_px,
        args.mask_softness_px,
        args.cache_images,
    )
    metrics = {
        "best_epoch_metrics": best_metrics,
        "final_argmax_valid": argmax_metrics,
        "train_groups": len(train_ds.groups),
        "valid_groups": len(valid_ds.groups),
    }
    write_json(args.out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
