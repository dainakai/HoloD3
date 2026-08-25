#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_repo_root(start: Path) -> Path:
    """Find this workspace without relying on the source repository layout."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
try:
    from src.diam.model import DiameterNet, load_model, save_model_checkpoint
except ModuleNotFoundError:  # Direct execution: python src/diam/train_slice_diammodel.py
    from model import DiameterNet, load_model, save_model_checkpoint

DEFAULT_DATA_ROOT = Path("data/downloaded/diameter-combined")
DEFAULT_CROPS_CSV = DEFAULT_DATA_ROOT / "crops.csv"
DEFAULT_OUT_DIR = Path("runs/training/diameter")
DIAMETER_BINS_UM: tuple[tuple[float, float], ...] = (
    (25.0, 60.0),
    (60.0, 160.0),
    (160.0, 250.0),
    (250.0, 400.0),
    (400.0, 500.0),
)


@dataclass(frozen=True)
class CropRow:
    row_id: str
    split: str
    image_path: str
    file: str
    frame: int
    particle_id: int
    diameter_um: float
    diameter_px: float
    z_um: float
    z_slice: int
    training_domain: str
    source_frame_id: str
    resolved_image_path: Path


@dataclass(frozen=True)
class AugmentationConfig:
    center_shift_px: int = 4
    geometric: bool = True
    photometric: bool = True
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_90_probability: float = 0.5
    intensity_scale_min: float = 0.90
    intensity_scale_max: float = 1.12
    intensity_offset_abs_max: float = 0.04
    gamma_min: float = 0.88
    gamma_max: float = 1.14
    gaussian_blur_probability: float = 0.20
    gaussian_blur_sigma_min: float = 0.20
    gaussian_blur_sigma_max: float = 0.65
    gaussian_noise_std_min: float = 0.002
    gaussian_noise_std_max: float = 0.012
    z_quality_proxy_probability: float = 0.0
    z_quality_proxy_blur_sigma_min: float = 0.8
    z_quality_proxy_blur_sigma_max: float = 2.0
    z_quality_proxy_contrast_min: float = 0.55
    z_quality_proxy_contrast_max: float = 0.90


def repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def seed_everything(seed: int, *, deterministic: bool) -> dict[str, Any]:
    if deterministic:
        # Required by deterministic CUDA GEMM on CUDA 10.2 and newer. It must be
        # set before the first CUDA context is initialized.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    return {
        "enabled": deterministic,
        "seed": seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": deterministic,
        "warn_only": False,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "note": (
            "Repeated runs are configured deterministically. Bitwise equality across different GPU models, "
            "CUDA, cuDNN, or PyTorch versions is not guaranteed."
        ),
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_crop_tree(paths: set[Path], *, data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    entries: list[tuple[str, int, str]] = []
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        resolved = path.resolve()
        if not _is_within(resolved, data_root):
            raise ValueError(f"Resolved crop escaped --data-root: {resolved} is not inside {data_root}")
        relative = resolved.relative_to(data_root).as_posix()
        size = resolved.stat().st_size
        entries.append((relative, size, sha256_file(resolved)))
        total_bytes += size
    aggregate = hashlib.sha256()
    for relative, size, file_hash in entries:
        aggregate.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "root": portable_path(data_root),
        "files": len(entries),
        "bytes": total_bytes,
        "sha256": aggregate.hexdigest(),
        "algorithm": "sha256(relative_path NUL size NUL sha256(file) newline), sorted by relative_path",
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def portable_path(path: Path) -> str:
    path = path.resolve()
    if _is_within(path, REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    return str(path)


def resolve_data_root(data_root: Path) -> Path:
    resolved = repo_path(data_root)
    if not resolved.is_dir():
        raise NotADirectoryError(f"--data-root must be an existing dataset directory: {resolved}")
    return resolved


def resolve_crop_image_path(
    recorded_path: str,
    *,
    data_root: Path,
) -> Path:
    """Resolve a crop path locally while refusing implicit legacy-repo access.

    The copied manifest records paths such as
    ``legacy/source/dataset/slice_crops/images/...``.
    Only unambiguous candidates below the explicitly selected data root are
    considered. The old path itself is never opened and basename-only fallback
    is deliberately forbidden.
    """
    raw = Path(recorded_path).expanduser()
    data_root = data_root.resolve()
    if not recorded_path.strip():
        raise ValueError("Crop manifest contains an empty image_path")
    if not raw.is_absolute() and len(raw.parts) == 1:
        raise ValueError(
            f"Ambiguous basename-only image_path is not allowed: {recorded_path!r}. "
            "Record slice_crops/images/<file> or a path containing slice_crops/."
        )
    candidates: list[Path] = []

    def add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if _is_within(resolved, data_root) and resolved not in candidates:
            candidates.append(resolved)

    if raw.is_absolute():
        # Absolute paths are accepted only when already contained by the
        # explicitly selected data root.
        add(raw)
    else:
        add(repo_path(raw))
        add(data_root / raw)

    parts = raw.parts
    if data_root.name in parts:
        root_index = parts.index(data_root.name)
        if root_index + 1 < len(parts):
            add(data_root / Path(*parts[root_index + 1 :]))
    if "slice_crops" in parts:
        suffix = Path(*parts[parts.index("slice_crops") :])
        add(data_root / suffix)
    elif parts and parts[0] == "images":
        add(data_root / "slice_crops" / raw)

    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        choices = "\n  - ".join(str(path) for path in existing)
        raise RuntimeError(f"Ambiguous image_path {recorded_path!r}; multiple files exist:\n  - {choices}")
    attempted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve crop image {recorded_path!r} without using an external repository. "
        f"data_root={data_root}; attempted:\n  - {attempted}"
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_rows(path: Path, *, data_root: Path) -> list[CropRow]:
    path = repo_path(path)
    resolved_data_root = resolve_data_root(data_root)
    rows: list[CropRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            resolved_image_path = resolve_crop_image_path(
                raw["image_path"],
                data_root=resolved_data_root,
            )
            frame = int(raw["frame"])
            training_domain = raw.get("training_domain", "unspecified")
            source_frame_id = f"{training_domain}/frame={frame}/source={raw['file']}"
            rows.append(
                CropRow(
                    row_id=raw["row_id"],
                    split=raw["split"],
                    image_path=raw["image_path"],
                    file=raw["file"],
                    frame=frame,
                    particle_id=int(raw["particle_id"]),
                    diameter_um=float(raw["diameter_um"]),
                    diameter_px=float(raw["diameter_px"]),
                    z_um=float(raw["z_um"]),
                    z_slice=int(raw["z_slice"]),
                    training_domain=training_domain,
                    source_frame_id=source_frame_id,
                    resolved_image_path=resolved_image_path,
                )
            )
    if not rows:
        raise ValueError(f"Crop manifest has no rows: {path}")
    validate_diameter_labels(rows)
    return rows


def validate_diameter_labels(rows: list[CropRow]) -> None:
    invalid = [
        {"row_id": row.row_id, "diameter_um": row.diameter_um}
        for row in rows
        if not math.isfinite(row.diameter_um) or not 25.0 <= row.diameter_um <= 500.0
    ]
    if invalid:
        raise ValueError(
            "Diameter labels must all be finite and inside the required 25..500 um range; "
            f"invalid_count={len(invalid)}, examples={invalid[:10]}"
        )


def audit_frame_splits(rows: list[CropRow], *, raise_on_leakage: bool = True) -> dict[str, Any]:
    allowed_splits = {"train", "valid", "test"}
    frame_splits: dict[int, set[str]] = defaultdict(set)
    acquisition_frame_splits: dict[str, set[str]] = defaultdict(set)
    split_rows: Counter[str] = Counter()
    split_frames: dict[str, set[int]] = defaultdict(set)
    split_source_frames: dict[str, set[str]] = defaultdict(set)
    row_ids: Counter[str] = Counter()
    for row in rows:
        frame_splits[row.frame].add(row.split)
        acquisition_frame_splits[row.source_frame_id].add(row.split)
        split_rows[row.split] += 1
        split_frames[row.split].add(row.frame)
        split_source_frames[row.split].add(row.source_frame_id)
        row_ids[row.row_id] += 1

    unknown_splits = sorted(set(split_rows) - allowed_splits)
    frame_leakage = {str(frame): sorted(splits) for frame, splits in frame_splits.items() if len(splits) > 1}
    acquisition_frame_leakage = {
        identity: sorted(splits) for identity, splits in acquisition_frame_splits.items() if len(splits) > 1
    }
    duplicate_row_ids = sorted(row_id for row_id, count in row_ids.items() if count > 1)
    audit = {
        "passed": (
            not unknown_splits
            and not frame_leakage
            and not acquisition_frame_leakage
            and not duplicate_row_ids
        ),
        "row_count": len(rows),
        "unique_frames": len(frame_splits),
        "unique_acquisition_frames": len(acquisition_frame_splits),
        "training_domains": sorted({row.training_domain for row in rows}),
        "rows_by_split": {split: int(split_rows.get(split, 0)) for split in sorted(allowed_splits)},
        "frames_by_split": {split: len(split_frames.get(split, set())) for split in sorted(allowed_splits)},
        "source_frames_by_split": {
            split: len(split_source_frames.get(split, set())) for split in sorted(allowed_splits)
        },
        "unknown_splits": unknown_splits,
        "frame_leakage": frame_leakage,
        "acquisition_frame_leakage": acquisition_frame_leakage,
        "duplicate_row_ids": duplicate_row_ids,
        "split_unit": "training domain + frame + source file",
    }
    if raise_on_leakage and not audit["passed"]:
        raise ValueError(f"Frame-level split integrity check failed: {json.dumps(audit, ensure_ascii=False)}")
    return audit


def diameter_in_bin(value: float, lo: float, hi: float, *, include_upper: bool) -> bool:
    return lo <= value <= hi if include_upper else lo <= value < hi


def summarize_rows_by_diameter(rows: list[CropRow]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("train", "valid", "test", "all"):
        selected = rows if split == "all" else [row for row in rows if row.split == split]
        for index, (lo, hi) in enumerate(DIAMETER_BINS_UM):
            in_bin = [
                row
                for row in selected
                if diameter_in_bin(
                    row.diameter_um,
                    lo,
                    hi,
                    include_upper=index == len(DIAMETER_BINS_UM) - 1,
                )
            ]
            result.append(
                {
                    "split": split,
                    "diameter_bin_lo_um": lo,
                    "diameter_bin_hi_um": hi,
                    "count": len(in_bin),
                    "unique_frames": len({row.frame for row in in_bin}),
                }
            )
    return result


class SliceCropDataset(Dataset):
    def __init__(
        self,
        rows: list[CropRow],
        *,
        training: bool,
        log_diam_mean: float,
        log_diam_std: float,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        self.rows = rows
        self.training = training
        self.log_diam_mean = log_diam_mean
        self.log_diam_std = log_diam_std
        self.augmentation = augmentation or AugmentationConfig()
        self.cache = {row.resolved_image_path: read_gray(row.resolved_image_path) for row in rows}

    def __len__(self) -> int:
        return len(self.rows)

    def image_tensor(self, row: CropRow) -> torch.Tensor:
        crop = self.cache[row.resolved_image_path].astype(np.float32) / 255.0
        if self.training:
            aug = self.augmentation
            pad = aug.center_shift_px
            if pad > 0:
                padded = np.pad(crop, ((pad, pad), (pad, pad)), mode="edge")
                ox = random.randint(0, 2 * pad)
                oy = random.randint(0, 2 * pad)
                crop = padded[oy : oy + crop.shape[0], ox : ox + crop.shape[1]]
            if aug.geometric and random.random() < aug.horizontal_flip_probability:
                crop = np.flip(crop, axis=1)
            if aug.geometric and random.random() < aug.vertical_flip_probability:
                crop = np.flip(crop, axis=0)
            if aug.geometric and random.random() < aug.rotate_90_probability:
                crop = np.rot90(crop, k=random.randrange(4))
            if aug.photometric:
                scale = random.uniform(aug.intensity_scale_min, aug.intensity_scale_max)
                offset = random.uniform(-aug.intensity_offset_abs_max, aug.intensity_offset_abs_max)
                gamma = random.uniform(aug.gamma_min, aug.gamma_max)
                crop = np.clip(crop * scale + offset, 0.0, 1.0)
                crop = np.clip(crop, 1e-4, 1.0) ** gamma
                if random.random() < aug.gaussian_blur_probability:
                    crop = cv2.GaussianBlur(
                        crop,
                        (3, 3),
                        sigmaX=random.uniform(aug.gaussian_blur_sigma_min, aug.gaussian_blur_sigma_max),
                    )
                noise = np.random.normal(
                    0.0,
                    random.uniform(aug.gaussian_noise_std_min, aug.gaussian_noise_std_max),
                    crop.shape,
                ).astype(np.float32)
                crop = np.clip(crop + noise, 0.0, 1.0)
            if random.random() < aug.z_quality_proxy_probability:
                sigma = random.uniform(aug.z_quality_proxy_blur_sigma_min, aug.z_quality_proxy_blur_sigma_max)
                degraded = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma)
                contrast = random.uniform(aug.z_quality_proxy_contrast_min, aug.z_quality_proxy_contrast_max)
                mean = float(degraded.mean())
                crop = np.clip(mean + contrast * (degraded - mean), 0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(crop[None, :, :]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        log_diam = math.log(row.diameter_um)
        target = (log_diam - self.log_diam_mean) / self.log_diam_std
        return {
            "image": self.image_tensor(row),
            "target": torch.tensor(target, dtype=torch.float32),
            "diameter_um": torch.tensor(row.diameter_um, dtype=torch.float32),
        }


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    z = (target - mu) / sigma
    return (torch.log(sigma) + 0.5 * z.pow(2)).mean()


def seed_data_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    rows: list[CropRow],
    *,
    batch_size: int,
    training: bool,
    log_diam_mean: float,
    log_diam_std: float,
    num_workers: int,
    seed: int,
    augmentation: AugmentationConfig | None = None,
) -> DataLoader:
    ds = SliceCropDataset(
        rows,
        training=training,
        log_diam_mean=log_diam_mean,
        log_diam_std=log_diam_std,
        augmentation=augmentation,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(rows) >= batch_size,
        worker_init_fn=seed_data_worker,
        generator=generator,
    )


def train_epoch(
    model: DiameterNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    norm: dict[str, float],
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "nll": 0.0, "huber": 0.0, "mae_norm": 0.0, "items": 0.0}
    diameter_totals = {
        "absolute_error_um": 0.0,
        "large_absolute_error_um": 0.0,
        "very_large_absolute_error_um": 0.0,
        "large_items": 0.0,
        "very_large_items": 0.0,
    }
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        truth_um = batch["diameter_um"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            mu, sigma = model(x)
            nll = gaussian_nll(mu, sigma, target)
            huber = F.smooth_l1_loss(mu, target, beta=0.10)
            loss = nll + 0.04 * huber
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        n = x.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * n
        totals["nll"] += float(nll.detach().cpu()) * n
        totals["huber"] += float(huber.detach().cpu()) * n
        totals["mae_norm"] += float(torch.mean(torch.abs(mu.detach() - target)).cpu()) * n
        totals["items"] += n
        pred_log = mu.detach() * norm["log_diam_std"] + norm["log_diam_mean"]
        pred_um = torch.exp(pred_log)
        absolute_error_um = torch.abs(pred_um - truth_um)
        diameter_totals["absolute_error_um"] += float(absolute_error_um.sum().cpu())
        large_mask = (truth_um >= 250.0) & (truth_um <= 500.0)
        very_large_mask = (truth_um >= 400.0) & (truth_um <= 500.0)
        if bool(large_mask.any()):
            diameter_totals["large_absolute_error_um"] += float(absolute_error_um[large_mask].sum().cpu())
            diameter_totals["large_items"] += int(large_mask.sum().item())
        if bool(very_large_mask.any()):
            diameter_totals["very_large_absolute_error_um"] += float(absolute_error_um[very_large_mask].sum().cpu())
            diameter_totals["very_large_items"] += int(very_large_mask.sum().item())
    denom = max(1.0, totals["items"])
    metrics: dict[str, Any] = {k: v / denom for k, v in totals.items() if k != "items"}
    metrics.update(
        {
            "mae_um": diameter_totals["absolute_error_um"] / denom,
            "large_250_500_mae_um": (
                diameter_totals["large_absolute_error_um"] / diameter_totals["large_items"]
                if diameter_totals["large_items"] > 0
                else None
            ),
            "very_large_400_500_mae_um": (
                diameter_totals["very_large_absolute_error_um"] / diameter_totals["very_large_items"]
                if diameter_totals["very_large_items"] > 0
                else None
            ),
            "large_250_500_count": int(diameter_totals["large_items"]),
            "very_large_400_500_count": int(diameter_totals["very_large_items"]),
        }
    )
    return metrics


def validation_selection_score(
    metrics: dict[str, Any],
    *,
    overall_weight: float,
    large_weight: float,
    very_large_weight: float,
) -> tuple[float, dict[str, Any]]:
    weights = {
        "mae_um": float(overall_weight),
        "large_250_500_mae_um": float(large_weight),
        "very_large_400_500_mae_um": float(very_large_weight),
    }
    if any(weight < 0.0 for weight in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError("Best-checkpoint selection weights must be non-negative and sum to > 0")
    missing = [key for key in weights if metrics.get(key) is None]
    if missing:
        raise ValueError(f"Validation data is missing required large-particle groups for best selection: {missing}")
    total_weight = sum(weights.values())
    normalized_weights = {key: weight / total_weight for key, weight in weights.items()}
    components = {key: float(metrics[key]) for key in weights}
    score = sum(normalized_weights[key] * components[key] for key in components)
    return score, {
        "name": "weighted_validation_diameter_mae_um",
        "score": score,
        "components_um": components,
        "weights": normalized_weights,
    }


@torch.no_grad()
def predict_rows(
    model: DiameterNet,
    rows: list[CropRow],
    *,
    device: torch.device,
    batch_size: int,
    norm: dict[str, float],
    calibration_scale: float,
) -> list[dict[str, Any]]:
    ds = SliceCropDataset(
        rows,
        training=False,
        log_diam_mean=norm["log_diam_mean"],
        log_diam_std=norm["log_diam_std"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[dict[str, Any]] = []
    offset = 0
    model.eval()
    for batch in loader:
        x = batch["image"].to(device)
        mu_norm, sigma_norm = model(x)
        mu_log = mu_norm.cpu().numpy() * norm["log_diam_std"] + norm["log_diam_mean"]
        sigma_log = sigma_norm.cpu().numpy() * norm["log_diam_std"] * calibration_scale
        for ml, sl in zip(mu_log.tolist(), sigma_log.tolist(), strict=True):
            row = rows[offset]
            pred_um = math.exp(ml)
            out.append(
                {
                    "row_id": row.row_id,
                    "split": row.split,
                    "image_path": portable_path(row.resolved_image_path),
                    "file": row.file,
                    "frame": row.frame,
                    "particle_id": row.particle_id,
                    "training_domain": row.training_domain,
                    "source_frame_id": row.source_frame_id,
                    "z_um": row.z_um,
                    "z_slice": row.z_slice,
                    "true_diameter_um": row.diameter_um,
                    "true_diameter_px": row.diameter_px,
                    "pred_log_diameter": ml,
                    "pred_sigma_log": sl,
                    "pred_diameter_um": pred_um,
                    "pred_sigma_um": pred_um * sl,
                    "pred_low68_um": math.exp(ml - sl),
                    "pred_high68_um": math.exp(ml + sl),
                    "bbox_64crop_upper_bound_um": 640.0,
                }
            )
            offset += 1
    return out


def _float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def compute_metrics(preds: list[dict[str, Any]]) -> dict[str, Any]:
    if not preds:
        return {
            "count": 0,
            "bias_um": None,
            "mae_um": None,
            "median_ae_um": None,
            "p90_ae_um": None,
            "p95_ae_um": None,
            "rmse_um": None,
            "mape": None,
            "within_5pct": None,
            "within_10pct": None,
            "within_20pct": None,
            "relative_error_gt_20pct_rate": None,
            "underestimate_rate": None,
            "coverage_1sigma": None,
            "coverage_2sigma": None,
            "mean_sigma_um": None,
        }
    truth = _float_array(preds, "true_diameter_um")
    if not np.all(np.isfinite(truth)) or np.any(truth < 25.0) or np.any(truth > 500.0):
        raise ValueError("Metric truth diameters must all be finite and inside 25..500 um")
    pred = _float_array(preds, "pred_diameter_um")
    sigma = _float_array(preds, "pred_sigma_um")
    err = pred - truth
    abs_err = np.abs(pred - truth)
    relative_abs_err = abs_err / np.maximum(truth, 1e-9)
    return {
        "count": int(len(preds)),
        "bias_um": float(np.mean(err)),
        "mae_um": float(np.mean(abs_err)),
        "median_ae_um": float(np.median(abs_err)),
        "p90_ae_um": float(np.percentile(abs_err, 90)),
        "p95_ae_um": float(np.percentile(abs_err, 95)),
        "rmse_um": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mape": float(np.mean(relative_abs_err)),
        "within_5pct": float(np.mean(abs_err <= 0.05 * truth)),
        "within_10pct": float(np.mean(abs_err <= 0.10 * truth)),
        "within_20pct": float(np.mean(abs_err <= 0.20 * truth)),
        "relative_error_gt_20pct_rate": float(np.mean(relative_abs_err > 0.20)),
        "underestimate_rate": float(np.mean(err < 0.0)),
        "coverage_1sigma": float(np.mean(abs_err <= np.maximum(sigma, 1e-9))),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * np.maximum(sigma, 1e-9))),
        "mean_sigma_um": float(np.mean(sigma)),
    }


def fixed_bin_metrics(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, (lo, hi) in enumerate(DIAMETER_BINS_UM):
        sub = [
            pred
            for pred in preds
            if diameter_in_bin(
                float(pred["true_diameter_um"]),
                lo,
                hi,
                include_upper=index == len(DIAMETER_BINS_UM) - 1,
            )
        ]
        row = {
            "diameter_bin_lo_um": lo,
            "diameter_bin_hi_um": hi,
            "diameter_bin_mid_um": 0.5 * (lo + hi),
            "true_diameter_mean_um": float(np.mean(_float_array(sub, "true_diameter_um"))) if sub else None,
        }
        row.update(compute_metrics(sub))
        out.append(row)
    return out


def metrics_for_range(preds: list[dict[str, Any]], lo_um: float, hi_um: float) -> dict[str, Any]:
    selected = [pred for pred in preds if lo_um <= float(pred["true_diameter_um"]) <= hi_um]
    return {"range_lo_um": lo_um, "range_hi_um": hi_um, **compute_metrics(selected)}


def calibrate_sigma(preds: list[dict[str, Any]]) -> float:
    truth = _float_array(preds, "true_diameter_um")
    pred = _float_array(preds, "pred_diameter_um")
    sigma_log = _float_array(preds, "pred_sigma_log")
    err_log = np.abs(np.log(pred) - np.log(truth))
    scale = float(np.sqrt(np.mean((err_log / np.maximum(sigma_log, 1e-6)) ** 2)))
    return float(np.clip(scale, 0.20, 10.0))


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    plt.figure(figsize=(9.0, 5.0))
    plt.plot(epochs, [float(r["train_loss"]) for r in history], label="train loss")
    plt.plot(epochs, [float(r["valid_loss"]) for r in history], label="valid loss")
    plt.plot(epochs, [float(r["valid_mae_norm"]) for r in history], label="valid MAE norm")
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_pred_vs_true(preds: list[dict[str, Any]], path: Path) -> None:
    truth = _float_array(preds, "true_diameter_um")
    pred = _float_array(preds, "pred_diameter_um")
    sigma = _float_array(preds, "pred_sigma_um")
    lo = min(float(truth.min()), float(pred.min()))
    hi = max(float(truth.max()), float(pred.max()))
    plt.figure(figsize=(6.2, 6.0))
    sc = plt.scatter(truth, pred, c=sigma, s=8, cmap="viridis", alpha=0.65, linewidths=0)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    plt.xlabel("true diameter (um)")
    plt.ylabel("predicted diameter (um)")
    plt.colorbar(sc, label="predicted sigma (um)")
    plt.grid(True, alpha=0.20)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_error_bins(bins: list[dict[str, Any]], path: Path) -> None:
    populated = [row for row in bins if int(row["count"]) > 0]
    if not populated:
        return
    x = np.asarray([float(r["true_diameter_mean_um"]) for r in populated])
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(x, [float(r["mae_um"]) for r in populated], marker="o", label="MAE")
    plt.plot(x, [float(r["median_ae_um"]) for r in populated], marker="s", label="median AE")
    plt.xlabel("true diameter (um)")
    plt.ylabel("absolute error (um)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_montage(preds: list[dict[str, Any]], path: Path, *, mode: str) -> None:
    if not preds:
        return
    selected = sorted(
        preds,
        key=lambda r: abs(float(r["pred_diameter_um"]) - float(r["true_diameter_um"])),
        reverse=(mode == "worst"),
    )
    if mode != "worst":
        idx = np.linspace(0, len(selected) - 1, min(48, len(selected)), dtype=int)
        selected = [selected[int(i)] for i in idx]
    else:
        selected = selected[:48]
    tile = 128
    label_h = 44
    cols = 8
    rows_n = math.ceil(len(selected) / cols)
    canvas = Image.new("RGB", (cols * tile, rows_n * (tile + label_h)), "white")
    font = ImageFont.load_default()
    for i, pred in enumerate(selected):
        crop = read_gray(repo_path(Path(str(pred["image_path"]))))
        im = Image.fromarray(crop, mode="L")
        im = ImageOps.autocontrast(im, cutoff=0.5).resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle([1, 1, tile - 2, tile - 2], outline=(30, 120, 80), width=2)
        x = (i % cols) * tile
        y = (i // cols) * (tile + label_h)
        canvas.paste(im, (x, y))
        label_draw = ImageDraw.Draw(canvas)
        truth = float(pred["true_diameter_um"])
        pred_um = float(pred["pred_diameter_um"])
        label_draw.text((x + 3, y + tile + 1), f"p={pred_um:.0f} t={truth:.0f}", fill=(0, 0, 0), font=font)
        label_draw.text(
            (x + 3, y + tile + 15),
            f"e={pred_um - truth:+.0f} z={float(pred['z_um']) / 1000:.1f}mm",
            fill=(0, 0, 0),
            font=font,
        )
        label_draw.text((x + 3, y + tile + 29), Path(str(pred["file"])).name[:18], fill=(80, 80, 80), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def write_outputs(preds: list[dict[str, Any]], out_dir: Path, prefix: str) -> dict[str, Any]:
    metrics = compute_metrics(preds)
    bins = fixed_bin_metrics(preds)
    metrics["large_particle_250_500_um"] = metrics_for_range(preds, 250.0, 500.0)
    metrics["very_large_particle_400_500_um"] = metrics_for_range(preds, 400.0, 500.0)
    metrics["diameter_bins"] = bins
    write_csv(out_dir / f"{prefix}_predictions.csv", preds)
    write_json(out_dir / f"{prefix}_metrics.json", metrics)
    write_csv(out_dir / f"{prefix}_metrics_by_diameter.csv", bins)
    write_json(
        out_dir / f"{prefix}_large_particle_metrics.json",
        {
            "large_particle_250_500_um": metrics["large_particle_250_500_um"],
            "very_large_particle_400_500_um": metrics["very_large_particle_400_500_um"],
        },
    )
    plot_pred_vs_true(preds, out_dir / f"{prefix}_pred_vs_true.png")
    plot_error_bins(bins, out_dir / f"{prefix}_error_by_diameter.png")
    make_montage(preds, out_dir / f"{prefix}_worst_examples.png", mode="worst")
    make_montage(preds, out_dir / f"{prefix}_size_examples.png", mode="sizes")
    return metrics


def augmentation_from_args(args: argparse.Namespace) -> AugmentationConfig:
    if args.center_shift_px < 0:
        raise ValueError("--center-shift-px must be >= 0")
    if not 0.0 <= args.z_quality_proxy_prob <= 1.0:
        raise ValueError("--z-quality-proxy-prob must be between 0 and 1")
    return AugmentationConfig(
        center_shift_px=args.center_shift_px,
        geometric=args.geometric_augmentation,
        photometric=args.photometric_augmentation,
        z_quality_proxy_probability=args.z_quality_proxy_prob,
    )


def prepare_input_data(args: argparse.Namespace) -> tuple[Path, Path, list[CropRow], dict[str, Any]]:
    crops_csv = repo_path(args.crops_csv)
    data_root = resolve_data_root(args.data_root)
    rows = load_rows(crops_csv, data_root=data_root)
    split_audit = audit_frame_splits(rows)
    unique_images = {row.resolved_image_path for row in rows}
    outside = [path for path in unique_images if not _is_within(path.resolve(), data_root)]
    if outside:
        raise ValueError(f"All crop images must resolve inside --data-root; outside examples={outside[:10]}")
    if len(unique_images) != len(rows):
        raise ValueError(
            f"Crop manifest must map one unique image per row: rows={len(rows)}, unique_images={len(unique_images)}"
        )
    sample_image = read_gray(rows[0].resolved_image_path)
    crop_tree = fingerprint_crop_tree(unique_images, data_root=data_root)
    metadata = {
        "manifest": {
            "path": portable_path(crops_csv),
            "bytes": crops_csv.stat().st_size,
            "sha256": sha256_file(crops_csv),
            "rows": len(rows),
        },
        "data_root": portable_path(data_root),
        "resolved_unique_images": len(unique_images),
        "all_images_exist": all(path.is_file() for path in unique_images),
        "all_images_inside_data_root": True,
        "crop_tree_fingerprint": crop_tree,
        "sample_image_shape": list(sample_image.shape),
        "sample_image_dtype": str(sample_image.dtype),
        "source_identity": {
            "columns": ["training_domain", "frame", "file"],
            "training_domains": sorted({row.training_domain for row in rows}),
            "source_frames": len({row.source_frame_id for row in rows}),
        },
        "split_audit": split_audit,
        "diameter_distribution": summarize_rows_by_diameter(rows),
    }
    return crops_csv, data_root, rows, metadata


def runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def validate_data_only(args: argparse.Namespace) -> dict[str, Any]:
    _crops_csv, _data_root, rows, input_metadata = prepare_input_data(args)
    log_train = np.log(np.asarray([row.diameter_um for row in rows if row.split == "train"], dtype=np.float64))
    ds = SliceCropDataset(
        rows[:1],
        training=False,
        log_diam_mean=float(log_train.mean()),
        log_diam_std=float(max(log_train.std(), 1e-6)),
    )
    sample = ds[0]
    return {
        "status": "ok",
        "mode": "validate-only; no training or output files were produced",
        "input": input_metadata,
        "cpu_sample": {
            "tensor_shape": list(sample["image"].shape),
            "tensor_dtype": str(sample["image"].dtype),
            "diameter_um": float(sample["diameter_um"]),
        },
    }


def prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    out_dir = repo_path(path)
    if not _is_within(out_dir, REPO_ROOT) or out_dir == REPO_ROOT:
        raise ValueError(f"Training output must be a subdirectory of this workspace: {out_dir}")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists; choose a new path or pass --overwrite: {out_dir}")
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def run_train(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    determinism = seed_everything(args.seed, deterministic=args.deterministic)
    crops_csv, data_root, rows, input_metadata = prepare_input_data(args)
    augmentation = augmentation_from_args(args)
    out_dir = prepare_output_dir(args.out_dir, overwrite=args.overwrite)
    train_rows = [r for r in rows if r.split == "train"]
    valid_rows = [r for r in rows if r.split == "valid"]
    test_rows = [r for r in rows if r.split == "test"]
    if not train_rows or not valid_rows:
        raise RuntimeError(f"Need train and valid rows, got train={len(train_rows)} valid={len(valid_rows)}")
    log_train = np.log(np.asarray([r.diameter_um for r in train_rows], dtype=np.float64))
    norm = {
        "log_diam_mean": float(log_train.mean()),
        "log_diam_std": float(max(log_train.std(), 1e-6)),
        "target": "log(diameter_um)",
    }
    run_meta: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "arguments": serialized_args(args),
        "crops_csv": portable_path(crops_csv),
        "data_root": portable_path(data_root),
        "out_dir": portable_path(out_dir),
        "device": args.device,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "input": "64x64 focused phase-retrieval reconstruction slice crop at true particle z",
        "input_metadata": input_metadata,
        "norm": norm,
        "determinism": determinism,
        "environment": runtime_environment(),
        "augmentation": {
            **asdict(augmentation),
            "physical_z_jitter_supported": False,
            "physical_z_jitter_applied": False,
            "z_quality_limitation": (
                "Each particle has only one saved reconstruction crop at truth z. A physically correct off-focus "
                "slice or z-stack cannot be generated from this CSV. z_quality_proxy_probability only applies "
                "generic blur and contrast degradation and must not be interpreted as calibrated depth error."
            ),
        },
        "fixed_evaluation_bins_um": [list(pair) for pair in DIAMETER_BINS_UM],
        "large_particle_threshold_um": 250.0,
        "best_checkpoint_selection": {
            "name": "weighted_validation_diameter_mae_um",
            "components": ["overall MAE", "250-500 um MAE", "400-500 um MAE"],
            "configured_weights": {
                "overall": args.best_overall_weight,
                "large_250_500": args.best_large_weight,
                "very_large_400_500": args.best_very_large_weight,
            },
        },
    }
    write_json(out_dir / "run_meta.json", run_meta)
    train_loader = make_loader(
        train_rows,
        batch_size=args.batch_size,
        training=True,
        log_diam_mean=norm["log_diam_mean"],
        log_diam_std=norm["log_diam_std"],
        num_workers=args.num_workers,
        seed=args.seed,
        augmentation=augmentation,
    )
    valid_loader = make_loader(
        valid_rows,
        batch_size=args.batch_size,
        training=False,
        log_diam_mean=norm["log_diam_mean"],
        log_diam_std=norm["log_diam_std"],
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")
    model = DiameterNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    history: list[dict[str, Any]] = []
    best_selection_score = float("inf")
    best_selection: dict[str, Any] | None = None
    best_epoch_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, device, optimizer, norm)
        valid_m = train_epoch(model, valid_loader, device, None, norm)
        selection_score, selection = validation_selection_score(
            valid_m,
            overall_weight=args.best_overall_weight,
            large_weight=args.best_large_weight,
            very_large_weight=args.best_very_large_weight,
        )
        valid_m["selection_score"] = selection_score
        valid_m["selection_weight_overall"] = selection["weights"]["mae_um"]
        valid_m["selection_weight_large_250_500"] = selection["weights"]["large_250_500_mae_um"]
        valid_m["selection_weight_very_large_400_500"] = selection["weights"]["very_large_400_500_mae_um"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"valid_{k}": v for k, v in valid_m.items()},
        }
        history.append(row)
        write_csv(out_dir / "history.csv", history)
        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best_selection = {"epoch": epoch, **selection}
            best_epoch_metrics = dict(valid_m)
            best_epoch = epoch
            stale = 0
            save_model_checkpoint(
                out_dir / "slice_diammodel_best.pt", model, args, epoch, valid_m, norm, calibration_scale=1.0
            )
        else:
            stale += 1
        save_model_checkpoint(
            out_dir / "slice_diammodel_last.pt", model, args, epoch, valid_m, norm, calibration_scale=1.0
        )
        print(
            f"epoch {epoch:03d} train_loss={train_m['loss']:.4f} valid_loss={valid_m['loss']:.4f} "
            f"valid_mae_um={valid_m['mae_um']:.3f} "
            f"valid_large_mae_um={valid_m['large_250_500_mae_um']:.3f} "
            f"valid_very_large_mae_um={valid_m['very_large_400_500_mae_um']:.3f} "
            f"selection={selection_score:.3f}"
        )
        if stale >= args.patience:
            print(f"early stopping after epoch {epoch}; best_epoch={best_epoch}")
            break

    model, norm, _, _ = load_model(out_dir / "slice_diammodel_best.pt", device)
    raw_valid = predict_rows(
        model, valid_rows, device=device, batch_size=args.batch_size, norm=norm, calibration_scale=1.0
    )
    calibration_scale = calibrate_sigma(raw_valid)
    if best_selection is None or best_epoch_metrics is None:
        raise RuntimeError("Training completed without selecting a best checkpoint")
    save_model_checkpoint(
        out_dir / "slice_diammodel_best.pt",
        model,
        args,
        best_epoch,
        {**best_epoch_metrics, "checkpoint_selection": best_selection},
        norm,
        calibration_scale,
    )
    model, norm, calibration_scale, _ = load_model(out_dir / "slice_diammodel_best.pt", device)
    valid_preds = predict_rows(
        model, valid_rows, device=device, batch_size=args.batch_size, norm=norm, calibration_scale=calibration_scale
    )
    valid_metrics = write_outputs(valid_preds, out_dir, "valid")
    valid_metrics["calibration_scale"] = calibration_scale
    metrics = {"valid": valid_metrics}
    if test_rows:
        test_preds = predict_rows(
            model, test_rows, device=device, batch_size=args.batch_size, norm=norm, calibration_scale=calibration_scale
        )
        metrics["test"] = write_outputs(test_preds, out_dir, "test")
    write_json(out_dir / "metrics.json", metrics)
    plot_history(history, out_dir / "results.png")
    run_meta["status"] = "complete"
    run_meta["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    run_meta["training_result"] = {
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "best_epoch_validation_metrics": best_epoch_metrics,
        "epochs_completed": len(history),
        "calibration_scale": calibration_scale,
        "best_checkpoint": portable_path(out_dir / "slice_diammodel_best.pt"),
        "last_checkpoint": portable_path(out_dir / "slice_diammodel_last.pt"),
        "metrics": metrics,
    }
    write_json(out_dir / "run_meta.json", run_meta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train SliceDiamModel from focused reconstruction crops. Legacy image_path values in the copied CSV "
            "are remapped strictly below --data-root; external repositories and basename fallback are never used."
        )
    )
    parser.add_argument("--crops-csv", type=Path, default=DEFAULT_CROPS_CSV)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Explicit dataset root containing slice_crops/. Default: {DEFAULT_DATA_ROOT}.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing --out-dir before training. Without this flag an existing path is rejected.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.8e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--best-overall-weight",
        type=float,
        default=0.50,
        help="Best-checkpoint weight for validation MAE over all 25-500 um samples.",
    )
    parser.add_argument(
        "--best-large-weight",
        type=float,
        default=0.30,
        help="Best-checkpoint weight for validation MAE over 250-500 um samples.",
    )
    parser.add_argument(
        "--best-very-large-weight",
        type=float,
        default=0.20,
        help="Best-checkpoint weight for validation MAE over 400-500 um samples.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure deterministic Python/NumPy/PyTorch execution (default: enabled).",
    )
    parser.add_argument(
        "--center-shift-px",
        type=int,
        default=4,
        help="Maximum integer crop-center shift in each direction; 0 disables center-shift augmentation.",
    )
    parser.add_argument(
        "--geometric-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable flips and 90-degree rotations (default: enabled).",
    )
    parser.add_argument(
        "--photometric-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable intensity, gamma, blur, and noise augmentation (default: enabled).",
    )
    parser.add_argument(
        "--z-quality-proxy-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of generic blur/contrast degradation. This is not physical z jitter: the dataset has only "
            "one truth-z slice per particle. Default 0."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve every crop, audit frame splits, and load one CPU tensor without creating a run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only:
        print(json.dumps(validate_data_only(args), ensure_ascii=False, indent=2))
        return
    run_train(args)


if __name__ == "__main__":
    main()
