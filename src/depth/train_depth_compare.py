from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("data/downloaded/depth-primary/manifest.csv")
DEFAULT_OUT_DIR = Path("runs/reproduction/manual/depth-primary")

DIAMETER_BINS: tuple[tuple[str, float, float], ...] = (
    ("25-60", 25.0, 60.0),
    ("60-160", 60.0, 160.0),
    ("160-250", 160.0, 250.0),
    ("250-400", 250.0, 400.0),
    ("400-500", 400.0, 500.0),
)

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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_repo_path(path: str | Path, *, repo_root: Path = REPO_ROOT, must_exist: bool = True) -> Path:
    value = Path(path).expanduser()
    resolved = (value if value.is_absolute() else repo_root / value).resolve()
    if not _inside(resolved, repo_root):
        raise ValueError(f"Path is outside the self-contained repository: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def display_repo_path(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Manifest column {column} contains non-finite values")
    return values


def _frame_key_series(df: pd.DataFrame) -> pd.Series:
    columns = [column for column in ("dataset", "scene", "frame") if column in df.columns]
    if "frame" not in columns:
        raise ValueError("Manifest needs a frame column for leakage-safe splitting")
    keys = df[columns[0]].astype(str)
    for column in columns[1:]:
        keys = keys.str.cat(df[column].astype(str), sep="|")
    if "file" in df.columns:
        # file disambiguates merged datasets that omitted dataset/scene columns.
        keys = keys.str.cat(df["file"].astype(str), sep="|")
    return keys


def audit_split_leakage(df: pd.DataFrame) -> dict[str, Any]:
    if "split" not in df.columns:
        raise ValueError("Manifest needs a split column")
    work = df.copy()
    work["_frame_key"] = _frame_key_series(work)
    sample_splits = work.groupby("sample_id", sort=False)["split"].nunique()
    frame_splits = work.groupby("_frame_key", sort=False)["split"].nunique()
    sample_overlap = sample_splits[sample_splits > 1].index.astype(str).tolist()
    frame_overlap = frame_splits[frame_splits > 1].index.astype(str).tolist()
    split_rows = {str(key): int(value) for key, value in work["split"].value_counts().items()}
    split_samples = {
        str(key): int(value)
        for key, value in work[["split", "sample_id"]].drop_duplicates()["split"].value_counts().items()
    }
    split_frames = {
        str(key): int(value)
        for key, value in work[["split", "_frame_key"]].drop_duplicates()["split"].value_counts().items()
    }
    return {
        "rows": int(len(work)),
        "samples": int(work["sample_id"].nunique()),
        "frames": int(work["_frame_key"].nunique()),
        "split_rows": split_rows,
        "split_samples": split_samples,
        "split_frames": split_frames,
        "sample_overlap_count": len(sample_overlap),
        "sample_overlap_examples": sample_overlap[:10],
        "frame_overlap_count": len(frame_overlap),
        "frame_overlap_examples": frame_overlap[:10],
    }


def assert_no_split_leakage(audit: dict[str, Any], *, context: str) -> None:
    if audit["sample_overlap_count"] or audit["frame_overlap_count"]:
        raise ValueError(
            f"{context} split leakage: sample_overlap={audit['sample_overlap_count']}, "
            f"frame_overlap={audit['frame_overlap_count']}. Use --split-policy frame-hash or fix the manifest."
        )


def apply_split_policy(
    df: pd.DataFrame,
    *,
    policy: str,
    seed: int,
    valid_frame_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 < valid_frame_fraction < 1.0:
        raise ValueError("valid_frame_fraction must be between 0 and 1")
    original = audit_split_leakage(df)
    work = df.copy()
    work["source_split"] = work["split"].astype(str)
    if policy == "strict":
        assert_no_split_leakage(original, context="Manifest")
    elif policy == "frame-hash":
        frame_keys = sorted(_frame_key_series(work).unique().tolist())
        if len(frame_keys) < 2:
            raise ValueError("frame-hash splitting needs at least two distinct frames")
        ranked = sorted(
            frame_keys,
            key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
        )
        valid_count = max(1, min(len(ranked) - 1, int(round(len(ranked) * valid_frame_fraction))))
        valid_keys = set(ranked[:valid_count])
        keys = _frame_key_series(work)
        work["split"] = np.where(keys.isin(valid_keys), "valid", "train")
    elif policy == "all-valid":
        work["split"] = "valid"
    else:
        raise ValueError(f"Unknown split policy: {policy}")

    effective = audit_split_leakage(work)
    assert_no_split_leakage(effective, context="Effective")
    if policy != "all-valid" and not {"train", "valid"}.issubset(set(work["split"].unique())):
        raise ValueError("Effective manifest must contain both train and valid frames")
    return work, {"policy": policy, "original": original, "effective": effective}


def load_manifest(path: str | Path, *, repo_root: Path = REPO_ROOT, check_crop_files: bool = True) -> pd.DataFrame:
    manifest_path = resolve_repo_path(path, repo_root=repo_root)
    df = pd.read_csv(manifest_path)
    required = {
        "split",
        "sample_id",
        "frame",
        "crop_path",
        "focus_dist_um",
        "z_rel_um",
        "z_true_rel_um",
        "slice",
        "diameter_um",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if df.empty:
        raise ValueError("Manifest is empty")
    if df["sample_id"].isna().any() or (df["sample_id"].astype(str).str.strip() == "").any():
        raise ValueError("Manifest contains empty sample_id values")
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.strip()
    if (df["split"] == "").any():
        raise ValueError("Manifest contains empty split values")
    if df["frame"].isna().any() or (df["frame"].astype(str).str.strip() == "").any():
        raise ValueError("Manifest contains empty frame values")
    for column in ("focus_dist_um", "z_rel_um", "z_true_rel_um", "slice", "diameter_um"):
        df[column] = _finite_series(df, column)
    if (df["diameter_um"] <= 0).any():
        raise ValueError("diameter_um must be positive")
    duplicates = df.duplicated(["sample_id", "slice"], keep=False)
    if duplicates.any():
        example = df.loc[duplicates, ["sample_id", "slice"]].iloc[0].to_dict()
        raise ValueError(f"Duplicate sample/slice row: {example}")
    inconsistent_diameter = df.groupby("sample_id")["diameter_um"].nunique().gt(1)
    if inconsistent_diameter.any():
        raise ValueError(f"diameter_um changes within sample: {inconsistent_diameter[inconsistent_diameter].index[0]}")
    inconsistent_true_depth = df.groupby("sample_id")["z_true_rel_um"].nunique().gt(1)
    if inconsistent_true_depth.any():
        raise ValueError(
            f"z_true_rel_um changes within sample: {inconsistent_true_depth[inconsistent_true_depth].index[0]}"
        )
    frame_keys = _frame_key_series(df)
    sample_frame_counts = frame_keys.groupby(df["sample_id"]).nunique()
    if sample_frame_counts.gt(1).any():
        raise ValueError(f"sample_id spans multiple frames: {sample_frame_counts[sample_frame_counts.gt(1)].index[0]}")

    crop_cache: dict[str, str] = {}
    for raw in df["crop_path"].astype(str).unique():
        value = Path(raw).expanduser()
        resolved = (value if value.is_absolute() else manifest_path.parent / value).resolve()
        if not _inside(resolved, repo_root):
            raise ValueError(f"Crop path escapes the repository: {raw} -> {resolved}")
        if check_crop_files and not resolved.is_file():
            raise FileNotFoundError(f"Crop not found: {raw} -> {resolved}")
        crop_cache[raw] = str(resolved)
    df["_crop_abs_path"] = df["crop_path"].astype(str).map(crop_cache)
    df.attrs["manifest_path"] = str(manifest_path)
    df.attrs["manifest_sha256"] = sha256_file(manifest_path)
    return df


@dataclass(frozen=True)
class AugmentationConfig:
    pixel_augment: bool = True
    center_shift_prob: float = 0.8
    center_shift_max_px: int = 2
    outer_swap_prob: float = 0.4
    outer_swap_radius_diam_scale: float = 2.5
    neighbor_mix_prob: float = 0.25
    diameter_scalar_perturb_log_std: float = 0.08
    diameter_scalar_dropout: float = 0.05
    diameter_scalar_overestimate_prob: float = 0.0
    diameter_scalar_overestimate_max_factor: float = 1.0

    def validate(self) -> None:
        for name in (
            "center_shift_prob",
            "outer_swap_prob",
            "neighbor_mix_prob",
            "diameter_scalar_dropout",
            "diameter_scalar_overestimate_prob",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.center_shift_max_px < 0:
            raise ValueError("center_shift_max_px must be non-negative")
        if self.diameter_scalar_perturb_log_std < 0:
            raise ValueError("diameter_scalar_perturb_log_std must be non-negative")
        if self.diameter_scalar_overestimate_max_factor < 1.0:
            raise ValueError("diameter_scalar_overestimate_max_factor must be >= 1")


@dataclass(frozen=True)
class PairBatchStats:
    loss: float
    pair_loss: float
    score_reg_loss: float
    acc: float
    n: int


@dataclass(frozen=True)
class PreferenceStats:
    loss: float
    acc: float
    n: int
    ties: int


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


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
    value = (math.log(max(float(diameter_um), 1e-6)) - lo) / max(hi - lo, 1e-6)
    return float(np.clip(value * 2.0 - 1.0, -1.5, 1.5))


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
    center = (crop_size - 1) / 2.0
    distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    mask = 1.0 / (1.0 + np.exp((distance - radius) / max(softness_px, 1e-3)))
    return mask.astype(np.float32)


def make_input_tensor(
    image: np.ndarray,
    diameter_um: float,
    input_mode: str,
    mask_radius_diam_scale: float,
    mask_min_radius_px: float,
    mask_softness_px: float,
) -> np.ndarray:
    raw = ((image.astype(np.float32) - 0.5) / 0.25)[None, :, :]
    channels = [raw]
    if "mask" in input_mode:
        mask = soft_center_mask(
            image.shape[0],
            diameter_um,
            mask_radius_diam_scale,
            mask_min_radius_px,
            mask_softness_px,
        )
        channels.append(mask[None, :, :])
    if input_mode in FULL_DIAM_CHANNEL_MODES:
        value = diameter_value(diameter_um)
        channels.append(np.full((1, image.shape[0], image.shape[1]), value, dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(channels, axis=0), dtype=np.float32)


def make_condition_tensor(diameter_um: float, input_mode: str) -> np.ndarray:
    if input_mode in SCALAR_DIAM_MODES:
        return np.asarray([diameter_value(diameter_um)], dtype=np.float32)
    return np.zeros((0,), dtype=np.float32)


def shift_edge(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return image
    height, width = image.shape
    pad_x = abs(dx)
    pad_y = abs(dy)
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[y0 : y0 + height, x0 : x0 + width]


def augment_pixels(image: np.ndarray, rng: random.Random, config: AugmentationConfig) -> np.ndarray:
    if not config.pixel_augment:
        return np.ascontiguousarray(image, dtype=np.float32)
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
    if rng.random() < 0.35:
        image = np.rot90(image, rng.randint(0, 3))
    if config.center_shift_max_px > 0 and rng.random() < config.center_shift_prob:
        image = shift_edge(
            image,
            rng.randint(-config.center_shift_max_px, config.center_shift_max_px),
            rng.randint(-config.center_shift_max_px, config.center_shift_max_px),
        )

    scale = rng.uniform(0.88, 1.12)
    offset = rng.uniform(-0.045, 0.045)
    gamma = rng.uniform(0.88, 1.12)
    image = np.clip(image * scale + offset, 0.0, 1.0)
    image = np.clip(image, 1e-4, 1.0) ** gamma

    if rng.random() < 0.75:
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        noise = noise_rng.normal(0.0, rng.uniform(0.002, 0.014), size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
    if rng.random() < 0.12:
        height, width = image.shape
        side = rng.randint(4, 9)
        x0 = rng.randint(0, max(0, width - side))
        y0 = rng.randint(0, max(0, height - side))
        image = image.copy()
        image[y0 : y0 + side, x0 : x0 + side] = float(np.median(image))
    return np.ascontiguousarray(image, dtype=np.float32)


def blend_outer_with_distractor(
    image: np.ndarray,
    distractor: np.ndarray,
    diameter_um: float,
    rng: random.Random,
    radius_diam_scale: float,
    min_radius_px: float,
    softness_px: float,
) -> np.ndarray:
    mask = soft_center_mask(image.shape[0], diameter_um, radius_diam_scale, min_radius_px, softness_px)
    alpha = rng.uniform(0.15, 0.55)
    outer = alpha * image + (1.0 - alpha) * distractor
    return np.clip(image * mask + outer * (1.0 - mask), 0.0, 1.0).astype(np.float32)


def shifted_neighbor_residual(
    image: np.ndarray,
    neighbor: np.ndarray,
    diameter_um: float,
    rng: random.Random,
    radius_diam_scale: float,
    min_radius_px: float,
    softness_px: float,
) -> np.ndarray:
    crop_size = image.shape[0]
    diameter_px = float(diameter_um) / 10.0
    min_shift = max(8.0, 1.7 * diameter_px)
    max_shift = max(min_shift + 1.0, crop_size * 0.48)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    radius = rng.uniform(min_shift, max_shift)
    residual = neighbor.astype(np.float32) - float(np.median(neighbor))
    residual = shift_edge(residual, int(round(math.cos(angle) * radius)), int(round(math.sin(angle) * radius)))
    center_keep = soft_center_mask(image.shape[0], diameter_um, radius_diam_scale, min_radius_px, softness_px)
    return np.clip(image + rng.uniform(0.15, 0.65) * residual * (1.0 - center_keep), 0.0, 1.0).astype(np.float32)


class DepthPairDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        pairs_per_epoch: int,
        min_delta_um: float,
        training: bool,
        seed: int,
        input_mode: str,
        mask_radius_diam_scale: float,
        mask_min_radius_px: float,
        mask_softness_px: float,
        augmentation: AugmentationConfig,
        near_far_prob: float,
        diameter_sampling: str,
        cache_images: bool,
    ) -> None:
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.pairs_per_epoch = pairs_per_epoch
        self.min_delta_um = min_delta_um
        self.training = training
        self.rng = random.Random(seed)
        self.cache: dict[int, np.ndarray] = {}
        self.input_mode = input_mode
        self.mask_radius_diam_scale = mask_radius_diam_scale
        self.mask_min_radius_px = mask_min_radius_px
        self.mask_softness_px = mask_softness_px
        self.augmentation = augmentation
        self.near_far_prob = near_far_prob
        self.diameter_sampling = diameter_sampling
        self.groups = [
            group.index.to_list() for _, group in self.rows.groupby("sample_id", sort=False) if len(group) >= 2
        ]
        if not self.groups:
            raise RuntimeError(f"No usable groups for split={split}")
        self.groups_by_diameter_bin: list[list[list[int]]] = []
        for _, low, high in DIAMETER_BINS:
            matching = [
                group
                for group in self.groups
                if low <= float(self.rows.loc[group[0], "diameter_um"]) < high
            ]
            if matching:
                self.groups_by_diameter_bin.append(matching)
        if cache_images:
            for row_index, row in self.rows.iterrows():
                self.cache[int(row_index)] = read_gray(Path(str(row["_crop_abs_path"])))
        self.fixed_pairs = None if training else [self._choose_pair_random() for _ in range(pairs_per_epoch)]

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _raw_crop(self, row_index: int) -> np.ndarray:
        if row_index not in self.cache:
            self.cache[row_index] = read_gray(Path(str(self.rows.loc[row_index, "_crop_abs_path"])))
        return self.cache[row_index]

    def _input_crop(self, row_index: int) -> np.ndarray:
        row = self.rows.loc[row_index]
        image = self._raw_crop(row_index)
        diameter_um = float(row["diameter_um"])
        if self.training:
            if self.augmentation.outer_swap_prob > 0 and self.rng.random() < self.augmentation.outer_swap_prob:
                other = self.rng.randrange(len(self.rows))
                image = blend_outer_with_distractor(
                    image,
                    self._raw_crop(other),
                    diameter_um,
                    self.rng,
                    self.augmentation.outer_swap_radius_diam_scale,
                    self.mask_min_radius_px,
                    self.mask_softness_px,
                )
            if self.augmentation.neighbor_mix_prob > 0 and self.rng.random() < self.augmentation.neighbor_mix_prob:
                other = self.rng.randrange(len(self.rows))
                image = shifted_neighbor_residual(
                    image,
                    self._raw_crop(other),
                    diameter_um,
                    self.rng,
                    self.augmentation.outer_swap_radius_diam_scale,
                    self.mask_min_radius_px,
                    self.mask_softness_px,
                )
            image = augment_pixels(image, self.rng, self.augmentation)
        return make_input_tensor(
            image,
            diameter_um,
            self.input_mode,
            self.mask_radius_diam_scale,
            self.mask_min_radius_px,
            self.mask_softness_px,
        )

    def _choose_pair_random(self) -> tuple[int, int]:
        def choose_group() -> list[int]:
            if self.training and self.diameter_sampling == "bin-balanced":
                return self.rng.choice(self.rng.choice(self.groups_by_diameter_bin))
            return self.rng.choice(self.groups)

        if self.training and self.near_far_prob > 0 and self.rng.random() < self.near_far_prob:
            for _ in range(32):
                group = choose_group()
                ordered = sorted(group, key=lambda index: float(self.rows.loc[index, "focus_dist_um"]))
                near = self.rng.choice(ordered[: max(1, len(ordered) // 4)])
                far = self.rng.choice(ordered[-max(1, len(ordered) // 2) :])
                delta = abs(float(self.rows.loc[near, "focus_dist_um"]) - float(self.rows.loc[far, "focus_dist_um"]))
                if delta >= self.min_delta_um:
                    return (near, far) if self.rng.random() < 0.5 else (far, near)
        for _ in range(32):
            group = choose_group()
            left, right = self.rng.sample(group, 2)
            delta = abs(float(self.rows.loc[left, "focus_dist_um"]) - float(self.rows.loc[right, "focus_dist_um"]))
            if delta >= self.min_delta_um:
                return left, right
        return tuple(self.rng.sample(choose_group(), 2))  # type: ignore[return-value]

    def _condition_pair(self, diameter_um: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.training or self.input_mode not in SCALAR_DIAM_MODES:
            condition = make_condition_tensor(diameter_um, self.input_mode)
            return condition, condition.copy()
        if self.rng.random() < self.augmentation.diameter_scalar_dropout:
            condition = np.zeros((1,), dtype=np.float32)
        else:
            perturbed = diameter_um * math.exp(self.rng.gauss(0.0, self.augmentation.diameter_scalar_perturb_log_std))
            if self.rng.random() < self.augmentation.diameter_scalar_overestimate_prob:
                max_factor = self.augmentation.diameter_scalar_overestimate_max_factor
                perturbed *= math.exp(self.rng.uniform(0.0, math.log(max_factor)))
            condition = make_condition_tensor(perturbed, self.input_mode)
        # The same perturbation is used for both slices, so it cannot leak the pair label.
        return condition, condition.copy()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        left_index, right_index = self._choose_pair_random() if self.fixed_pairs is None else self.fixed_pairs[index]
        left_row = self.rows.loc[left_index]
        right_row = self.rows.loc[right_index]
        left_distance = float(left_row["focus_dist_um"])
        right_distance = float(right_row["focus_dist_um"])
        left_condition, right_condition = self._condition_pair(float(left_row["diameter_um"]))
        return {
            "left": torch.from_numpy(self._input_crop(left_index)),
            "right": torch.from_numpy(self._input_crop(right_index)),
            "left_cond": torch.from_numpy(left_condition),
            "right_cond": torch.from_numpy(right_condition),
            "target": torch.tensor(1.0 if left_distance < right_distance else 0.0, dtype=torch.float32),
            "left_dist_um": torch.tensor(left_distance, dtype=torch.float32),
            "right_dist_um": torch.tensor(right_distance, dtype=torch.float32),
        }


class DepthCropEvalDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        input_mode: str,
        mask_radius_diam_scale: float,
        mask_min_radius_px: float,
        mask_softness_px: float,
        cache_images: bool,
    ) -> None:
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.input_mode = input_mode
        self.mask_radius_diam_scale = mask_radius_diam_scale
        self.mask_min_radius_px = mask_min_radius_px
        self.mask_softness_px = mask_softness_px
        self.cache: dict[int, np.ndarray] = {}
        if cache_images:
            for row_index, row in self.rows.iterrows():
                self.cache[int(row_index)] = read_gray(Path(str(row["_crop_abs_path"])))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        row = self.rows.loc[index]
        image = self.cache.get(index)
        if image is None:
            image = read_gray(Path(str(row["_crop_abs_path"])))
        diameter_um = float(row["diameter_um"])
        model_input = make_input_tensor(
            image,
            diameter_um,
            self.input_mode,
            self.mask_radius_diam_scale,
            self.mask_min_radius_px,
            self.mask_softness_px,
        )
        return {
            "image": torch.from_numpy(model_input),
            "cond": torch.from_numpy(make_condition_tensor(diameter_um, self.input_mode)),
            "sample_id": str(row["sample_id"]),
            "slice": float(row["slice"]),
            "z_rel_um": float(row["z_rel_um"]),
            "z_true_rel_um": float(row["z_true_rel_um"]),
            "focus_dist_um": float(row["focus_dist_um"]),
            "diameter_um": diameter_um,
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
    """Checkpoint-compatible focus scorer used by the existing inference pipeline."""

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
        features = self.backbone(x).mean(dim=(2, 3))
        if self.cond_dim:
            if cond is None:
                raise RuntimeError("Condition tensor is required for this model")
            features = torch.cat([features, cond.to(features.dtype)], dim=1)
        return self.head(features).squeeze(1)


def closeness_target(distance_um: torch.Tensor, focus_scale_um: float) -> torch.Tensor:
    return torch.exp(-distance_um / focus_scale_um).clamp(0.0, 1.0)


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
    total_loss = total_pair_loss = total_reg_loss = 0.0
    total_correct = total_n = 0
    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        left_condition = batch["left_cond"].to(device, non_blocking=True)
        right_condition = batch["right_cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        left_distance = batch["left_dist_um"].to(device, non_blocking=True)
        right_distance = batch["right_dist_um"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            left_score = model(left, left_condition)
            right_score = model(right, right_condition)
            logits = (left_score - right_score) / pair_temperature
            pair_loss = F.binary_cross_entropy_with_logits(logits, target)
            reg_left = F.mse_loss(torch.sigmoid(left_score), closeness_target(left_distance, focus_scale_um))
            reg_right = F.mse_loss(torch.sigmoid(right_score), closeness_target(right_distance, focus_scale_um))
            reg_loss = 0.5 * (reg_left + reg_right)
            loss = pair_loss + score_reg_weight * reg_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        prediction = (logits.detach() > 0).float()
        total_correct += int((prediction == target).sum().item())
        count = int(target.numel())
        total_n += count
        total_loss += float(loss.detach().item()) * count
        total_pair_loss += float(pair_loss.detach().item()) * count
        total_reg_loss += float(reg_loss.detach().item()) * count
    return PairBatchStats(
        loss=total_loss / max(total_n, 1),
        pair_loss=total_pair_loss / max(total_n, 1),
        score_reg_loss=total_reg_loss / max(total_n, 1),
        acc=total_correct / max(total_n, 1),
        n=total_n,
    )


@dataclass(frozen=True)
class PreferencePair:
    sample_id: str
    image_a: Path
    image_b: Path
    slice_a: int | float
    slice_b: int | float
    diameter_um: float
    target: float
    preference: str


def _preference_image(candidate: dict[str, Any], *, repo_root: Path) -> Path:
    raw = candidate.get("image_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Preference candidate is missing image_path")
    return resolve_repo_path(raw, repo_root=repo_root)


def load_preference_pairs(
    path: str | Path, *, repo_root: Path = REPO_ROOT
) -> tuple[list[PreferencePair], dict[str, Any]]:
    jsonl_path = resolve_repo_path(path, repo_root=repo_root)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    record_count = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid preference JSONL at line {line_number}: {exc}") from exc
            if event.get("record_type") != "depth_annotation_revision":
                continue
            sample_id = event.get("sample_id")
            revision = event.get("revision")
            if not isinstance(sample_id, str) or not isinstance(revision, int):
                raise ValueError(f"Invalid preference revision at line {line_number}")
            key = (str(event.get("manifest_sha256", "")), sample_id)
            if key not in latest or revision > int(latest[key]["revision"]):
                latest[key] = event
            record_count += 1

    pairs: list[PreferencePair] = []
    ignored_without_pair = ignored_flagged = 0
    for event in latest.values():
        annotation = event.get("annotation")
        pairwise = annotation.get("pairwise") if isinstance(annotation, dict) else None
        if not isinstance(pairwise, dict):
            ignored_without_pair += 1
            continue
        flags = annotation.get("flags") or {}
        if any(bool(flags.get(name)) for name in ("unfocusable", "multi_particle", "bad_roi")):
            ignored_flagged += 1
            continue
        selected = event.get("selected_candidates") or {}
        candidate_a = selected.get("pair_a")
        candidate_b = selected.get("pair_b")
        if not isinstance(candidate_a, dict) or not isinstance(candidate_b, dict):
            raise ValueError(f"Preference sample {event['sample_id']} lacks selected pair image provenance")
        preference = str(pairwise.get("preference", "")).strip()
        targets = {"A": 1.0, "B": 0.0, "tie": 0.5}
        if preference not in targets:
            raise ValueError(f"Unknown preference={preference!r} for sample {event['sample_id']}")
        sample_metadata = event.get("sample_metadata") or {}
        try:
            diameter_um = float(sample_metadata["diameter_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Preference sample {event['sample_id']} lacks diameter_um") from exc
        if not math.isfinite(diameter_um) or diameter_um <= 0:
            raise ValueError(f"Invalid preference diameter for sample {event['sample_id']}")
        pairs.append(
            PreferencePair(
                sample_id=str(event["sample_id"]),
                image_a=_preference_image(candidate_a, repo_root=repo_root),
                image_b=_preference_image(candidate_b, repo_root=repo_root),
                slice_a=pairwise["slice_a"],
                slice_b=pairwise["slice_b"],
                diameter_um=diameter_um,
                target=targets[preference],
                preference=preference,
            )
        )
    return pairs, {
        "jsonl": display_repo_path(jsonl_path, repo_root=repo_root),
        "sha256": sha256_file(jsonl_path),
        "revision_records": record_count,
        "latest_records": len(latest),
        "accepted_pairs": len(pairs),
        "accepted_ties": sum(pair.preference == "tie" for pair in pairs),
        "ignored_without_pair": ignored_without_pair,
        "ignored_flagged": ignored_flagged,
    }


class PreferencePairDataset(Dataset):
    def __init__(
        self,
        pairs: list[PreferencePair],
        *,
        pairs_per_epoch: int,
        seed: int,
        input_mode: str,
        mask_radius_diam_scale: float,
        mask_min_radius_px: float,
        mask_softness_px: float,
        cache_images: bool,
    ) -> None:
        if not pairs:
            raise ValueError("PreferencePairDataset needs at least one pair")
        self.pairs = pairs
        self.input_mode = input_mode
        self.mask_radius_diam_scale = mask_radius_diam_scale
        self.mask_min_radius_px = mask_min_radius_px
        self.mask_softness_px = mask_softness_px
        count = pairs_per_epoch if pairs_per_epoch > 0 else len(pairs)
        rng = random.Random(seed)
        self.schedule = (
            [rng.randrange(len(pairs)) for _ in range(count)] if count != len(pairs) else list(range(len(pairs)))
        )
        self.cache: dict[Path, np.ndarray] = {}
        if cache_images:
            for pair in pairs:
                self.cache[pair.image_a] = read_gray(pair.image_a)
                self.cache[pair.image_b] = read_gray(pair.image_b)

    def __len__(self) -> int:
        return len(self.schedule)

    def _input(self, path: Path, diameter_um: float) -> torch.Tensor:
        image = self.cache.get(path)
        if image is None:
            image = read_gray(path)
        tensor = make_input_tensor(
            image,
            diameter_um,
            self.input_mode,
            self.mask_radius_diam_scale,
            self.mask_min_radius_px,
            self.mask_softness_px,
        )
        return torch.from_numpy(tensor)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[self.schedule[index]]
        condition = torch.from_numpy(make_condition_tensor(pair.diameter_um, self.input_mode))
        return {
            "left": self._input(pair.image_a, pair.diameter_um),
            "right": self._input(pair.image_b, pair.diameter_um),
            "left_cond": condition,
            "right_cond": condition.clone(),
            "target": torch.tensor(pair.target, dtype=torch.float32),
            "is_tie": torch.tensor(pair.preference == "tie", dtype=torch.bool),
        }


def run_preference_epoch(
    model: FocusScoreNet,
    loader: DataLoader | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    pair_temperature: float,
    loss_weight: float,
    tie_logit_margin: float,
) -> PreferenceStats:
    if loader is None:
        return PreferenceStats(loss=0.0, acc=0.0, n=0, ties=0)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = total_n = total_ties = 0
    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        left_condition = batch["left_cond"].to(device, non_blocking=True)
        right_condition = batch["right_cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        is_tie = batch["is_tie"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = (model(left, left_condition) - model(right, right_condition)) / pair_temperature
            raw_loss = F.binary_cross_entropy_with_logits(logits, target)
            loss = raw_loss * loss_weight
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        prediction = logits.detach() > 0
        non_tie_correct = prediction == (target > 0.5)
        tie_correct = logits.detach().abs() <= tie_logit_margin
        correct = torch.where(is_tie, tie_correct, non_tie_correct)
        count = int(target.numel())
        total_correct += int(correct.sum().item())
        total_ties += int(is_tie.sum().item())
        total_n += count
        total_loss += float(raw_loss.detach().item()) * count
    return PreferenceStats(
        loss=total_loss / max(total_n, 1),
        acc=total_correct / max(total_n, 1),
        n=total_n,
        ties=total_ties,
    )


def _error_metrics(errors_um: np.ndarray) -> dict[str, Any]:
    if errors_um.size == 0:
        return {
            "n": 0,
            "bias_um": None,
            "mae_um": None,
            "median_abs_error_um": None,
            "p90_abs_error_um": None,
            "p95_abs_error_um": None,
            "within_1000um": None,
            "catastrophic_gt_5000_rate": None,
        }
    absolute = np.abs(errors_um)
    return {
        "n": int(errors_um.size),
        "bias_um": float(errors_um.mean()),
        "mae_um": float(absolute.mean()),
        "median_abs_error_um": float(np.median(absolute)),
        "p90_abs_error_um": float(np.percentile(absolute, 90)),
        "p95_abs_error_um": float(np.percentile(absolute, 95)),
        "within_1000um": float((absolute <= 1000.0).mean()),
        "catastrophic_gt_5000_rate": float((absolute > 5000.0).mean()),
    }


def summarize_argmax_predictions(predictions: pd.DataFrame, score_rows: pd.DataFrame) -> dict[str, Any]:
    errors = predictions["depth_error_um"].to_numpy(dtype=np.float64)
    overall = _error_metrics(errors)
    by_diameter: dict[str, dict[str, Any]] = {}
    for index, (label, low, high) in enumerate(DIAMETER_BINS):
        if index == len(DIAMETER_BINS) - 1:
            mask = (predictions["diameter_um"] >= low) & (predictions["diameter_um"] <= high)
        else:
            mask = (predictions["diameter_um"] >= low) & (predictions["diameter_um"] < high)
        by_diameter[label] = _error_metrics(predictions.loc[mask, "depth_error_um"].to_numpy(dtype=np.float64))
    populated = [(label, metrics) for label, metrics in by_diameter.items() if metrics["n"]]
    worst_mae = max(populated, key=lambda item: float(item[1]["mae_um"])) if populated else (None, None)
    worst_outlier = (
        max(populated, key=lambda item: float(item[1]["catastrophic_gt_5000_rate"])) if populated else (None, None)
    )
    candidate_counts = score_rows.groupby("sample_id")["slice"].nunique().to_numpy(dtype=np.int64)
    exact_full_grid = score_rows.groupby("sample_id", sort=False)["slice"].apply(
        lambda values: np.array_equal(
            np.sort(values.to_numpy(dtype=np.float64)), np.arange(1.0, 1025.0, dtype=np.float64)
        )
    )
    out_of_range = int(((predictions["diameter_um"] < 25.0) | (predictions["diameter_um"] > 500.0)).sum())
    return {
        "overall": overall,
        "diameter_bins_um": by_diameter,
        "worst_bin_mae": {
            "bin": worst_mae[0],
            "mae_um": worst_mae[1]["mae_um"] if worst_mae[1] else None,
        },
        "worst_bin_catastrophic": {
            "bin": worst_outlier[0],
            "rate": worst_outlier[1]["catastrophic_gt_5000_rate"] if worst_outlier[1] else None,
        },
        "diameter_outside_25_500_count": out_of_range,
        "candidate_grid": {
            "rows": int(len(score_rows)),
            "samples": int(score_rows["sample_id"].nunique()),
            "slice_min": float(score_rows["slice"].min()),
            "slice_max": float(score_rows["slice"].max()),
            "candidates_per_sample_min": int(candidate_counts.min()),
            "candidates_per_sample_median": float(np.median(candidate_counts)),
            "candidates_per_sample_max": int(candidate_counts.max()),
            "is_full_1024_grid": bool(
                candidate_counts.size and np.all(candidate_counts == 1024) and exact_full_grid.all()
            ),
        },
    }


@torch.no_grad()
def evaluate_global_grid_argmax(
    model: FocusScoreNet,
    manifest: pd.DataFrame,
    split: str,
    device: torch.device,
    batch_size: int,
    input_mode: str,
    mask_radius_diam_scale: float,
    mask_min_radius_px: float,
    mask_softness_px: float,
    cache_images: bool,
    predictions_output: Path | None = None,
) -> dict[str, Any]:
    dataset = DepthCropEvalDataset(
        manifest,
        split,
        input_mode,
        mask_radius_diam_scale,
        mask_min_radius_px,
        mask_softness_px,
        cache_images,
    )
    if not len(dataset):
        raise ValueError(f"No rows for global argmax split={split}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        condition = batch["cond"].to(device, non_blocking=True)
        scores = model(image, condition).detach().cpu().numpy()
        for index, score in enumerate(scores):
            rows.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "slice": float(batch["slice"][index]),
                    "z_rel_um": float(batch["z_rel_um"][index]),
                    "z_true_rel_um": float(batch["z_true_rel_um"][index]),
                    "focus_dist_um": float(batch["focus_dist_um"][index]),
                    "diameter_um": float(batch["diameter_um"][index]),
                    "score": float(score),
                }
            )
    score_rows = pd.DataFrame(rows)
    predicted = score_rows.loc[score_rows.groupby("sample_id", sort=False)["score"].idxmax()].copy()
    predicted["depth_error_um"] = predicted["z_rel_um"] - predicted["z_true_rel_um"]
    oracle = score_rows.loc[score_rows.groupby("sample_id", sort=False)["focus_dist_um"].idxmin()].copy()
    metrics = summarize_argmax_predictions(predicted, score_rows)
    metrics["oracle"] = _error_metrics((oracle["z_rel_um"] - oracle["z_true_rel_um"]).to_numpy(dtype=np.float64))
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predicted.to_csv(predictions_output, index=False)
        metrics["predictions_csv"] = display_repo_path(predictions_output)
    return metrics


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
) -> dict[str, Any]:
    """Backward-compatible wrapper; manifest_dir is retained for old callers."""
    del manifest_dir
    return evaluate_global_grid_argmax(
        model,
        manifest,
        split,
        device,
        batch_size,
        input_mode,
        mask_radius_diam_scale,
        mask_min_radius_px,
        mask_softness_px,
        cache_images,
    )


def checkpoint_selection_objective(
    argmax_metrics: dict[str, Any],
    pair_accuracy: float,
    *,
    worst_bin_weight: float,
    catastrophic_penalty_um: float,
    pair_error_penalty_um: float,
) -> dict[str, float]:
    overall_mae = float(argmax_metrics["overall"]["mae_um"])
    worst_mae = float(argmax_metrics["worst_bin_mae"]["mae_um"])
    overall_catastrophic = float(argmax_metrics["overall"]["catastrophic_gt_5000_rate"])
    worst_catastrophic = float(argmax_metrics["worst_bin_catastrophic"]["rate"])
    components = {
        "overall_argmax_mae_um": overall_mae,
        "weighted_worst_bin_mae_um": worst_bin_weight * worst_mae,
        "catastrophic_penalty_um": catastrophic_penalty_um * (overall_catastrophic + worst_catastrophic),
        "pair_error_penalty_um": pair_error_penalty_um * (1.0 - pair_accuracy),
    }
    components["objective"] = float(sum(components.values()))
    return components


def _model_config(width: int, input_mode: str, arch: str) -> dict[str, Any]:
    return {
        "width": width,
        "in_channels": input_channels(input_mode),
        "cond_dim": condition_dim(input_mode),
        "input_mode": input_mode,
        "arch": arch,
    }


def load_checkpoint_model(path: str | Path, device: torch.device, *, repo_root: Path = REPO_ROOT):
    checkpoint_path = resolve_repo_path(path, repo_root=repo_root)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("model") or {})
    required = {"width", "in_channels", "cond_dim", "input_mode"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Checkpoint model metadata missing: {missing}")
    config.setdefault("arch", "default")
    model = FocusScoreNet(
        width=int(config["width"]),
        in_channels=int(config["in_channels"]),
        cond_dim=int(config["cond_dim"]),
        arch=str(config["arch"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint, config, checkpoint_path


def _checkpoint_payload(
    model: FocusScoreNet,
    model_config: dict[str, Any],
    run_meta: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "model": model_config,
        "args": run_meta,
        "metrics": metrics,
        "schema_version": 2,
    }


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status[:200]}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def _sample_diameter_counts(df: pd.DataFrame, split: str) -> dict[str, int]:
    samples = df[df["split"] == split].drop_duplicates("sample_id")
    output: dict[str, int] = {}
    for index, (label, low, high) in enumerate(DIAMETER_BINS):
        if index == len(DIAMETER_BINS) - 1:
            mask = (samples["diameter_um"] >= low) & (samples["diameter_um"] <= high)
        else:
            mask = (samples["diameter_um"] >= low) & (samples["diameter_um"] < high)
        output[label] = int(mask.sum())
    return output


def _data_loader(dataset: Dataset, args: argparse.Namespace, *, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )


def _pair_dataset(
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    split: str,
    *,
    training: bool,
    pairs: int,
    input_mode: str,
) -> DepthPairDataset:
    augmentation = AugmentationConfig(
        pixel_augment=args.pixel_augment,
        center_shift_prob=args.center_shift_prob,
        center_shift_max_px=args.center_shift_max_px,
        outer_swap_prob=args.outer_swap_prob,
        outer_swap_radius_diam_scale=args.outer_swap_radius_diam_scale,
        neighbor_mix_prob=args.neighbor_mix_prob,
        diameter_scalar_perturb_log_std=args.diameter_scalar_perturb_log_std,
        diameter_scalar_dropout=args.diameter_scalar_dropout,
        diameter_scalar_overestimate_prob=args.diameter_scalar_overestimate_prob,
        diameter_scalar_overestimate_max_factor=args.diameter_scalar_overestimate_max_factor,
    )
    augmentation.validate()
    return DepthPairDataset(
        manifest,
        split,
        pairs,
        args.min_delta_um,
        training,
        args.seed if training else args.seed + 1,
        input_mode,
        args.mask_radius_diam_scale,
        args.mask_min_radius_px,
        args.mask_softness_px,
        augmentation,
        args.near_far_prob if training else 0.0,
        args.diameter_sampling if training else "natural",
        args.cache_images,
    )


def _preference_loader(pairs: list[PreferencePair], args: argparse.Namespace, input_mode: str) -> DataLoader | None:
    if not pairs:
        return None
    dataset = PreferencePairDataset(
        pairs,
        pairs_per_epoch=args.preference_pairs_per_epoch,
        seed=args.seed + 17,
        input_mode=input_mode,
        mask_radius_diam_scale=args.mask_radius_diam_scale,
        mask_min_radius_px=args.mask_min_radius_px,
        mask_softness_px=args.mask_softness_px,
        cache_images=args.cache_images,
    )
    return _data_loader(dataset, args)


def _prepare_manifests(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT):
    manifest = load_manifest(args.manifest, repo_root=repo_root)
    manifest_path = Path(str(manifest.attrs["manifest_path"]))
    manifest_hash = str(manifest.attrs["manifest_sha256"])
    effective, split_audit = apply_split_policy(
        manifest,
        policy=args.split_policy,
        seed=args.seed,
        valid_frame_fraction=args.valid_frame_fraction,
    )
    if args.eval_manifest is None:
        evaluation = effective
        evaluation_path = manifest_path
        evaluation_hash = manifest_hash
        evaluation_audit = split_audit
    else:
        evaluation_raw = load_manifest(args.eval_manifest, repo_root=repo_root)
        evaluation_path = Path(str(evaluation_raw.attrs["manifest_path"]))
        evaluation_hash = str(evaluation_raw.attrs["manifest_sha256"])
        evaluation, evaluation_audit = apply_split_policy(
            evaluation_raw,
            policy=args.eval_split_policy,
            seed=args.seed + 101,
            valid_frame_fraction=args.valid_frame_fraction,
        )
    return {
        "train": effective,
        "train_path": manifest_path,
        "train_hash": manifest_hash,
        "split_audit": split_audit,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "evaluation_hash": evaluation_hash,
        "evaluation_audit": evaluation_audit,
    }


def _base_run_meta(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    device: torch.device,
    preference_meta: dict[str, Any] | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    augmentation = AugmentationConfig(
        pixel_augment=args.pixel_augment,
        center_shift_prob=args.center_shift_prob,
        center_shift_max_px=args.center_shift_max_px,
        outer_swap_prob=args.outer_swap_prob,
        outer_swap_radius_diam_scale=args.outer_swap_radius_diam_scale,
        neighbor_mix_prob=args.neighbor_mix_prob,
        diameter_scalar_perturb_log_std=args.diameter_scalar_perturb_log_std,
        diameter_scalar_dropout=args.diameter_scalar_dropout,
        diameter_scalar_overestimate_prob=args.diameter_scalar_overestimate_prob,
        diameter_scalar_overestimate_max_factor=args.diameter_scalar_overestimate_max_factor,
    )
    return {
        "schema_version": 2,
        "command": _jsonable_args(args),
        "manifest": display_repo_path(prepared["train_path"], repo_root=repo_root),
        "manifest_sha256": prepared["train_hash"],
        "evaluation_manifest": display_repo_path(prepared["evaluation_path"], repo_root=repo_root),
        "evaluation_manifest_sha256": prepared["evaluation_hash"],
        "split_audit": prepared["split_audit"],
        "evaluation_split_audit": prepared["evaluation_audit"],
        "diameter_bin_sample_counts": {
            split: _sample_diameter_counts(prepared["train"], split) for split in ("train", "valid")
        },
        "seed": args.seed,
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only,
        "augmentation": asdict(augmentation),
        "preference": preference_meta,
        "device_requested": args.device,
        "device_resolved": str(device),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git": _git_metadata(repo_root),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}. Use --device cpu.")
    return torch.device(requested)


def run_validate_only(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if args.checkpoint is None:
        raise ValueError("--validate-only requires --checkpoint")
    if args.deterministic and args.num_workers != 0:
        raise ValueError("Deterministic mode requires --num-workers 0")
    seed_everything(args.seed, args.deterministic, args.deterministic_warn_only)
    device = _resolve_device(args.device)
    prepared = _prepare_manifests(args, repo_root=repo_root)
    model, checkpoint, config, checkpoint_path = load_checkpoint_model(args.checkpoint, device, repo_root=repo_root)
    input_mode = str(config["input_mode"])
    preference_pairs: list[PreferencePair] = []
    preference_meta = None
    if args.preference_jsonl:
        preference_pairs, preference_meta = load_preference_pairs(args.preference_jsonl, repo_root=repo_root)
    run_meta = _base_run_meta(args, prepared, device, preference_meta, repo_root=repo_root)
    run_meta["checkpoint"] = display_repo_path(checkpoint_path, repo_root=repo_root)
    run_meta["checkpoint_sha256"] = sha256_file(checkpoint_path)
    run_meta["checkpoint_model"] = config

    eval_manifest = prepared["evaluation"]
    valid_dataset = _pair_dataset(
        eval_manifest,
        args,
        args.eval_split,
        training=False,
        pairs=args.valid_pairs,
        input_mode=input_mode,
    )
    pair_stats = run_epoch(
        model,
        _data_loader(valid_dataset, args),
        device,
        None,
        args.pair_temperature,
        args.score_reg_weight,
        args.focus_scale_um,
    )
    preference_stats = run_preference_epoch(
        model,
        _preference_loader(preference_pairs, args, input_mode),
        device,
        None,
        pair_temperature=args.pair_temperature,
        loss_weight=1.0,
        tie_logit_margin=args.preference_tie_logit_margin,
    )
    out_dir = resolve_repo_path(args.out_dir, repo_root=repo_root, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "argmax_predictions.csv" if args.save_eval_predictions else None
    argmax = evaluate_global_grid_argmax(
        model,
        eval_manifest,
        args.eval_split,
        device,
        args.batch_size,
        input_mode,
        args.mask_radius_diam_scale,
        args.mask_min_radius_px,
        args.mask_softness_px,
        args.cache_images,
        predictions_path,
    )
    metrics = {
        "mode": "validate_only",
        "checkpoint_metrics": checkpoint.get("metrics"),
        "pair_validation": asdict(pair_stats),
        "preference_validation": asdict(preference_stats),
        "global_grid_argmax": argmax,
    }
    write_json(out_dir / "run_meta.json", run_meta)
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def run_training(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if args.deterministic and args.num_workers != 0:
        raise ValueError("Deterministic mode requires --num-workers 0")
    seed_everything(args.seed, args.deterministic, args.deterministic_warn_only)
    device = _resolve_device(args.device)
    prepared = _prepare_manifests(args, repo_root=repo_root)
    manifest = prepared["train"]
    evaluation = prepared["evaluation"]
    preference_pairs: list[PreferencePair] = []
    preference_meta = None
    if args.preference_jsonl:
        preference_pairs, preference_meta = load_preference_pairs(args.preference_jsonl, repo_root=repo_root)

    train_dataset = _pair_dataset(
        manifest, args, "train", training=True, pairs=args.pairs_per_epoch, input_mode=args.input_mode
    )
    valid_dataset = _pair_dataset(
        manifest, args, "valid", training=False, pairs=args.valid_pairs, input_mode=args.input_mode
    )
    train_loader = _data_loader(train_dataset, args)
    valid_loader = _data_loader(valid_dataset, args)
    preference_loader = _preference_loader(preference_pairs, args, args.input_mode)
    config = _model_config(args.width, args.input_mode, args.arch)
    model = FocusScoreNet(
        width=config["width"],
        in_channels=config["in_channels"],
        cond_dim=config["cond_dim"],
        arch=config["arch"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    out_dir = resolve_repo_path(args.out_dir, repo_root=repo_root, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = _base_run_meta(args, prepared, device, preference_meta, repo_root=repo_root)
    run_meta["model"] = config
    run_meta["train_groups"] = len(train_dataset.groups)
    run_meta["valid_groups"] = len(valid_dataset.groups)
    run_meta["selection"] = {
        "worst_bin_weight": args.selection_worst_bin_weight,
        "catastrophic_penalty_um": args.selection_catastrophic_penalty_um,
        "pair_error_penalty_um": args.selection_pair_error_penalty_um,
    }
    write_json(out_dir / "run_meta.json", run_meta)

    history_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_pair_loss",
        "train_score_reg_loss",
        "train_acc",
        "preference_loss",
        "preference_acc",
        "valid_loss",
        "valid_pair_loss",
        "valid_score_reg_loss",
        "valid_acc",
        "argmax_mae_um",
        "argmax_worst_bin_mae_um",
        "argmax_catastrophic_gt5000_rate",
        "selection_objective",
    ]
    history_path = out_dir / "history.csv"
    best_objective = math.inf
    best_epoch_metrics: dict[str, Any] = {}
    last_epoch_metrics: dict[str, Any] = {}
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
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
            preference_stats = run_preference_epoch(
                model,
                preference_loader,
                device,
                optimizer,
                pair_temperature=args.pair_temperature,
                loss_weight=args.preference_loss_weight,
                tie_logit_margin=args.preference_tie_logit_margin,
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
            argmax = evaluate_global_grid_argmax(
                model,
                evaluation,
                args.eval_split,
                device,
                args.batch_size,
                args.input_mode,
                args.mask_radius_diam_scale,
                args.mask_min_radius_px,
                args.mask_softness_px,
                args.cache_images,
            )
            selection = checkpoint_selection_objective(
                argmax,
                valid_stats.acc,
                worst_bin_weight=args.selection_worst_bin_weight,
                catastrophic_penalty_um=args.selection_catastrophic_penalty_um,
                pair_error_penalty_um=args.selection_pair_error_penalty_um,
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_stats.loss,
                "train_pair_loss": train_stats.pair_loss,
                "train_score_reg_loss": train_stats.score_reg_loss,
                "train_acc": train_stats.acc,
                "preference_loss": preference_stats.loss,
                "preference_acc": preference_stats.acc,
                "valid_loss": valid_stats.loss,
                "valid_pair_loss": valid_stats.pair_loss,
                "valid_score_reg_loss": valid_stats.score_reg_loss,
                "valid_acc": valid_stats.acc,
                "argmax_mae_um": argmax["overall"]["mae_um"],
                "argmax_worst_bin_mae_um": argmax["worst_bin_mae"]["mae_um"],
                "argmax_catastrophic_gt5000_rate": argmax["overall"]["catastrophic_gt_5000_rate"],
                "selection_objective": selection["objective"],
            }
            writer.writerow(row)
            handle.flush()
            last_epoch_metrics = {
                **row,
                "train": asdict(train_stats),
                "preference": asdict(preference_stats),
                "valid": asdict(valid_stats),
                "global_grid_argmax": argmax,
                "selection": selection,
            }
            print(
                f"epoch={epoch:03d} train_acc={train_stats.acc:.4f} valid_acc={valid_stats.acc:.4f} "
                f"argmax_mae={argmax['overall']['mae_um']:.1f}um "
                f"worst_bin={argmax['worst_bin_mae']['mae_um']:.1f}um objective={selection['objective']:.1f}"
            )
            if selection["objective"] < best_objective:
                best_objective = selection["objective"]
                best_epoch_metrics = last_epoch_metrics
                torch.save(
                    _checkpoint_payload(model, config, run_meta, best_epoch_metrics),
                    out_dir / "depth_compare_best.pt",
                )

    torch.save(
        _checkpoint_payload(model, config, run_meta, last_epoch_metrics),
        out_dir / "depth_compare_last.pt",
    )

    # Reload the selected checkpoint before final reporting. This avoids the old last-model/best-model mismatch.
    best_model, best_checkpoint, best_config, _ = load_checkpoint_model(
        out_dir / "depth_compare_best.pt", device, repo_root=repo_root
    )
    best_pair_stats = run_epoch(
        best_model,
        valid_loader,
        device,
        None,
        args.pair_temperature,
        args.score_reg_weight,
        args.focus_scale_um,
    )
    best_preference_stats = run_preference_epoch(
        best_model,
        preference_loader,
        device,
        None,
        pair_temperature=args.pair_temperature,
        loss_weight=1.0,
        tie_logit_margin=args.preference_tie_logit_margin,
    )
    predictions_path = out_dir / "best_argmax_predictions.csv" if args.save_eval_predictions else None
    best_argmax = evaluate_global_grid_argmax(
        best_model,
        evaluation,
        args.eval_split,
        device,
        args.batch_size,
        str(best_config["input_mode"]),
        args.mask_radius_diam_scale,
        args.mask_min_radius_px,
        args.mask_softness_px,
        args.cache_images,
        predictions_path,
    )
    metrics = {
        "mode": "train",
        "selection_policy": run_meta["selection"],
        "best_epoch_metrics": best_checkpoint.get("metrics"),
        "best_checkpoint_reloaded": True,
        "best_checkpoint_pair_validation": asdict(best_pair_stats),
        "best_checkpoint_preference_validation": asdict(best_preference_stats),
        "best_checkpoint_global_grid_argmax": best_argmax,
        "last_epoch_metrics": last_epoch_metrics,
        "train_groups": len(train_dataset.groups),
        "valid_groups": len(valid_dataset.groups),
    }
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or validate the checkpoint-compatible DepthModel focus scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--eval-manifest", type=Path, default=None, help="Optional denser/full-grid evaluation manifest"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Required by --validate-only")
    parser.add_argument("--split-policy", choices=["strict", "frame-hash"], default="frame-hash")
    parser.add_argument("--eval-split-policy", choices=["strict", "frame-hash", "all-valid"], default="strict")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--valid-frame-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-warn-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-epoch", type=int, default=20000)
    parser.add_argument("--valid-pairs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--arch", choices=["default", "faststem"], default="default")
    parser.add_argument("--input-mode", choices=sorted(INPUT_CHANNELS), default="raw_diam_scalar")
    parser.add_argument("--mask-radius-diam-scale", type=float, default=2.0)
    parser.add_argument("--mask-min-radius-px", type=float, default=8.0)
    parser.add_argument("--mask-softness-px", type=float, default=3.0)
    parser.add_argument("--pixel-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--center-shift-prob", type=float, default=0.8)
    parser.add_argument("--center-shift-max-px", type=int, default=2)
    parser.add_argument("--outer-swap-prob", type=float, default=0.4)
    parser.add_argument("--outer-swap-radius-diam-scale", type=float, default=2.5)
    parser.add_argument("--neighbor-mix-prob", type=float, default=0.25)
    parser.add_argument("--diameter-scalar-perturb-log-std", type=float, default=0.08)
    parser.add_argument("--diameter-scalar-dropout", type=float, default=0.05)
    parser.add_argument("--diameter-scalar-overestimate-prob", type=float, default=0.0)
    parser.add_argument("--diameter-scalar-overestimate-max-factor", type=float, default=1.0)
    parser.add_argument("--diameter-sampling", choices=["natural", "bin-balanced"], default="natural")
    parser.add_argument("--near-far-prob", type=float, default=0.0)
    parser.add_argument("--min-delta-um", type=float, default=50.0)
    parser.add_argument("--pair-temperature", type=float, default=0.35)
    parser.add_argument("--score-reg-weight", type=float, default=0.15)
    parser.add_argument("--focus-scale-um", type=float, default=1200.0)
    parser.add_argument("--preference-jsonl", type=Path, default=None)
    parser.add_argument("--preference-loss-weight", type=float, default=0.25)
    parser.add_argument("--preference-pairs-per-epoch", type=int, default=0)
    parser.add_argument("--preference-tie-logit-margin", type=float, default=0.25)
    parser.add_argument("--selection-worst-bin-weight", type=float, default=0.5)
    parser.add_argument("--selection-catastrophic-penalty-um", type=float, default=5000.0)
    parser.add_argument("--selection-pair-error-penalty-um", type=float, default=500.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--save-eval-predictions", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return run_validate_only(args) if args.validate_only else run_training(args)


if __name__ == "__main__":
    main()
