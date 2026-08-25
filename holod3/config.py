"""Typed configuration for the public HoloD3 API."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    """Return the HoloD3 asset root.

    Editable installations discover the root from this package. A packaged
    application may point at an external asset checkout with ``HOLOD3_ROOT``.
    """

    override = os.environ.get("HOLOD3_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"HoloD3 root does not contain pyproject.toml: {root}")
    return root


def resolve_asset_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a repository-relative or explicit external asset path."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((root or repository_root()) / path).resolve()


@dataclass(frozen=True)
class ModelConfig:
    yolo: str
    depth_primary: str
    depth_fallback: str
    diameter: str


@dataclass(frozen=True)
class DetectionConfig:
    image_size: int = 1024
    confidence: float = 0.10
    nms_iou: float = 0.15
    max_detections: int = 600
    batch_size: int = 8
    image_load_workers: int = 4
    contrast_input_max: int = 75


@dataclass(frozen=True)
class ReconstructionConfig:
    crop_size: int = 64
    slice_start: int = 1
    slice_end: int = 0
    slice_step: int = 1
    slice_block: int = 8
    recenter_on_reconstructed_slice: bool = False


@dataclass(frozen=True)
class FallbackConfig:
    minip_bbox: bool = True
    depth_router: bool = True
    depth_router_max_diameter_um: float = 75.0
    diameter_underprediction: bool = True
    diameter_ratio_threshold: float = 0.35
    diameter_min_bbox_side_um: float = 250.0


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    yolo_device: str = "0"
    depth_model_backend: str = "tensorrt"
    diameter_model_backend: str = "tensorrt"
    depth_batch_size: int = 256
    diameter_batch_size: int = 512
    strict_backend: bool = True
    roi_image_prefetch_workers: int = 4
    roi_image_prefetch_frames: int = 32
    hologram_prefetch_workers: int = 4
    hologram_prefetch_frames: int = 16
    stream_csv_writes: bool = True
    keep_intermediate_csv: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Model, fallback, and runtime policy for the HoloD3 inference chain."""

    schema_version: int
    preset_name: str
    description: str
    models: ModelConfig
    detection: DetectionConfig
    reconstruction: ReconstructionConfig
    fallbacks: FallbackConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PipelineConfig:
        required = {
            "schema_version",
            "preset_name",
            "description",
            "models",
            "detection",
            "reconstruction",
            "fallbacks",
            "runtime",
        }
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        if missing or unknown:
            raise ValueError(f"Inference config keys mismatch: missing={missing}, unknown={unknown}")
        config = cls(
            schema_version=int(value["schema_version"]),
            preset_name=str(value["preset_name"]),
            description=str(value["description"]),
            models=ModelConfig(**dict(value["models"])),
            detection=DetectionConfig(**dict(value["detection"])),
            reconstruction=ReconstructionConfig(**dict(value["reconstruction"])),
            fallbacks=FallbackConfig(**dict(value["fallbacks"])),
            runtime=RuntimeConfig(**dict(value["runtime"])),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        resolved = resolve_asset_path(path)
        try:
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not read inference config {resolved}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Inference config must contain a mapping: {resolved}")
        return cls.from_mapping(payload)

    @classmethod
    def preset(cls, name: str = "production") -> PipelineConfig:
        aliases = {
            "strict": "production.yaml",
            "production": "production.yaml",
            "portable": "portable-torch.yaml",
            "torch": "portable-torch.yaml",
            "portable-torch": "portable-torch.yaml",
        }
        try:
            filename = aliases[name]
        except KeyError as exc:
            raise ValueError(f"Unknown preset {name!r}; use production or portable-torch") from exc
        return cls.load(repository_root() / "configs" / "inference" / filename)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported inference schema_version: {self.schema_version}")
        if not math.isfinite(self.detection.confidence) or not 0.0 <= self.detection.confidence <= 1.0:
            raise ValueError("detection.confidence must be between 0 and 1")
        if not math.isfinite(self.detection.nms_iou) or not 0.0 <= self.detection.nms_iou <= 1.0:
            raise ValueError("detection.nms_iou must be between 0 and 1")
        positive_integers = {
            "detection.image_size": self.detection.image_size,
            "detection.max_detections": self.detection.max_detections,
            "detection.batch_size": self.detection.batch_size,
            "detection.contrast_input_max": self.detection.contrast_input_max,
            "reconstruction.crop_size": self.reconstruction.crop_size,
            "reconstruction.slice_start": self.reconstruction.slice_start,
            "reconstruction.slice_step": self.reconstruction.slice_step,
            "reconstruction.slice_block": self.reconstruction.slice_block,
            "runtime.depth_batch_size": self.runtime.depth_batch_size,
            "runtime.diameter_batch_size": self.runtime.diameter_batch_size,
            "runtime.roi_image_prefetch_frames": self.runtime.roi_image_prefetch_frames,
            "runtime.hologram_prefetch_frames": self.runtime.hologram_prefetch_frames,
        }
        nonnegative_integers = {
            "detection.image_load_workers": self.detection.image_load_workers,
            "reconstruction.slice_end": self.reconstruction.slice_end,
            "runtime.roi_image_prefetch_workers": self.runtime.roi_image_prefetch_workers,
            "runtime.hologram_prefetch_workers": self.runtime.hologram_prefetch_workers,
        }
        for field, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be positive")
        for field, value in nonnegative_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.runtime.depth_model_backend not in {"torch", "tensorrt"}:
            raise ValueError("runtime.depth_model_backend must be torch or tensorrt")
        if self.runtime.diameter_model_backend not in {"torch", "tensorrt"}:
            raise ValueError("runtime.diameter_model_backend must be torch or tensorrt")
        if not self.runtime.device.strip() or not self.runtime.yolo_device.strip():
            raise ValueError("runtime device values must not be empty")
        if self.runtime.strict_backend and {
            self.runtime.depth_model_backend,
            self.runtime.diameter_model_backend,
        } != {"tensorrt"}:
            raise ValueError("strict_backend requires TensorRT for both learned reconstruction models")
        if self.runtime.strict_backend and (
            self.runtime.device == "auto" or self.runtime.yolo_device == "auto"
        ):
            raise ValueError("strict_backend requires explicit CUDA device values")
        if (
            not math.isfinite(self.fallbacks.depth_router_max_diameter_um)
            or self.fallbacks.depth_router_max_diameter_um <= 0
        ):
            raise ValueError("fallbacks.depth_router_max_diameter_um must be positive")
        if (
            not math.isfinite(self.fallbacks.diameter_ratio_threshold)
            or not 0.0 < self.fallbacks.diameter_ratio_threshold < 1.0
        ):
            raise ValueError("fallbacks.diameter_ratio_threshold must be between 0 and 1")
        if (
            not math.isfinite(self.fallbacks.diameter_min_bbox_side_um)
            or self.fallbacks.diameter_min_bbox_side_um <= 0
        ):
            raise ValueError("fallbacks.diameter_min_bbox_side_um must be positive")

    def model_paths(self, root: Path | None = None) -> dict[str, Path]:
        return {key: resolve_asset_path(value, root) for key, value in asdict(self.models).items()}

    def required_model_fields(self) -> set[str]:
        fields = {"yolo", "depth_primary", "diameter"}
        if self.fallbacks.depth_router:
            fields.add("depth_fallback")
        return fields

    def require_model_files(self, root: Path | None = None) -> dict[str, Path]:
        paths = self.model_paths(root)
        required = self.required_model_fields()
        missing = [f"{key}: {path}" for key, path in paths.items() if key in required and not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing model artifacts:\n- " + "\n- ".join(missing))
        return paths

    def with_runtime(self, **changes: Any) -> PipelineConfig:
        updated = replace(self, runtime=replace(self.runtime, **changes))
        updated.validate()
        return updated

    def with_models(self, **changes: Any) -> PipelineConfig:
        updated = replace(self, models=replace(self.models, **changes))
        updated.validate()
        return updated

    def with_fallbacks(self, **changes: Any) -> PipelineConfig:
        updated = replace(self, fallbacks=replace(self.fallbacks, **changes))
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
