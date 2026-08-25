"""Validated, portable descriptions of hologram acquisitions.

An acquisition file describes images and optical geometry.  It intentionally
does not contain model paths, GPU choices, or training settings; those belong
to an inference preset.  Relative paths are resolved from the acquisition
file, which makes a complete acquisition directory movable as one unit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

HolographyMode = Literal["dual_phase_retrieval", "single_gabor"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _natural_key(value: str) -> list[str | int]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


@dataclass(frozen=True)
class TransformStep:
    """One image transformation applied to a normalized grayscale array."""

    function: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> TransformStep:
        if isinstance(value, str):
            return cls(function=value)
        if not isinstance(value, Mapping):
            raise ValueError("Each transform must be a function string or a mapping.")
        unknown = sorted(set(value) - {"function", "kwargs"})
        if unknown or "function" not in value:
            raise ValueError(f"Transform keys mismatch: missing function or unknown={unknown}")
        kwargs = value.get("kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise ValueError("transform.kwargs must be a mapping")
        return cls(function=str(value["function"]), kwargs=dict(kwargs))


@dataclass(frozen=True)
class FrameSources:
    """Directories containing synchronized images with matching file stems."""

    primary_holograms: str
    secondary_holograms: str | None = None
    minip: str | None = None


@dataclass(frozen=True)
class OpticsConfig:
    """Optical geometry in micrometres and pixels."""

    wavelength_um: float
    pixel_pitch_um: float
    image_size_px: int
    reconstruction_start_um: float
    slice_spacing_um: float
    slice_count: int
    phase_retrieval_distance_um: float | None = None


@dataclass(frozen=True)
class ReconstructionSettings:
    """Numerical settings that affect reconstructed images."""

    phase_retrieval_iterations: int = 3
    fft_padding_side: int = 1536
    minip_slice_step: int = 2


@dataclass(frozen=True)
class CalibrationConfig:
    """Optional spatial calibration applied to the secondary image."""

    secondary_distortion_coefficients: str | None = None


@dataclass(frozen=True)
class FrameRecord:
    """Resolved files for one synchronized frame."""

    stem: str
    primary: Path
    secondary: Path | None
    minip: Path | None


@dataclass(frozen=True)
class AcquisitionConfig:
    """A complete, model-independent acquisition contract."""

    schema_version: int
    name: str
    description: str
    mode: HolographyMode
    frames: FrameSources
    optics: OpticsConfig
    reconstruction: ReconstructionSettings = field(default_factory=ReconstructionSettings)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    transforms: dict[str, tuple[TransformStep, ...]] = field(default_factory=dict)
    source_path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> AcquisitionConfig:
        required = {"schema_version", "name", "description", "mode", "frames", "optics"}
        optional = {"reconstruction", "calibration", "transforms"}
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required - optional)
        if missing or unknown:
            raise ValueError(f"Acquisition config keys mismatch: missing={missing}, unknown={unknown}")

        transform_value = value.get("transforms", {})
        if not isinstance(transform_value, Mapping):
            raise ValueError("transforms must be a mapping")
        allowed_roles = {"minip", "primary", "secondary"}
        unknown_roles = sorted(set(transform_value) - allowed_roles)
        if unknown_roles:
            raise ValueError(f"Unknown transform roles: {unknown_roles}")
        transforms: dict[str, tuple[TransformStep, ...]] = {}
        for role in allowed_roles:
            steps = transform_value.get(role, [])
            if not isinstance(steps, list):
                raise ValueError(f"transforms.{role} must be a list")
            transforms[role] = tuple(TransformStep.from_value(item) for item in steps)

        config = cls(
            schema_version=int(value["schema_version"]),
            name=str(value["name"]),
            description=str(value["description"]),
            mode=str(value["mode"]),  # type: ignore[arg-type]
            frames=FrameSources(**dict(value["frames"])),
            optics=OpticsConfig(**dict(value["optics"])),
            reconstruction=ReconstructionSettings(**dict(value.get("reconstruction", {}))),
            calibration=CalibrationConfig(**dict(value.get("calibration", {}))),
            transforms=transforms,
            source_path=Path(source_path).expanduser().resolve() if source_path is not None else None,
        )
        config.validate_schema()
        return config

    @classmethod
    def load(cls, path: str | Path) -> AcquisitionConfig:
        resolved = Path(path).expanduser().resolve()
        try:
            value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not read acquisition config {resolved}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"Acquisition config must contain a mapping: {resolved}")
        return cls.from_mapping(value, source_path=resolved)

    @property
    def base_dir(self) -> Path:
        if self.source_path is None:
            raise ValueError("Path resolution requires an acquisition loaded from a file.")
        return self.source_path.parent

    def resolve_path(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def primary_dir(self) -> Path:
        path = self.resolve_path(self.frames.primary_holograms)
        assert path is not None
        return path

    @property
    def secondary_dir(self) -> Path | None:
        return self.resolve_path(self.frames.secondary_holograms)

    @property
    def minip_dir(self) -> Path | None:
        return self.resolve_path(self.frames.minip)

    @property
    def distortion_coefficients_path(self) -> Path | None:
        return self.resolve_path(self.calibration.secondary_distortion_coefficients)

    def validate_schema(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported acquisition schema_version: {self.schema_version}")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.mode not in {"dual_phase_retrieval", "single_gabor"}:
            raise ValueError("mode must be dual_phase_retrieval or single_gabor")
        if self.mode == "dual_phase_retrieval":
            if not self.frames.secondary_holograms:
                raise ValueError("dual_phase_retrieval requires frames.secondary_holograms")
            if self.optics.phase_retrieval_distance_um is None:
                raise ValueError("dual_phase_retrieval requires optics.phase_retrieval_distance_um")
        if self.mode == "single_gabor" and self.frames.secondary_holograms is not None:
            raise ValueError("single_gabor accepts one hologram directory; omit frames.secondary_holograms")
        positive = {
            "optics.wavelength_um": self.optics.wavelength_um,
            "optics.pixel_pitch_um": self.optics.pixel_pitch_um,
            "optics.image_size_px": self.optics.image_size_px,
            "optics.slice_spacing_um": self.optics.slice_spacing_um,
            "optics.slice_count": self.optics.slice_count,
            "reconstruction.fft_padding_side": self.reconstruction.fft_padding_side,
            "reconstruction.minip_slice_step": self.reconstruction.minip_slice_step,
        }
        integer_fields = {
            "optics.image_size_px": self.optics.image_size_px,
            "optics.slice_count": self.optics.slice_count,
            "reconstruction.phase_retrieval_iterations": self.reconstruction.phase_retrieval_iterations,
            "reconstruction.fft_padding_side": self.reconstruction.fft_padding_side,
            "reconstruction.minip_slice_step": self.reconstruction.minip_slice_step,
        }
        for label, number in integer_fields.items():
            if isinstance(number, bool) or not isinstance(number, int):
                raise ValueError(f"{label} must be an integer")
        for label, number in positive.items():
            if not math.isfinite(float(number)) or float(number) <= 0:
                raise ValueError(f"{label} must be positive")
        if not math.isfinite(float(self.optics.reconstruction_start_um)):
            raise ValueError("optics.reconstruction_start_um must be finite")
        phase_distance = self.optics.phase_retrieval_distance_um
        if phase_distance is not None and (
            not math.isfinite(float(phase_distance)) or float(phase_distance) <= 0
        ):
            raise ValueError("optics.phase_retrieval_distance_um must be positive when provided")
        if self.reconstruction.phase_retrieval_iterations < 0:
            raise ValueError("reconstruction.phase_retrieval_iterations must be non-negative")
        if self.reconstruction.fft_padding_side < self.optics.image_size_px:
            raise ValueError("reconstruction.fft_padding_side must be at least optics.image_size_px")
        if (self.reconstruction.fft_padding_side - self.optics.image_size_px) % 2:
            raise ValueError("fft padding minus image size must be even for a centred crop")

    @staticmethod
    def _image_map(directory: Path, label: str) -> dict[str, Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")
        paths = sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: _natural_key(path.name),
        )
        if not paths:
            raise FileNotFoundError(f"{label} directory contains no supported images: {directory}")
        by_stem: dict[str, Path] = {}
        for path in paths:
            if path.stem in by_stem:
                raise ValueError(f"{label} has duplicate image stem {path.stem!r}: {directory}")
            by_stem[path.stem] = path
        return by_stem

    def frame_records(self, *, require_minip: bool = False) -> list[FrameRecord]:
        """Resolve synchronized frames and reject partial or ambiguous inputs."""

        primary = self._image_map(self.primary_dir, "primary hologram")
        secondary_dir = self.secondary_dir
        secondary = self._image_map(secondary_dir, "secondary hologram") if secondary_dir is not None else None
        minip_dir = self.minip_dir
        minip = self._image_map(minip_dir, "MinIP") if minip_dir is not None else None
        if require_minip and minip is None:
            raise FileNotFoundError("This operation requires frames.minip, but the acquisition generates it from raw data.")

        expected = set(primary)
        comparisons = [("secondary hologram", secondary), ("MinIP", minip)]
        for label, mapping in comparisons:
            if mapping is None:
                continue
            missing = sorted(expected - set(mapping), key=_natural_key)
            extra = sorted(set(mapping) - expected, key=_natural_key)
            if missing or extra:
                sample_missing = missing[:5]
                sample_extra = extra[:5]
                raise ValueError(
                    f"{label} stems do not match primary holograms: "
                    f"missing={sample_missing} ({len(missing)} total), extra={sample_extra} ({len(extra)} total)"
                )

        self.load_distortion_coefficients()

        return [
            FrameRecord(
                stem=stem,
                primary=primary[stem],
                secondary=secondary[stem] if secondary is not None else None,
                minip=minip[stem] if minip is not None else None,
            )
            for stem in sorted(primary, key=_natural_key)
        ]

    def load_distortion_coefficients(self) -> np.ndarray | None:
        """Load and validate the optional 12-value quadratic calibration."""

        path = self.distortion_coefficients_path
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"Distortion coefficient file does not exist: {path}")
        raw = np.loadtxt(path, dtype=np.float64)
        values = raw[:, 0] if raw.ndim == 2 else raw
        if values.shape != (12,):
            raise ValueError(f"Distortion calibration must contain 12 coefficients: {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Distortion calibration contains NaN or infinite values: {path}")
        return values

    def validate_image_contracts(self, records: list[FrameRecord] | None = None) -> list[FrameRecord]:
        """Execute configured transforms and verify every input image shape."""

        from holod3.transforms import load_transformed_image

        selected = records if records is not None else self.frame_records()
        expected = (self.optics.image_size_px, self.optics.image_size_px)
        for record in selected:
            for role, path in (
                ("primary", record.primary),
                ("secondary", record.secondary),
                ("minip", record.minip),
            ):
                if path is None:
                    continue
                image = load_transformed_image(path, self.transform_steps(role), base_dir=self.base_dir)
                if image.shape != expected:
                    raise ValueError(
                        f"{role} frame {record.stem!r} has shape {image.shape}; "
                        f"expected {expected} after transforms."
                    )
        return selected

    def selected_records(
        self,
        *,
        limit: int = 0,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> list[FrameRecord]:
        if limit < 0:
            raise ValueError("limit must be zero or greater")
        if start_index is not None and start_index < 0:
            raise ValueError("start_index must be zero or greater")
        if end_index is not None and end_index < 0:
            raise ValueError("end_index must be zero or greater")
        if start_index is not None and end_index is not None and end_index < start_index:
            raise ValueError("end_index must be greater than or equal to start_index")
        records = self.frame_records()
        first = max(0, start_index or 0)
        last = len(records) if end_index is None else min(len(records), end_index + 1)
        selected = records[first:last]
        if limit > 0:
            selected = selected[:limit]
        if not selected:
            raise ValueError("No acquisition frames were selected.")
        return selected

    def transform_steps(self, role: str) -> tuple[TransformStep, ...]:
        return self.transforms.get(role, ())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_path", None)
        value["transforms"] = {
            role: [asdict(step) for step in steps]
            for role, steps in self.transforms.items()
            if steps
        }
        return value
