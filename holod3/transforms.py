"""Image loading and opt-in transformation plugins for acquisitions."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from holod3.acquisition import TransformStep

ImageTransform = Callable[..., np.ndarray]


def identity(image: np.ndarray) -> np.ndarray:
    return image


def invert(image: np.ndarray) -> np.ndarray:
    return 1.0 - image


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def flip_vertical(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[::-1, :])


def rotate_quarter_turns(image: np.ndarray, *, turns: int = 1) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, k=int(turns) % 4))


def crop(image: np.ndarray, *, x: int, y: int, width: int, height: int) -> np.ndarray:
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + int(width), y0 + int(height)
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        raise ValueError(f"Crop {(x0, y0, x1, y1)} is outside image shape {image.shape}.")
    return np.ascontiguousarray(image[y0:y1, x0:x1])


BUILTINS: dict[str, ImageTransform] = {
    "identity": identity,
    "invert": invert,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "rotate_quarter_turns": rotate_quarter_turns,
    "crop": crop,
}


def _load_file_module(path: Path) -> ModuleType:
    module_name = f"holod3_user_transform_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load transform module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_transform(function: str, *, base_dir: Path) -> ImageTransform:
    """Resolve a built-in, ``module:function``, or ``file.py:function`` transform."""

    if function in BUILTINS:
        return BUILTINS[function]
    if ":" not in function:
        raise ValueError(
            f"Unknown transform {function!r}. Use a built-in name or module:function (including file.py:function)."
        )
    module_value, attribute = function.rsplit(":", 1)
    candidate = Path(module_value).expanduser()
    if candidate.suffix == ".py" or "/" in module_value or "\\" in module_value:
        module_path = candidate if candidate.is_absolute() else base_dir / candidate
        if not module_path.is_file():
            raise FileNotFoundError(f"Transform module does not exist: {module_path}")
        module = _load_file_module(module_path.resolve())
    else:
        module = importlib.import_module(module_value)
    value = getattr(module, attribute, None)
    if not callable(value):
        raise ValueError(f"Transform target is not callable: {function}")
    return value


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or colour image to finite float32 values in [0, 1]."""

    value = np.asarray(image)
    if value.ndim == 3:
        if value.shape[2] == 4:
            value = cv2.cvtColor(value, cv2.COLOR_BGRA2GRAY)
        elif value.shape[2] == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported channel count: {value.shape}")
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        scale = float(np.iinfo(value.dtype).max)
        value = value.astype(np.float32) / scale
    else:
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Image contains NaN or infinite values.")
    minimum = float(value.min())
    maximum = float(value.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"Floating-point transform output must remain in [0, 1], got [{minimum:.6g}, {maximum:.6g}]."
        )
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0), dtype=np.float32)


def read_grayscale(path: str | Path) -> np.ndarray:
    resolved = Path(path)
    image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {resolved}")
    return normalize_grayscale(image)


def apply_transforms(
    image: np.ndarray,
    steps: Sequence[TransformStep],
    *,
    base_dir: Path,
) -> np.ndarray:
    value = normalize_grayscale(image)
    for step in steps:
        transform = resolve_transform(step.function, base_dir=base_dir)
        try:
            value = normalize_grayscale(transform(value, **step.kwargs))
        except Exception as exc:
            raise RuntimeError(f"Image transform {step.function!r} failed: {exc}") from exc
    return value


def load_transformed_image(path: str | Path, steps: Sequence[TransformStep], *, base_dir: Path) -> np.ndarray:
    return apply_transforms(read_grayscale(path), steps, base_dir=base_dir)


def save_grayscale_png(path: str | Path, image: np.ndarray) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(normalize_grayscale(image) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(resolved), pixels):
        raise RuntimeError(f"Could not write image: {resolved}")
    return resolved
