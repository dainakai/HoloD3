"""Direct use of the production MinIP YOLO checkpoint."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from holod3.config import PipelineConfig, resolve_asset_path


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict[str, int | float]:
        value = asdict(self)
        value.update(
            center_x=self.center_x,
            center_y=self.center_y,
            width=self.width,
            height=self.height,
        )
        return value


def contrast_stretch_0_75_to_255(image: np.ndarray, input_max: int = 75) -> np.ndarray:
    """Apply the validated production MinIP detector contrast transform."""

    if input_max <= 0:
        raise ValueError("input_max must be positive")
    values = np.clip(image.astype(np.float32), 0, float(input_max))
    return (values * (255.0 / float(input_max))).astype(np.uint8)


def prepare_minip(image: np.ndarray, input_max: int = 75) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Expected a grayscale, BGR, or BGRA image; got shape {image.shape}")
    stretched = contrast_stretch_0_75_to_255(gray, input_max=input_max)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def _natural_key(path: Path) -> list[str | int]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


class ParticleDetector:
    """Lazy-loading direct Python API for the production YOLO detector."""

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        config: PipelineConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or PipelineConfig.preset("portable-torch")
        self.weights = resolve_asset_path(weights or self.config.models.yolo)
        self.device = self._normalize_device(device or self.config.runtime.yolo_device)
        self._model: Any | None = None

    @staticmethod
    def _normalize_device(device: str) -> str:
        if device == "auto":
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        return device.split(":", 1)[1] if device.startswith("cuda:") else device

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.weights.is_file():
                raise FileNotFoundError(f"YOLO weights do not exist: {self.weights}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    @staticmethod
    def read_image(path: str | Path) -> np.ndarray:
        resolved = Path(path).expanduser().resolve()
        image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read image: {resolved}")
        return image

    def predict(
        self,
        image: str | Path | np.ndarray,
        *,
        confidence: float | None = None,
        nms_iou: float | None = None,
        max_detections: int | None = None,
        image_size: int | None = None,
    ) -> list[Detection]:
        raw = self.read_image(image) if isinstance(image, (str, Path)) else image
        height, width = raw.shape[:2]
        prepared = prepare_minip(raw, input_max=self.config.detection.contrast_input_max)
        results = self.model.predict(
            source=[prepared],
            imgsz=image_size or self.config.detection.image_size,
            conf=self.config.detection.confidence if confidence is None else confidence,
            iou=self.config.detection.nms_iou if nms_iou is None else nms_iou,
            device=self.device,
            max_det=max_detections or self.config.detection.max_detections,
            verbose=False,
            end2end=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
        return [
            Detection(
                class_id=int(class_id),
                confidence=float(score),
                x1=float(coords[0]),
                y1=float(coords[1]),
                x2=float(coords[2]),
                y2=float(coords[3]),
                image_width=int(width),
                image_height=int(height),
            )
            for coords, score, class_id in zip(xyxy, scores, classes, strict=True)
        ]

    def predict_directory(
        self,
        image_dir: str | Path,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        directory = Path(image_dir).expanduser().resolve()
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        paths = sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions),
            key=_natural_key,
        )
        if limit is not None and limit > 0:
            paths = paths[:limit]
        rows: list[dict[str, Any]] = []
        for frame, path in enumerate(paths):
            for index, detection in enumerate(self.predict(path)):
                rows.append({"frame": frame, "file": path.name, "detection_id": index, **detection.to_dict()})
        return pd.DataFrame(rows)

    def annotate(
        self,
        image: str | Path | np.ndarray,
        detections: Iterable[Detection] | None = None,
    ) -> np.ndarray:
        raw = self.read_image(image) if isinstance(image, (str, Path)) else image
        output = prepare_minip(raw, input_max=self.config.detection.contrast_input_max)
        selected = list(detections) if detections is not None else self.predict(raw)
        for detection in selected:
            start = (int(round(detection.x1)), int(round(detection.y1)))
            end = (int(round(detection.x2)), int(round(detection.y2)))
            cv2.rectangle(output, start, end, (31, 211, 255), 2, cv2.LINE_AA)
            label = f"particle {detection.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (start[0], max(14, start[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (31, 211, 255),
                1,
                cv2.LINE_AA,
            )
        return output
