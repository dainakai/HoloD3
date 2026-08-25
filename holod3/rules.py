"""Pure fallback rules shared by tests, UI descriptions, and integrations."""

from __future__ import annotations

from dataclasses import dataclass


def should_use_depth_fallback(
    diameter_method: str,
    diameter_um: float,
    *,
    enabled: bool = True,
    max_diameter_um: float = 75.0,
) -> bool:
    """Return the validated row-level depth-model routing decision."""

    return bool(enabled and "fallback" in str(diameter_method).lower() and float(diameter_um) <= float(max_diameter_um))


@dataclass(frozen=True)
class DiameterDecision:
    final_diameter_um: float
    source: str
    fallback_applied: bool
    bbox_average_side_um: float


def choose_final_diameter(
    *,
    slice_prediction_um: float,
    minip_diameter_um: float,
    bbox_width_px: float,
    bbox_height_px: float,
    diameter_px: float | None = None,
    pixel_pitch_um: float = 10.0,
    enabled: bool = True,
    ratio_threshold: float = 0.35,
    min_bbox_side_um: float = 250.0,
) -> DiameterDecision:
    """Apply the validated large-particle underprediction safety rule."""

    pitch = float(pixel_pitch_um)
    if diameter_px is not None and float(diameter_px) > 0:
        pitch = float(minip_diameter_um) / float(diameter_px)
    bbox_average_side_um = 0.5 * (float(bbox_width_px) + float(bbox_height_px)) * pitch
    fallback = bool(
        enabled
        and bbox_average_side_um >= float(min_bbox_side_um)
        and float(slice_prediction_um) < float(ratio_threshold) * bbox_average_side_um
    )
    return DiameterDecision(
        final_diameter_um=float(minip_diameter_um if fallback else slice_prediction_um),
        source="contrast_area_fallback" if fallback else "slice_diammodel",
        fallback_applied=fallback,
        bbox_average_side_um=bbox_average_side_um,
    )
