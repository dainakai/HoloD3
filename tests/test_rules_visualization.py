from __future__ import annotations

from pathlib import Path

import pandas as pd

from holod3.rules import choose_final_diameter, should_use_depth_fallback
from holod3.visualization import write_particle_animation
from src.detection.depth_and_slice_fused import add_hybrid_columns


def test_depth_router_requires_both_fallback_measurement_and_small_diameter() -> None:
    assert should_use_depth_fallback("bbox_fallback_empty_roi", 75.0)
    assert not should_use_depth_fallback("contrast_area", 50.0)
    assert not should_use_depth_fallback("bbox_fallback", 75.1)
    assert not should_use_depth_fallback("bbox_fallback", 50.0, enabled=False)


def test_diameter_safety_rule_preserves_measured_pitch_when_available() -> None:
    decision = choose_final_diameter(
        slice_prediction_um=60.0,
        minip_diameter_um=200.0,
        diameter_px=20.0,
        bbox_width_px=30.0,
        bbox_height_px=30.0,
    )
    assert decision.bbox_average_side_um == 300.0
    assert decision.fallback_applied
    assert decision.final_diameter_um == 200.0
    assert decision.source == "contrast_area_fallback"


def test_physical_output_columns_use_acquisition_scales() -> None:
    frame = pd.DataFrame(
        [
            {
                "diameter_px": 10.0,
                "diameter_um": 80.0,
                "w": 20.0,
                "h": 20.0,
                "slice_diam_pred_um": 70.0,
                "seg_xc": 2.5,
                "seg_yc": 4.0,
                "slice": 3,
            }
        ]
    )
    output, count = add_hybrid_columns(
        frame,
        ratio_threshold=0.35,
        min_bbox_side_um=250.0,
        pixel_pitch_um=8.0,
        reconstruction_start_um=1200.0,
        slice_spacing_um=40.0,
    )
    assert count == 0
    assert output.loc[0, "x_um"] == 20.0
    assert output.loc[0, "y_um"] == 32.0
    assert output.loc[0, "z_um"] == 80.0
    assert output.loc[0, "depth_um"] == 1280.0


def test_animation_is_self_contained_and_labels_independent_frames(tmp_path: Path) -> None:
    csv_path = tmp_path / "particles_3d.csv"
    pd.DataFrame(
        [
            {"frame": 0, "x_um": 1.0, "y_um": 2.0, "depth_um": 3.0, "final_diameter_um": 40.0},
            {"frame": 1, "x_um": 2.0, "y_um": 3.0, "depth_um": 4.0, "final_diameter_um": 60.0},
        ]
    ).to_csv(csv_path, index=False)
    output = write_particle_animation(csv_path, tmp_path / "particles_3d.html")
    html = output.read_text(encoding="utf-8")
    assert "plotly.js" in html.lower()
    assert "Independent per-frame measurements" in html
    assert "Frame " in html and "Play" in html and "Pause" in html


def test_animation_rejects_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["x_um", "y_um", "depth_um", "final_diameter_um"]).to_csv(csv_path, index=False)
    try:
        write_particle_animation(csv_path, tmp_path / "output.html")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty particle CSV should be rejected")
