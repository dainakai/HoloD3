from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from holod3.config import repository_root
from holod3.pipeline import HoloD3Pipeline

ROOT = repository_root()
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU integration test requires CUDA")


@pytest.mark.integration
@pytest.mark.slow
@requires_cuda
def test_included_experimental_frame_runs_end_to_end(tmp_path: Path) -> None:
    result = HoloD3Pipeline("portable-torch").run(
        acquisition=ROOT / "data/demo/experimental/acquisition.yaml",
        run_dir=tmp_path / "dual-run",
        limit=1,
        create_visualization=True,
    )
    particles = result.particles()
    summary = result.summary()
    assert len(particles) == 380
    assert {"x_um", "y_um", "z_um", "depth_um", "final_diameter_um"}.issubset(particles.columns)
    assert {
        "detector_checkpoint_sha256",
        "depth_primary_checkpoint_sha256",
        "depth_fallback_checkpoint_sha256",
        "diameter_checkpoint_sha256",
    }.issubset(particles.columns)
    assert summary["holography_mode"] == "dual_phase_retrieval"
    assert len(summary["model_artifacts"]["detector"]["sha256"]) == 64
    assert str(ROOT) not in json.dumps(summary)
    fused_metrics = json.loads(result.fused_metrics_json.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert not fused_metrics["intermediate_csv_retained"]
    assert fused_metrics["depth_output"] is None and fused_metrics["slice_output"] is None
    assert result.visualization_html is not None and result.visualization_html.is_file()


@pytest.mark.integration
@pytest.mark.slow
@requires_cuda
def test_true_single_gabor_path_reconstructs_minip_without_secondary_image(tmp_path: Path) -> None:
    mapping = {
        "schema_version": 1,
        "name": "single-gabor-integration",
        "description": "One real hologram through the single-image branch.",
        "mode": "single_gabor",
        "frames": {
            "primary_holograms": str((ROOT / "data/demo/experimental/holograms/primary").resolve()),
            "minip": None,
        },
        "optics": {
            "wavelength_um": 0.6328,
            "pixel_pitch_um": 10.0,
            "image_size_px": 1024,
            "reconstruction_start_um": 80200.0,
            "slice_spacing_um": 100.0,
            "slice_count": 8,
            "phase_retrieval_distance_um": None,
        },
        "reconstruction": {
            "phase_retrieval_iterations": 0,
            "fft_padding_side": 1536,
            "minip_slice_step": 2,
        },
        "calibration": {"secondary_distortion_coefficients": None},
        "transforms": {"primary": [], "minip": []},
    }
    acquisition = tmp_path / "gabor-acquisition.yaml"
    acquisition.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    result = HoloD3Pipeline("portable-torch").run(
        acquisition=acquisition,
        run_dir=tmp_path / "gabor-run",
        limit=1,
        stop_after_preprocessing=True,
    )
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    detections = pd.read_csv(result.raw_detections_csv)
    assert summary["acquisition"] == "single-gabor-integration"
    assert result.fused_metrics_json is None and result.hybrid_metrics_json is None
    assert (result.run_dir / "_inputs/minip/005302.png").is_file()
    assert not detections.empty
