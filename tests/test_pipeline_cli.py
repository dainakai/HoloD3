from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import yaml

from holod3.cli import main
from holod3.config import PipelineConfig
from holod3.pipeline import HoloD3Pipeline
from src.pipeline.run_pipeline_fused import add_checkpoint_provenance, mark_intermediate_metrics


def make_gabor_acquisition(tmp_path: Path) -> Path:
    image_dir = tmp_path / "holograms"
    image_dir.mkdir()
    assert cv2.imwrite(str(image_dir / "frame.png"), np.full((8, 8), 100, dtype=np.uint8))
    mapping = {
        "schema_version": 1,
        "name": "pipeline-fixture",
        "description": "single frame",
        "mode": "single_gabor",
        "frames": {"primary_holograms": "holograms", "minip": None},
        "optics": {
            "wavelength_um": 0.63,
            "pixel_pitch_um": 8.0,
            "image_size_px": 8,
            "reconstruction_start_um": 200.0,
            "slice_spacing_um": 20.0,
            "slice_count": 3,
            "phase_retrieval_distance_um": None,
        },
        "reconstruction": {
            "phase_retrieval_iterations": 0,
            "fft_padding_side": 12,
            "minip_slice_step": 1,
        },
        "calibration": {"secondary_distortion_coefficients": None},
        "transforms": {"primary": [], "minip": []},
    }
    path = tmp_path / "acquisition.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def config_with_tiny_weights(tmp_path: Path) -> PipelineConfig:
    paths: dict[str, str] = {}
    for field in ("yolo", "depth_primary", "depth_fallback", "diameter"):
        path = tmp_path / f"{field}.pt"
        path.write_bytes(field.encode())
        paths[field] = str(path)
    return PipelineConfig.preset("portable-torch").with_models(**paths)


def test_build_command_wires_acquisition_models_and_fallbacks(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)  # type: ignore[attr-defined]
    acquisition = make_gabor_acquisition(tmp_path)
    config = config_with_tiny_weights(tmp_path).with_fallbacks(
        minip_bbox=False,
        depth_router=False,
        diameter_underprediction=False,
    )
    command = HoloD3Pipeline(config).build_command(
        acquisition=acquisition,
        run_dir=tmp_path / "run",
        limit=0,
    )
    joined = " ".join(command)
    assert f"--acquisition-config {acquisition}" in joined
    assert "--limit 0" in joined
    assert f"--yolo-weights {tmp_path / 'yolo.pt'}" in joined
    assert f"--depth-checkpoint {tmp_path / 'depth_primary.pt'}" in joined
    assert "--device cpu" in joined and "--yolo-device cpu" in joined
    assert "--yolo-imgsz 1024" in joined
    assert "--no-bbox-fallback" in command
    assert "--no-depth-router" in command
    assert "--no-diameter-underprediction-fallback" in command


def test_pipeline_result_contract_with_mocked_core(tmp_path: Path, monkeypatch: object) -> None:
    acquisition = make_gabor_acquisition(tmp_path)
    run_dir = tmp_path / "run"

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        run_dir.mkdir(parents=True)
        pd.DataFrame([{"frame": 0, "x_um": 1, "y_um": 2, "depth_um": 3, "final_diameter_um": 4}]).to_csv(
            run_dir / "particles_3d.csv", index=False
        )
        pd.DataFrame([{"frame": 0, "confidence": 0.9}]).to_csv(run_dir / "raw_detections.csv", index=False)
        (run_dir / "pipeline_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (run_dir / "fused_depth_slice_metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "hybrid_diameter_metrics.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("holod3.pipeline.subprocess.run", fake_run)  # type: ignore[attr-defined]
    result = HoloD3Pipeline(config_with_tiny_weights(tmp_path)).run(
        acquisition=acquisition,
        run_dir=run_dir,
        limit=1,
        check_artifacts=False,
        create_visualization=False,
    )
    assert result.particles_csv == run_dir / "particles_3d.csv"
    assert len(result.particles()) == 1
    assert result.summary() == {"ok": True}
    assert result.visualization_html is None


def test_custom_models_do_not_require_unselected_packaged_artifacts(
    tmp_path: Path, monkeypatch: object
) -> None:
    config = config_with_tiny_weights(tmp_path)

    def unexpected_verify(**_: object) -> list[object]:
        raise AssertionError("custom checkpoints must not trigger production-manifest verification")

    monkeypatch.setattr("holod3.pipeline.verify_production_artifacts", unexpected_verify)  # type: ignore[attr-defined]
    HoloD3Pipeline(config).verify_configured_artifacts()


def test_cli_validate_acquisition_reports_mode_and_warning(tmp_path: Path, capsys: object) -> None:
    acquisition = make_gabor_acquisition(tmp_path)
    assert main(["validate-acquisition", str(acquisition)]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["ok"] and payload["mode"] == "single_gabor" and payload["frames"] == 1
    assert "model_domain_warning" in payload


def test_cli_infer_dry_run_accepts_direct_weight_overrides(tmp_path: Path, capsys: object) -> None:
    acquisition = make_gabor_acquisition(tmp_path)
    weights = config_with_tiny_weights(tmp_path).models
    code = main(
        [
            "infer",
            "--acquisition",
            str(acquisition),
            "--run-dir",
            str(tmp_path / "run"),
            "--preset",
            "portable-torch",
            "--yolo-weights",
            weights.yolo,
            "--depth-primary-weights",
            weights.depth_primary,
            "--depth-fallback-weights",
            weights.depth_fallback,
            "--diameter-weights",
            weights.diameter,
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert code == 0
    assert "run_pipeline_fused.py" in output
    assert "--acquisition-config" in output
    assert str(tmp_path / "yolo.pt") in output


def test_shareable_provenance_hashes_models_and_clears_deleted_intermediate_paths(tmp_path: Path) -> None:
    csv_path = tmp_path / "particles_3d.csv"
    pd.DataFrame([{"frame": 0, "x_um": 1.0}]).to_csv(csv_path, index=False)
    add_checkpoint_provenance(
        csv_path,
        {"detector": {"id": "external:detector.pt", "bytes": 7, "sha256": "a" * 64}},
    )
    output = pd.read_csv(csv_path)
    assert output.loc[0, "detector_checkpoint_id"] == "external:detector.pt"
    assert output.loc[0, "detector_checkpoint_sha256"] == "a" * 64

    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({"input": "/private/run/input.csv", "depth_output": "/private/run/depth.csv"}),
        encoding="utf-8",
    )
    mark_intermediate_metrics(metrics, retained=False)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["input"] is None and payload["input_name"] == "input.csv"
    assert payload["depth_output"] is None and not payload["intermediate_csv_retained"]
