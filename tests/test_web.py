from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holod3.config import PipelineConfig, repository_root
from holod3.pipeline import PipelineResult
from holod3.web import create_app


class FakeDetection:
    def to_dict(self) -> dict[str, float]:
        return {"confidence": 0.9, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}


class FakeDetector:
    def predict(self, image: np.ndarray) -> list[FakeDetection]:
        assert image.size
        return [FakeDetection()]

    def annotate(self, image: np.ndarray, detections: list[FakeDetection]) -> np.ndarray:
        assert detections
        return np.zeros((8, 8, 3), dtype=np.uint8)


class FakePipeline:
    configs: list[PipelineConfig] = []

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.configs.append(config)

    def run(self, *, run_dir: Path, **_: object) -> PipelineResult:
        run_dir = Path(run_dir)
        particles = run_dir / "particles_3d.csv"
        summary = run_dir / "pipeline_summary.json"
        visualization = run_dir / "particles_3d.html"
        pd.DataFrame(
            [{"frame": 0, "x_um": 1, "y_um": 2, "depth_um": 3, "final_diameter_um": 4}]
        ).to_csv(particles, index=False)
        summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
        visualization.write_text("<html>fixture</html>", encoding="utf-8")
        return PipelineResult(
            run_dir=run_dir,
            particles_csv=particles,
            summary_json=summary,
            raw_detections_csv=run_dir / "raw_detections.csv",
            fused_metrics_json=run_dir / "fused_depth_slice_metrics.json",
            hybrid_metrics_json=run_dir / "hybrid_diameter_metrics.json",
            visualization_html=visualization,
        )


def wait_for_job(client: object, job_id: str) -> dict[str, object]:
    for _ in range(200):
        response = client.get(f"/api/jobs/{job_id}")  # type: ignore[attr-defined]
        payload = response.get_json()
        if payload["status"] in {"complete", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("Web pipeline job did not finish")


def test_index_explains_local_security_and_model_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("holod3.web.HoloD3Pipeline", FakePipeline)
    app = create_app(config=PipelineConfig.preset("portable-torch"), detector=FakeDetector(), run_root=tmp_path)
    client = app.test_client()
    html = client.get("/").get_data(as_text=True)
    assert "trusted-local UI has no authentication" in html
    assert "Acquisition YAML" in html
    assert 'name="yolo_weights"' in html
    assert str(repository_root()) not in html
    assert 'value="data/demo/experimental/acquisition.yaml"' in html
    assert "Open animated 3D scatter" in (Path(app.static_folder) / "app.js").read_text(encoding="utf-8")


def test_detector_upload_returns_annotation() -> None:
    app = create_app(detector=FakeDetector(), pipeline=FakePipeline(PipelineConfig.preset("portable-torch")))
    client = app.test_client()
    import cv2

    ok, encoded = cv2.imencode(".png", np.zeros((8, 8), dtype=np.uint8))
    assert ok
    response = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(encoded.tobytes()), "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1 and payload["annotated_png_base64"]


def test_detector_errors_are_json() -> None:
    class BrokenDetector(FakeDetector):
        def predict(self, image: np.ndarray) -> list[FakeDetection]:
            raise RuntimeError("fixture detector failure")

    app = create_app(detector=BrokenDetector(), pipeline=FakePipeline(PipelineConfig.preset("portable-torch")))
    client = app.test_client()
    response = client.post("/api/detect", data={"image_path": "data/demo/experimental/minip/005302.png"})
    assert response.status_code == 500
    assert response.is_json and response.get_json()["error"] == "fixture detector failure"


def test_pipeline_rejects_invalid_limit() -> None:
    app = create_app(detector=FakeDetector(), pipeline=FakePipeline(PipelineConfig.preset("portable-torch")))
    client = app.test_client()
    response = client.post(
        "/api/pipeline",
        data={"acquisition": "data/demo/experimental/acquisition.yaml", "limit": "not-an-integer"},
    )
    assert response.status_code == 400
    assert "integer" in response.get_json()["error"]


def test_injected_pipeline_rejects_silently_ignored_overrides(tmp_path: Path) -> None:
    app = create_app(
        detector=FakeDetector(),
        pipeline=FakePipeline(PipelineConfig.preset("portable-torch")),
        run_root=tmp_path,
    )
    client = app.test_client()
    response = client.post(
        "/api/pipeline",
        data={
            "acquisition": "data/demo/experimental/acquisition.yaml",
            "depth_router": "false",
        },
    )
    assert response.status_code == 400
    assert "injected pipeline" in response.get_json()["error"]


def test_full_pipeline_accepts_model_paths_and_serves_only_job_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakePipeline.configs.clear()
    monkeypatch.setattr("holod3.web.HoloD3Pipeline", FakePipeline)
    app = create_app(config=PipelineConfig.preset("portable-torch"), detector=FakeDetector(), run_root=tmp_path)
    client = app.test_client()
    run_dir = tmp_path / "job"
    response = client.post(
        "/api/pipeline",
        data={
            "acquisition": "data/demo/experimental/acquisition.yaml",
            "run_dir": str(run_dir),
            "limit": "1",
            "bbox_fallback": "false",
            "depth_router": "true",
            "diameter_fallback": "false",
            "yolo_weights": "custom/detector.pt",
            "depth_primary_weights": "custom/depth-primary.pt",
            "depth_fallback_weights": "custom/depth-fallback.pt",
            "diameter_weights": "custom/diameter.pt",
        },
    )
    assert response.status_code == 202
    state = wait_for_job(client, response.get_json()["job_id"])
    assert state["status"] == "complete"
    job_config = FakePipeline.configs[-1]
    assert job_config.models.yolo == "custom/detector.pt"
    assert not job_config.fallbacks.minip_bbox
    assert job_config.fallbacks.depth_router
    assert not job_config.fallbacks.diameter_underprediction
    csv_response = client.get(state["result"]["downloads"]["particles_csv"])  # type: ignore[index]
    assert csv_response.status_code == 200
    assert client.get(f"/api/jobs/{state['job_id']}/files/not-allowed").status_code == 404
