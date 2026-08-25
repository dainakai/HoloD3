"""Local Flask UI for direct detector trials and full pipeline jobs."""

from __future__ import annotations

import base64
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from holod3.config import PipelineConfig, repository_root, resolve_asset_path
from holod3.detector import ParticleDetector
from holod3.pipeline import HoloD3Pipeline

MAX_DECODED_IMAGE_PIXELS = 16_777_216


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(resolved)


@dataclass
class JobState:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    run_dir: str
    message: str = ""
    result: dict[str, Any] | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, run_dir: Path) -> JobState:
        now = datetime.now(UTC).isoformat()
        state = JobState(uuid.uuid4().hex, "queued", now, now, _display_path(run_dir))
        with self._lock:
            if any(item.status in {"queued", "running"} for item in self._jobs.values()):
                raise RuntimeError("A pipeline job is already queued or running.")
            self._jobs[state.job_id] = state
        return state

    def update(self, job_id: str, **changes: Any) -> JobState:
        with self._lock:
            state = self._jobs[job_id]
            for key, value in changes.items():
                setattr(state, key, value)
            state.updated_at = datetime.now(UTC).isoformat()
            return state

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def create_app(
    *,
    config: PipelineConfig | None = None,
    detector: ParticleDetector | None = None,
    pipeline: HoloD3Pipeline | None = None,
    run_root: str | Path = "runs/web",
) -> Flask:
    selected = config or PipelineConfig.preset("portable-torch")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    app.extensions["holod3_detector"] = detector or ParticleDetector(config=selected)
    app.extensions["holod3_pipeline"] = pipeline or HoloD3Pipeline(selected)
    app.extensions["holod3_pipeline_injected"] = pipeline is not None
    app.extensions["holod3_jobs"] = JobStore()
    app.extensions["holod3_run_root"] = resolve_asset_path(run_root)

    @app.errorhandler(Exception)
    def json_error(error: Exception) -> Any:
        """Keep browser/API failures machine-readable instead of returning HTML."""

        if isinstance(error, HTTPException):
            return jsonify({"error": error.description}), error.code
        app.logger.exception("HoloD3 Web request failed", exc_info=error)
        return jsonify({"error": str(error) or error.__class__.__name__}), 500

    @app.get("/")
    def index() -> str:
        sample_data = Path("data") / "demo" / "experimental"
        sample_acquisition = sample_data / "acquisition.yaml"
        sample_image = sample_data / "minip" / "005302.png"
        return render_template(
            "index.html",
            preset=selected.preset_name,
            sample_acquisition=str(sample_acquisition),
            sample_image=str(sample_image),
            fallback=asdict(selected.fallbacks),
            models=asdict(selected.models),
        )

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"ok": True, "preset": selected.preset_name})

    @app.post("/api/detect")
    def detect_route() -> Any:
        image: np.ndarray | None = None
        name = "uploaded image"
        if "image" in request.files:
            upload = request.files["image"]
            data = np.frombuffer(upload.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            name = upload.filename or name
        else:
            payload = request.get_json(silent=True) or request.form
            image_path = str(payload.get("image_path", "")).strip()
            if image_path:
                name = image_path
                image = ParticleDetector.read_image(resolve_asset_path(image_path))
        if image is None:
            return jsonify({"error": "Provide an image upload or image_path."}), 400
        if int(image.shape[0]) * int(image.shape[1]) > MAX_DECODED_IMAGE_PIXELS:
            return jsonify({"error": "Decoded image exceeds the 16,777,216-pixel safety limit."}), 413
        detector_instance: ParticleDetector = app.extensions["holod3_detector"]
        detections = detector_instance.predict(image)
        annotated = detector_instance.annotate(image, detections)
        encoded_ok, encoded = cv2.imencode(".png", annotated)
        if not encoded_ok:
            return jsonify({"error": "Could not encode the annotated result."}), 500
        return jsonify(
            {
                "image": name,
                "count": len(detections),
                "detections": [item.to_dict() for item in detections],
                "annotated_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        )

    @app.post("/api/pipeline")
    def pipeline_route() -> Any:
        payload = request.get_json(silent=True) or request.form
        acquisition_raw = str(payload.get("acquisition", "")).strip()
        if not acquisition_raw:
            return jsonify({"error": "acquisition is required."}), 400
        acquisition = resolve_asset_path(acquisition_raw)
        run_dir_raw = str(payload.get("run_dir", "")).strip()
        run_root_path: Path = app.extensions["holod3_run_root"]
        run_dir = (
            resolve_asset_path(run_dir_raw)
            if run_dir_raw
            else run_root_path / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        try:
            limit = int(payload.get("limit", 1))
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be an integer; use 0 for all frames."}), 400
        if limit < 0:
            return jsonify({"error": "limit must be zero or greater."}), 400
        job_config = selected.with_fallbacks(
            minip_bbox=_bool(payload.get("bbox_fallback"), selected.fallbacks.minip_bbox),
            depth_router=_bool(payload.get("depth_router"), selected.fallbacks.depth_router),
            diameter_underprediction=_bool(
                payload.get("diameter_fallback"), selected.fallbacks.diameter_underprediction
            ),
        )
        model_overrides = {
            field: str(payload.get(form_name, "")).strip()
            for form_name, field in {
                "yolo_weights": "yolo",
                "depth_primary_weights": "depth_primary",
                "depth_fallback_weights": "depth_fallback",
                "diameter_weights": "diameter",
            }.items()
            if str(payload.get(form_name, "")).strip()
        }
        if model_overrides:
            job_config = job_config.with_models(**model_overrides)
        if app.extensions["holod3_pipeline_injected"] and job_config != selected:
            return jsonify(
                {
                    "error": (
                        "This embedded Web app uses an injected pipeline and cannot apply per-request "
                        "model or fallback overrides. Construct the app with the desired pipeline config."
                    )
                }
            ), 400

        job_store: JobStore = app.extensions["holod3_jobs"]
        try:
            state = job_store.create(run_dir)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

        def run_job() -> None:
            job_store.update(state.job_id, status="running", message="Pipeline is running.")
            log_path = run_dir / "web_pipeline.log"
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                pipeline_instance = (
                    app.extensions["holod3_pipeline"]
                    if app.extensions["holod3_pipeline_injected"]
                    else HoloD3Pipeline(job_config)
                )
                with log_path.open("w", encoding="utf-8") as log:
                    result = pipeline_instance.run(
                        acquisition=acquisition,
                        run_dir=run_dir,
                        limit=limit,
                        create_visualization=True,
                        output_stream=log,
                    )
                job_store.update(
                    state.job_id,
                    status="complete",
                    message="Pipeline completed.",
                    result={
                        "particles_csv": _display_path(result.particles_csv),
                        "summary_json": _display_path(result.summary_json),
                        "visualization_html": _display_path(result.visualization_html),
                        "log": _display_path(log_path),
                        "downloads": {
                            "particles_csv": f"/api/jobs/{state.job_id}/files/particles_csv",
                            "summary_json": f"/api/jobs/{state.job_id}/files/summary_json",
                            "visualization_html": f"/api/jobs/{state.job_id}/files/visualization_html",
                            "log": f"/api/jobs/{state.job_id}/files/log",
                        },
                    },
                )
            except Exception as exc:  # noqa: BLE001
                job_store.update(state.job_id, status="failed", message=str(exc))

        threading.Thread(target=run_job, name=f"holod3-{state.job_id[:8]}", daemon=True).start()
        return jsonify(asdict(state)), 202

    @app.get("/api/jobs/<job_id>")
    def job_route(job_id: str) -> Any:
        state = app.extensions["holod3_jobs"].get(job_id)
        if state is None:
            return jsonify({"error": "Job not found."}), 404
        return jsonify(asdict(state))

    @app.get("/api/jobs/<job_id>/files/<kind>")
    def job_file_route(job_id: str, kind: str) -> Any:
        state = app.extensions["holod3_jobs"].get(job_id)
        if state is None or state.status != "complete" or state.result is None:
            return jsonify({"error": "Completed job not found."}), 404
        allowed = {"particles_csv", "summary_json", "visualization_html", "log"}
        if kind not in allowed or kind not in state.result:
            return jsonify({"error": "File not found."}), 404
        path = resolve_asset_path(str(state.result[kind]))
        run_dir = resolve_asset_path(state.run_dir)
        if not path.is_file() or not path.is_relative_to(run_dir):
            return jsonify({"error": "File not found."}), 404
        return send_file(path, as_attachment=kind != "visualization_html")

    return app
