from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from holod3.artifacts import fetch_model_artifacts, load_production_manifest, load_remote_manifest
from holod3.config import PipelineConfig, repository_root
from holod3.detector import Detection, contrast_stretch_0_75_to_255, prepare_minip
from src.detection.depth_runtime import add_condition_channels


def test_presets_use_semantic_model_paths_and_preserve_backend_policy() -> None:
    production = PipelineConfig.preset("production")
    portable = PipelineConfig.preset("portable-torch")
    expected = {
        "models/production/detector.pt",
        "models/production/depth-primary.pt",
        "models/production/depth-fallback.pt",
        "models/production/diameter.pt",
    }
    assert set(production.to_dict()["models"].values()) == expected
    assert production.runtime.strict_backend
    assert production.runtime.depth_model_backend == "tensorrt"
    assert not portable.runtime.strict_backend
    assert portable.runtime.depth_model_backend == "torch"
    assert portable.runtime.device == "auto" and portable.runtime.yolo_device == "auto"


def test_strict_backend_rejects_torch_model_backends() -> None:
    with pytest.raises(ValueError, match="strict_backend requires TensorRT"):
        PipelineConfig.preset("portable-torch").with_runtime(strict_backend=True)


def test_production_and_remote_manifests_have_one_unambiguous_role_map() -> None:
    production = load_production_manifest()
    remote = load_remote_manifest()
    assert production["pipeline"] == "production"
    assert {artifact["id"] for artifact in production["artifacts"]} == {
        "detector",
        "depth-primary",
        "depth-fallback",
        "diameter",
    }
    production_paths = {artifact["path"] for artifact in production["artifacts"]}
    remote_production = {
        artifact["path"] for artifact in remote["artifacts"] if "production" in artifact["scopes"]
    }
    assert production_paths == remote_production
    assert len(remote["revision"]) == 40
    for artifact in [*production["artifacts"], *production["training_initializers"]]:
        assert len(artifact["sha256"]) == 64
        int(artifact["sha256"], 16)


def test_remote_model_fetch_downloads_only_selected_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"deterministic model fixture"
    digest = hashlib.sha256(content).hexdigest()
    manifest = tmp_path / "models/remote_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "huggingface",
                "repo_id": "owner/private-models",
                "repo_type": "model",
                "revision": "pinned-commit",
                "artifacts": [
                    {
                        "path": "models/production/test.pt",
                        "bytes": len(content),
                        "sha256": digest,
                        "scopes": ["production", "reproduction", "all"],
                    },
                    {
                        "path": "models/initializers/test.pt",
                        "bytes": len(content),
                        "sha256": digest,
                        "scopes": ["reproduction", "all"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_download(**kwargs: object) -> Path:
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        calls.append(str(kwargs["filename"]))
        return destination

    monkeypatch.setattr("holod3.artifacts._download_hugging_face_file", fake_download)
    statuses = fetch_model_artifacts(root=tmp_path, scope="production")
    assert calls == ["models/production/test.pt"]
    assert len(statuses) == 1 and statuses[0].ok and statuses[0].downloaded


def test_model_fetch_refuses_size_only_installation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_model_artifacts(root=tmp_path, check_hash=False)


def test_detector_preprocessing_and_detection_geometry() -> None:
    image = np.asarray([[0, 75, 100]], dtype=np.uint8)
    np.testing.assert_array_equal(contrast_stretch_0_75_to_255(image), [[0, 255, 255]])
    prepared = prepare_minip(image)
    assert prepared.shape == (1, 3, 3)
    detection = Detection(0, 0.9, 2.0, 4.0, 8.0, 10.0, 20, 20)
    assert detection.center_x == 5.0 and detection.center_y == 7.0
    assert detection.width == 6.0 and detection.height == 6.0


def test_detector_auto_device_resolves_before_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    from holod3.detector import ParticleDetector

    assert ParticleDetector._normalize_device("auto") == "0"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert ParticleDetector._normalize_device("auto") == "cpu"


def test_depth_mask_conditioning_uses_acquisition_pixel_pitch() -> None:
    import torch

    raw = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    diameter = torch.tensor([80.0])
    pitch_8 = add_condition_channels(raw, diameter, "raw_mask", 0.5, 0.0, 1.0, 8.0)
    pitch_10 = add_condition_channels(raw, diameter, "raw_mask", 0.5, 0.0, 1.0, 10.0)
    assert not torch.equal(pitch_8[:, 1], pitch_10[:, 1])


def test_no_checkpoint_is_tracked_as_a_git_object() -> None:
    root = repository_root()
    tracked = (root / ".gitignore").read_text(encoding="utf-8")
    assert "*.pt" in tracked


def test_disabled_depth_router_does_not_require_fallback_checkpoint(tmp_path: Path) -> None:
    config = PipelineConfig.preset("portable-torch").with_fallbacks(depth_router=False)
    overrides = {}
    for field in ("yolo", "depth_primary", "diameter"):
        path = tmp_path / f"{field}.pt"
        path.write_bytes(b"fixture")
        overrides[field] = str(path)
    config = config.with_models(**overrides, depth_fallback=str(tmp_path / "missing.pt"))
    resolved = config.require_model_files()
    assert resolved["depth_fallback"] == (tmp_path / "missing.pt").resolve()


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        ("detection", {"contrast_input_max": 0}, "contrast_input_max"),
        ("detection", {"image_load_workers": -1}, "image_load_workers"),
        ("fallbacks", {"diameter_min_bbox_side_um": float("nan")}, "diameter_min_bbox_side_um"),
        ("runtime", {"hologram_prefetch_workers": -1}, "hologram_prefetch_workers"),
    ],
)
def test_inference_config_rejects_malformed_values(
    section: str, changes: dict[str, object], message: str
) -> None:
    config = PipelineConfig.preset("portable-torch")
    malformed = replace(config, **{section: replace(getattr(config, section), **changes)})
    with pytest.raises(ValueError, match=message):
        malformed.validate()
