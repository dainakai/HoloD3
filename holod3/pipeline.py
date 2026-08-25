"""Programmatic facade for detection, depth, diameter, and visualization."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import pandas as pd

from holod3.acquisition import AcquisitionConfig
from holod3.artifacts import load_production_manifest, verify_production_artifacts
from holod3.config import PipelineConfig, repository_root, resolve_asset_path
from holod3.visualization import write_particle_animation


@dataclass(frozen=True)
class PipelineResult:
    """Stable paths created by one pipeline run."""

    run_dir: Path
    particles_csv: Path
    summary_json: Path
    raw_detections_csv: Path
    fused_metrics_json: Path | None
    hybrid_metrics_json: Path | None
    visualization_html: Path | None = None
    returncode: int = 0

    def particles(self) -> pd.DataFrame:
        return pd.read_csv(self.particles_csv)

    def summary(self) -> dict[str, Any]:
        return json.loads(self.summary_json.read_text(encoding="utf-8"))


class HoloD3Pipeline:
    """Reusable interface to the validated detection, depth, and diameter chain."""

    def __init__(self, config: PipelineConfig | str = "production") -> None:
        self.config = PipelineConfig.preset(config) if isinstance(config, str) else config
        self.root = repository_root()

    def verify(self, *, check_hash: bool = True) -> None:
        statuses = verify_production_artifacts(root=self.root, check_hash=check_hash)
        failures = [status for status in statuses if not status.ok]
        if failures:
            details = "\n".join(
                f"- {status.artifact_id}: exists={status.exists}, size={status.size_matches}, hash={status.hash_matches}"
                for status in failures
            )
            raise RuntimeError(f"Production artifact verification failed:\n{details}")

    def validate_backend(self) -> None:
        if not self.config.runtime.strict_backend:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The production preset requires PyTorch and a CUDA GPU.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The production preset requires a CUDA GPU. Use portable-torch with a supported device otherwise."
            )
        try:
            import torch_tensorrt  # noqa: F401
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "The production preset requires the GPU extra and torch-tensorrt. "
                "Install with `uv sync --extra gpu`, or use the portable-torch preset."
            ) from exc

    def verify_configured_artifacts(self, *, check_hash: bool = True) -> None:
        """Verify only packaged checkpoints that the current config still uses."""

        role_to_config = {
            "detector": "yolo",
            "depth-primary": "depth_primary",
            "depth-fallback": "depth_fallback",
            "diameter": "diameter",
        }
        configured = self.config.model_paths(self.root)
        required_fields = self.config.required_model_fields()
        manifest = load_production_manifest(self.root / "models" / "production" / "manifest.json")
        artifact_ids = [
            str(artifact["id"])
            for artifact in manifest["artifacts"]
            if role_to_config[str(artifact["id"])] in required_fields
            and configured[role_to_config[str(artifact["id"])]]
            == resolve_asset_path(str(artifact["path"]), self.root)
        ]
        if not artifact_ids:
            return
        statuses = verify_production_artifacts(
            root=self.root,
            artifact_ids=artifact_ids,
            check_hash=check_hash,
        )
        failures = [status for status in statuses if not status.ok]
        if failures:
            details = "\n".join(
                f"- {status.artifact_id}: exists={status.exists}, size={status.size_matches}, hash={status.hash_matches}"
                for status in failures
            )
            raise RuntimeError(f"Configured packaged artifact verification failed:\n{details}")

    def resolve_runtime_devices(self) -> tuple[str, str]:
        """Resolve the portable ``auto`` policy without changing strict presets."""

        runtime = self.config.runtime
        if runtime.device != "auto" and runtime.yolo_device != "auto":
            return runtime.device, runtime.yolo_device
        import torch

        cuda_available = torch.cuda.is_available()
        device = ("cuda:0" if cuda_available else "cpu") if runtime.device == "auto" else runtime.device
        yolo_device = ("0" if cuda_available else "cpu") if runtime.yolo_device == "auto" else runtime.yolo_device
        return device, yolo_device

    @staticmethod
    def resolve_acquisition(value: str | Path | AcquisitionConfig) -> AcquisitionConfig:
        acquisition = value if isinstance(value, AcquisitionConfig) else AcquisitionConfig.load(value)
        acquisition.frame_records()
        return acquisition

    def validate_inputs(self, acquisition: str | Path | AcquisitionConfig) -> AcquisitionConfig:
        """Validate acquisition paths, synchronization, calibration, and schema."""

        return self.resolve_acquisition(acquisition)

    def build_command(
        self,
        *,
        acquisition: str | Path | AcquisitionConfig,
        run_dir: str | Path,
        limit: int = 30,
        start_index: int | None = None,
        end_index: int | None = None,
        overwrite: bool = False,
        stop_after_preprocessing: bool = False,
    ) -> list[str]:
        if limit < 0:
            raise ValueError("limit must be zero or greater")
        acquisition_config = self.validate_inputs(acquisition)
        if acquisition_config.source_path is None:
            raise ValueError("Subprocess execution requires an AcquisitionConfig loaded from a YAML file.")
        output = Path(run_dir).expanduser().resolve()
        models = self.config.require_model_files(self.root)
        detection = self.config.detection
        reconstruction = self.config.reconstruction
        fallbacks = self.config.fallbacks
        runtime = self.config.runtime
        device, yolo_device = self.resolve_runtime_devices()
        command = [
            sys.executable,
            str(self.root / "src" / "pipeline" / "run_pipeline_fused.py"),
            "--acquisition-config",
            str(acquisition_config.source_path),
            "--run-dir",
            str(output),
            "--limit",
            str(limit),
            "--device",
            device,
            "--yolo-device",
            yolo_device,
            "--yolo-weights",
            str(models["yolo"]),
            "--depth-checkpoint",
            str(models["depth_primary"]),
            "--depth-fallback-checkpoint",
            str(models["depth_fallback"]),
            "--slice-diam-weights",
            str(models["diameter"]),
            "--conf",
            str(detection.confidence),
            "--iou",
            str(detection.nms_iou),
            "--max-det",
            str(detection.max_detections),
            "--contrast-in-max",
            str(detection.contrast_input_max),
            "--yolo-imgsz",
            str(detection.image_size),
            "--yolo-batch-size",
            str(detection.batch_size),
            "--yolo-image-load-workers",
            str(detection.image_load_workers),
            "--depth-model-backend",
            runtime.depth_model_backend,
            "--depth-batch-size",
            str(runtime.depth_batch_size),
            "--depth-slice-block",
            str(reconstruction.slice_block),
            "--depth-crop-size",
            str(reconstruction.crop_size),
            "--depth-slice-start",
            str(reconstruction.slice_start),
            "--depth-slice-end",
            str(reconstruction.slice_end),
            "--depth-slice-step",
            str(reconstruction.slice_step),
            "--diam-batch-size",
            str(runtime.diameter_batch_size),
            "--diam-model-backend",
            runtime.diameter_model_backend,
            "--roi-image-prefetch-workers",
            str(runtime.roi_image_prefetch_workers),
            "--roi-image-prefetch-frames",
            str(runtime.roi_image_prefetch_frames),
            "--prefetch-holo-workers",
            str(runtime.hologram_prefetch_workers),
            "--prefetch-holo-frames",
            str(runtime.hologram_prefetch_frames),
            "--depth-router-max-diameter-um",
            str(fallbacks.depth_router_max_diameter_um),
            "--hybrid-ratio-threshold",
            str(fallbacks.diameter_ratio_threshold),
            "--hybrid-min-bbox-side-um",
            str(fallbacks.diameter_min_bbox_side_um),
        ]
        command.append("--bbox-fallback" if fallbacks.minip_bbox else "--no-bbox-fallback")
        command.append("--depth-router" if fallbacks.depth_router else "--no-depth-router")
        command.append(
            "--diameter-underprediction-fallback"
            if fallbacks.diameter_underprediction
            else "--no-diameter-underprediction-fallback"
        )
        command.append("--stream-csv-writes" if runtime.stream_csv_writes else "--no-stream-csv-writes")
        command.append(
            "--recenter-on-slice" if reconstruction.recenter_on_reconstructed_slice else "--no-recenter-on-slice"
        )
        if runtime.keep_intermediate_csv:
            command.append("--keep-intermediate-csv")
        if start_index is not None:
            command.extend(["--start-index", str(start_index)])
        if end_index is not None:
            command.extend(["--end-index", str(end_index)])
        if overwrite:
            command.append("--overwrite")
        if stop_after_preprocessing:
            command.append("--stop-after-preprocessing")
        return command

    def run(
        self,
        *,
        acquisition: str | Path | AcquisitionConfig,
        run_dir: str | Path,
        limit: int = 30,
        start_index: int | None = None,
        end_index: int | None = None,
        overwrite: bool = False,
        stop_after_preprocessing: bool = False,
        check_artifacts: bool = True,
        create_visualization: bool = True,
        output_stream: IO[str] | None = None,
        extra_args: Sequence[str] = (),
    ) -> PipelineResult:
        if check_artifacts:
            self.verify_configured_artifacts(check_hash=True)
        self.validate_backend()
        command = self.build_command(
            acquisition=acquisition,
            run_dir=run_dir,
            limit=limit,
            start_index=start_index,
            end_index=end_index,
            overwrite=overwrite,
            stop_after_preprocessing=stop_after_preprocessing,
        )
        command.extend(extra_args)
        completed = subprocess.run(
            command,
            cwd=self.root,
            check=False,
            text=True,
            stdout=output_stream,
            stderr=subprocess.STDOUT if output_stream is not None else None,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        output = Path(run_dir).expanduser().resolve()
        if stop_after_preprocessing:
            return PipelineResult(
                run_dir=output,
                particles_csv=output / "raw_detections.csv",
                summary_json=output / "pipeline_preprocessing_summary.json",
                raw_detections_csv=output / "raw_detections.csv",
                fused_metrics_json=None,
                hybrid_metrics_json=None,
            )
        result = PipelineResult(
            run_dir=output,
            particles_csv=output / "particles_3d.csv",
            summary_json=output / "pipeline_summary.json",
            raw_detections_csv=output / "raw_detections.csv",
            fused_metrics_json=output / "fused_depth_slice_metrics.json",
            hybrid_metrics_json=output / "hybrid_diameter_metrics.json",
            visualization_html=output / "particles_3d.html" if create_visualization else None,
        )
        required_outputs = (
            result.particles_csv,
            result.summary_json,
            result.raw_detections_csv,
            result.fused_metrics_json,
            result.hybrid_metrics_json,
        )
        missing = [path for path in required_outputs if path is None or not path.is_file()]
        if missing:
            raise RuntimeError(f"Pipeline completed without required outputs: {missing}")
        if result.visualization_html is not None:
            write_particle_animation(result.particles_csv, result.visualization_html)
            if not result.visualization_html.is_file():
                raise RuntimeError(f"Pipeline visualization was not created: {result.visualization_html}")
        return result
