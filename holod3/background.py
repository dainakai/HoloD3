"""Prepare shared, background-corrected holograms for reconstruction and inference."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from holod3.acquisition import AcquisitionConfig, FrameRecord
from src.preprocessing.backrem_adaptive import AdaptiveConfig
from src.preprocessing.backrem_streaming import (
    correct_frames_streaming,
    estimate_anchor_backgrounds_cuda_to_disk,
    estimate_anchor_backgrounds_rolling_cuda_to_disk,
    estimate_anchor_backgrounds_to_disk,
    resolve_median_backend,
)


def _prepared_mapping(config: AcquisitionConfig, manifest: Path) -> dict[str, Any]:
    """Rebase external calibration and file transforms to the prepared acquisition."""

    value = config.to_dict()
    value["background_removal"]["enabled"] = False
    value["frames"] = {
        "primary_holograms": "primary",
        "secondary_holograms": "secondary" if config.mode == "dual_phase_retrieval" else None,
        "minip": None,
    }
    coefficients = config.distortion_coefficients_path
    if coefficients is not None:
        value["calibration"]["secondary_distortion_coefficients"] = os.path.relpath(coefficients, manifest.parent)
    for steps in value["transforms"].values():
        for step in steps:
            function = step["function"]
            if ":" not in function:
                continue
            module, attribute = function.rsplit(":", 1)
            if Path(module).suffix == ".py" or "/" in module or "\\" in module:
                module_path = config.resolve_path(module)
                assert module_path is not None
                step["function"] = f"{os.path.relpath(module_path, manifest.parent)}:{attribute}"
    return value


def prepare_background_holograms(
    config: AcquisitionConfig,
    records: list[FrameRecord],
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> tuple[AcquisitionConfig, list[FrameRecord], dict[str, Any]]:
    """Correct each camera in sensor coordinates before transforms and calibration.

    Background anchors use the complete synchronized source sequence, including
    when inference selects just one frame. Only selected corrected frames are
    written. Memory is bounded by one temporal window and one correction batch.
    The returned acquisition disables background removal to prevent applying it
    again when the depth/diameter subprocess reads the prepared images.
    """

    config.validate_schema()
    settings = config.background_removal
    if not settings.enabled:
        return config, records, {"enabled": False}
    source_records = config.frame_records()
    if len(source_records) < 2:
        raise ValueError(
            "Background removal requires at least two source frames. Supply a temporal sequence "
            "or set background_removal.enabled: false for an isolated hologram."
        )
    source_by_stem = {record.stem: record for record in source_records}
    if not records or len({record.stem for record in records}) != len(records):
        raise ValueError("Background removal requires a nonempty selection of distinct frames")
    if any(source_by_stem.get(record.stem) != record for record in records):
        raise ValueError("Selected background-removal frames must belong to the acquisition")
    selected_stems = {record.stem for record in records}
    selected_positions = {index for index, record in enumerate(source_records) if record.stem in selected_stems}

    destination = Path(output_dir).expanduser().resolve()
    for source_dir in (config.primary_dir, config.secondary_dir):
        if source_dir is not None and (
            destination.is_relative_to(source_dir) or source_dir.is_relative_to(destination)
        ):
            raise ValueError("Background output directory must be separate from the source hologram directories")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Prepared backgrounds already exist: {destination}. Choose a new run or pass --overwrite.")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Background output must be a directory: {destination}")

    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    torch_device = torch.device(requested_device)
    backend = resolve_median_backend(settings.median_backend, torch_device, settings.window)
    estimate = {
        "torch": estimate_anchor_backgrounds_to_disk,
        "cuda-window": estimate_anchor_backgrounds_cuda_to_disk,
        "cuda-rolling": estimate_anchor_backgrounds_rolling_cuda_to_disk,
    }[backend]
    algorithm = AdaptiveConfig(
        window=settings.window,
        anchor_stride=settings.anchor_stride,
        batch_size=settings.batch_size,
        lowpass=settings.lowpass,
        target_level=settings.target_level,
    )
    summary: dict[str, Any] = {
        "enabled": True,
        "config": asdict(settings),
        "source_frames": len(source_records),
        "selected_frames": len(records),
        "background_source": "complete_acquisition",
        "median_backend": backend,
        "device": str(torch_device),
        "output_dir": str(destination),
        "prepared_acquisition_config": str(destination / "acquisition.yaml"),
        "cameras": {},
    }
    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    device_context = torch.cuda.device(torch_device) if torch_device.type == "cuda" else nullcontext()
    with device_context, tempfile.TemporaryDirectory(prefix=".background-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        roles = ("primary", "secondary") if config.mode == "dual_phase_retrieval" else ("primary",)
        for role in roles:
            print(json.dumps({"event": "background_removal_started", "camera": role, "frames": len(source_records)}), flush=True)
            paths = [getattr(record, role) for record in source_records]
            anchors_dir = staging / role / "_anchors"
            anchors, means, timings = estimate(paths, algorithm, torch_device, anchors_dir)
            target = settings.target_level if settings.target_level is not None else float(np.median(means))
            timings.update(
                correct_frames_streaming(
                    paths,
                    staging / role,
                    algorithm,
                    torch_device,
                    anchors_dir,
                    anchors,
                    target,
                    settings.save_workers,
                    selected_positions=selected_positions,
                )
            )
            summary["cameras"][role] = {"target_level": target, "anchor_positions": anchors, "timings": timings}
            print(json.dumps({"event": "background_removal_done", "camera": role, "frames": len(records)}), flush=True)
        (staging / "acquisition.yaml").write_text(
            yaml.safe_dump(_prepared_mapping(config, destination / "acquisition.yaml"), sort_keys=False),
            encoding="utf-8",
        )
        summary["elapsed_sec"] = time.perf_counter() - started
        (staging / "background_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        # Replace all previously selected frames together, after both cameras succeed.
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)

    prepared = AcquisitionConfig.load(destination / "acquisition.yaml")
    prepared_by_stem = {record.stem: record for record in prepared.frame_records()}
    return prepared, [prepared_by_stem[record.stem] for record in records], summary
