"""Command-line interface for inference, verification, training, and the Web UI."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

from holod3.acquisition import AcquisitionConfig
from holod3.artifacts import fetch_model_artifacts, verify_production_artifacts
from holod3.config import PipelineConfig, repository_root
from holod3.datasets import fetch_dataset_bundles
from holod3.detector import ParticleDetector
from holod3.pipeline import HoloD3Pipeline
from holod3.visualization import write_particle_animation


def _load_pipeline_config(value: str) -> PipelineConfig:
    candidate = Path(value).expanduser()
    return PipelineConfig.load(candidate) if candidate.is_file() else PipelineConfig.preset(value)


def _config_with_cli_overrides(args: argparse.Namespace) -> PipelineConfig:
    config = _load_pipeline_config(args.preset)
    model_changes: dict[str, Any] = {}
    for argument, field in {
        "yolo_weights": "yolo",
        "depth_primary_weights": "depth_primary",
        "depth_fallback_weights": "depth_fallback",
        "diameter_weights": "diameter",
    }.items():
        value = getattr(args, argument, None)
        if value is not None:
            model_changes[field] = value
    if model_changes:
        config = config.with_models(**model_changes)
    runtime_changes: dict[str, Any] = {}
    if getattr(args, "device", None) is not None:
        runtime_changes["device"] = args.device
    if getattr(args, "yolo_device", None) is not None:
        runtime_changes["yolo_device"] = args.yolo_device
    if getattr(args, "depth_backend", None) is not None:
        runtime_changes["depth_model_backend"] = args.depth_backend
    if getattr(args, "diameter_backend", None) is not None:
        runtime_changes["diameter_model_backend"] = args.diameter_backend
    if getattr(args, "strict_backend", None) is not None:
        runtime_changes["strict_backend"] = args.strict_backend
    elif any(runtime_changes.get(key) == "torch" for key in ("depth_model_backend", "diameter_model_backend")):
        runtime_changes["strict_backend"] = False
    if runtime_changes:
        config = config.with_runtime(**runtime_changes)
    fallback_changes: dict[str, Any] = {}
    mappings = {
        "bbox_fallback": "minip_bbox",
        "depth_router": "depth_router",
        "diameter_fallback": "diameter_underprediction",
    }
    for argument, field in mappings.items():
        value = getattr(args, argument, None)
        if value is not None:
            fallback_changes[field] = value
    if fallback_changes:
        config = config.with_fallbacks(**fallback_changes)
    return config


def command_verify(args: argparse.Namespace) -> int:
    statuses = verify_production_artifacts(check_hash=not args.fast)
    payload = [
        {
            "id": status.artifact_id,
            "path": str(status.path),
            "exists": status.exists,
            "size_matches": status.size_matches,
            "hash_matches": status.hash_matches,
            "ok": status.ok,
        }
        for status in statuses
    ]
    print(
        json.dumps(
            {
                "ok": all(status.ok for status in statuses),
                "verification": "size-only" if args.fast else "sha256",
                "artifacts": payload,
            },
            indent=2,
        )
    )
    return 0 if all(status.ok for status in statuses) else 1


def command_fetch_models(args: argparse.Namespace) -> int:
    statuses = fetch_model_artifacts(
        scope=args.scope,
        repo_id=args.repo_id,
        revision=args.revision,
        force=args.force,
        check_hash=True,
    )
    payload = [
        {
            "path": str(status.path),
            "downloaded": status.downloaded,
            "size_matches": status.size_matches,
            "hash_matches": status.hash_matches,
            "ok": status.ok,
        }
        for status in statuses
    ]
    print(
        json.dumps(
            {
                "ok": all(status.ok for status in statuses),
                "scope": args.scope,
                "downloaded": sum(status.downloaded for status in statuses),
                "artifacts": payload,
            },
            indent=2,
        )
    )
    return 0


def command_fetch_data(args: argparse.Namespace) -> int:
    statuses = fetch_dataset_bundles(
        scope=args.scope,
        bundle_ids=args.bundle or None,
        repo_id=args.repo_id,
        revision=args.revision,
        force=args.force,
        check_hash=True,
    )
    print(
        json.dumps(
            {
                "ok": all(status.ok for status in statuses),
                "scope": args.scope,
                "bundles": [
                    {
                        "id": status.bundle_id,
                        "destination": str(status.destination),
                        "downloaded": status.downloaded,
                        "extracted": status.extracted,
                        "sha256": status.archive_sha256,
                    }
                    for status in statuses
                ],
            },
            indent=2,
        )
    )
    return 0


def command_config(args: argparse.Namespace) -> int:
    print(json.dumps(_load_pipeline_config(args.preset).to_dict(), indent=2))
    return 0


def command_detect(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.preset)
    detector = ParticleDetector(weights=args.weights, config=config, device=args.device)
    source = Path(args.input).expanduser().resolve()
    if source.is_dir():
        frame = detector.predict_directory(source, limit=args.limit)
        output = Path(args.output or "runs/detect/detections.csv").expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(
            json.dumps(
                {
                    "images": int(frame["file"].nunique()) if not frame.empty else 0,
                    "detections": len(frame),
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0
    if not source.is_file():
        raise FileNotFoundError(source)
    detections = detector.predict(source)
    payload = {
        "image": str(source),
        "weights": str(detector.weights),
        "count": len(detections),
        "detections": [detection.to_dict() for detection in detections],
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.annotated_output:
        annotated_path = Path(args.annotated_output).expanduser().resolve()
        if annotated_path.exists() and not args.overwrite:
            raise FileExistsError(f"{annotated_path} exists; pass --overwrite to replace it")
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(annotated_path), detector.annotate(source, detections)):
            raise RuntimeError(f"Could not write annotated image: {annotated_path}")
        payload["annotated_output"] = str(annotated_path)
    print(json.dumps(payload, indent=2))
    return 0


def command_infer(args: argparse.Namespace) -> int:
    config = _config_with_cli_overrides(args)
    pipeline = HoloD3Pipeline(config)
    command = pipeline.build_command(
        acquisition=args.acquisition,
        run_dir=args.run_dir,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        overwrite=args.overwrite,
        stop_after_preprocessing=args.stop_after_preprocessing,
    )
    if args.dry_run:
        print(shlex.join(command))
        return 0
    result = pipeline.run(
        acquisition=args.acquisition,
        run_dir=args.run_dir,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        overwrite=args.overwrite,
        stop_after_preprocessing=args.stop_after_preprocessing,
        check_artifacts=not args.skip_artifact_check,
        create_visualization=not args.no_visualization,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "particles_csv": str(result.particles_csv),
                "summary_json": str(result.summary_json),
                "visualization_html": str(result.visualization_html) if result.visualization_html else None,
            },
            indent=2,
        )
    )
    return 0


def command_validate_acquisition(args: argparse.Namespace) -> int:
    acquisition = AcquisitionConfig.load(args.acquisition)
    records = acquisition.validate_image_contracts()
    payload = {
        "ok": True,
        "name": acquisition.name,
        "mode": acquisition.mode,
        "frames": len(records),
        "first_frame": records[0].stem,
        "last_frame": records[-1].stem,
        "minip": "provided" if acquisition.minip_dir is not None else "generated from raw holograms",
        "optics": acquisition.to_dict()["optics"],
    }
    if acquisition.mode == "single_gabor":
        payload["model_domain_warning"] = (
            "Packaged depth and diameter checkpoints were trained on dual-camera phase-retrieval crops; "
            "single-Gabor accuracy is not validated."
        )
    print(json.dumps(payload, indent=2))
    return 0


def command_visualize(args: argparse.Namespace) -> int:
    output = write_particle_animation(
        args.particles_csv,
        args.output,
        title=args.title,
        embed_plotly=not args.cdn,
    )
    print(json.dumps({"particles_csv": str(Path(args.particles_csv).resolve()), "visualization_html": str(output)}, indent=2))
    return 0


def command_demo(args: argparse.Namespace) -> int:
    config = _config_with_cli_overrides(args)
    pipeline = HoloD3Pipeline(config)
    acquisition = repository_root() / "data" / "demo" / "experimental" / "acquisition.yaml"
    result = pipeline.run(
        acquisition=acquisition,
        run_dir=args.run_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        check_artifacts=not args.skip_artifact_check,
        create_visualization=True,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "particles_csv": str(result.particles_csv),
                "visualization_html": str(result.visualization_html),
            },
            indent=2,
        )
    )
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "run_reproduction.py"),
        "--config",
        args.config,
        "--stage",
        args.stage,
    ]
    if args.dry_run:
        command.append("--dry-run")
    command.extend(["--yolo-baseline-source", args.yolo_baseline_source])
    return subprocess.run(command, cwd=repository_root(), check=False).returncode


def command_web(args: argparse.Namespace) -> int:
    from holod3.web import create_app

    config = _load_pipeline_config(args.preset)
    if args.device is not None:
        config = config.with_runtime(device=args.device, yolo_device=args.device)
    app = create_app(config=config, run_root=args.run_root)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holod3",
        description="Detect hologram particles and estimate their 3D depth and diameter.",
    )
    parser.add_argument("--version", action="version", version="HoloD3 0.1.1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Verify downloaded production model hashes.")
    verify.add_argument("--fast", action="store_true", help="Check files and sizes without computing SHA-256.")
    verify.set_defaults(handler=command_verify)

    fetch_models = subparsers.add_parser(
        "fetch-models",
        help="Download SHA-256-verified checkpoints from the private Hugging Face repository.",
    )
    fetch_models.add_argument(
        "--scope",
        choices=["production", "reproduction", "all"],
        default="production",
        help="production: four inference weights; reproduction: inference plus exact initializers; all: archive.",
    )
    fetch_models.add_argument("--repo-id", default=None, help="Override the manifest Hugging Face repository ID.")
    fetch_models.add_argument("--revision", default=None, help="Override the pinned Hugging Face revision.")
    fetch_models.add_argument("--force", action="store_true", help="Redownload files that already verify.")
    fetch_models.set_defaults(handler=command_fetch_models)

    fetch_data = subparsers.add_parser(
        "fetch-data",
        help="Download and verify logical training/evaluation bundles from the private dataset repository.",
    )
    fetch_data.add_argument("--scope", choices=["training", "evaluation", "all"], default="training")
    fetch_data.add_argument("--bundle", action="append", help="Fetch one bundle id; repeat for multiple bundles.")
    fetch_data.add_argument("--repo-id", default=None)
    fetch_data.add_argument("--revision", default=None)
    fetch_data.add_argument("--force", action="store_true")
    fetch_data.set_defaults(handler=command_fetch_data)

    show_config = subparsers.add_parser("config", help="Print a resolved inference preset.")
    show_config.add_argument("--preset", default="production", help="Preset name or inference YAML path.")
    show_config.set_defaults(handler=command_config)

    detect = subparsers.add_parser("detect", help="Run the production YOLO checkpoint directly on MinIP images.")
    detect.add_argument("input", help="A MinIP image or a directory of MinIP images.")
    detect.add_argument("--weights", default=None, help="Optional YOLO checkpoint path.")
    detect.add_argument("--preset", default="portable-torch")
    detect.add_argument("--device", default=None, help="Ultralytics device such as 0 or cpu.")
    detect.add_argument("--limit", type=int, default=None, help="Maximum images when input is a directory.")
    detect.add_argument("--output", default=None, help="JSON for one image or CSV for a directory.")
    detect.add_argument("--annotated-output", default=None, help="Annotated image path for single-image input.")
    detect.add_argument("--overwrite", action="store_true")
    detect.set_defaults(handler=command_detect)

    validate_acquisition = subparsers.add_parser(
        "validate-acquisition", help="Validate paths, synchronization, calibration, and optical settings."
    )
    validate_acquisition.add_argument("acquisition", help="Path to acquisition.yaml.")
    validate_acquisition.set_defaults(handler=command_validate_acquisition)

    infer = subparsers.add_parser("infer", help="Run MinIP preparation, detection, depth, and diameter inference.")
    infer.add_argument("--acquisition", required=True, help="Path to acquisition.yaml.")
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--preset", default="production", help="Preset name or inference YAML path.")
    infer.add_argument("--limit", type=int, default=1, help="Maximum frames; use 0 to process all frames.")
    infer.add_argument("--start-index", type=int, default=None)
    infer.add_argument("--end-index", type=int, default=None)
    infer.add_argument("--device", default=None)
    infer.add_argument("--yolo-device", default=None)
    infer.add_argument("--depth-backend", choices=["tensorrt", "torch"], default=None)
    infer.add_argument("--diameter-backend", choices=["tensorrt", "torch"], default=None)
    infer.add_argument("--strict-backend", action=argparse.BooleanOptionalAction, default=None)
    infer.add_argument("--yolo-weights", default=None)
    infer.add_argument("--depth-primary-weights", default=None)
    infer.add_argument("--depth-fallback-weights", default=None)
    infer.add_argument("--diameter-weights", default=None)
    infer.add_argument("--bbox-fallback", action=argparse.BooleanOptionalAction, default=None)
    infer.add_argument("--depth-router", action=argparse.BooleanOptionalAction, default=None)
    infer.add_argument("--diameter-fallback", action=argparse.BooleanOptionalAction, default=None)
    infer.add_argument("--stop-after-preprocessing", action="store_true")
    infer.add_argument("--skip-artifact-check", action="store_true")
    infer.add_argument("--no-visualization", action="store_true", help="Skip particles_3d.html generation.")
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--dry-run", action="store_true")
    infer.set_defaults(handler=command_infer)

    demo = subparsers.add_parser("demo", help="Run the included experimental acquisition and create CSV + 3D HTML.")
    demo.add_argument("--run-dir", default="runs/demo")
    demo.add_argument("--preset", default="portable-torch", help="Preset name or inference YAML path.")
    demo.add_argument("--limit", type=int, default=0, help="Maximum demo frames; 0 uses all included frames.")
    demo.add_argument("--device", default=None)
    demo.add_argument("--yolo-device", default=None)
    demo.add_argument("--depth-backend", choices=["tensorrt", "torch"], default=None)
    demo.add_argument("--diameter-backend", choices=["tensorrt", "torch"], default=None)
    demo.add_argument("--strict-backend", action=argparse.BooleanOptionalAction, default=None)
    demo.add_argument("--yolo-weights", default=None)
    demo.add_argument("--depth-primary-weights", default=None)
    demo.add_argument("--depth-fallback-weights", default=None)
    demo.add_argument("--diameter-weights", default=None)
    demo.add_argument("--bbox-fallback", action=argparse.BooleanOptionalAction, default=None)
    demo.add_argument("--depth-router", action=argparse.BooleanOptionalAction, default=None)
    demo.add_argument("--diameter-fallback", action=argparse.BooleanOptionalAction, default=None)
    demo.add_argument("--skip-artifact-check", action="store_true")
    demo.add_argument("--overwrite", action="store_true")
    demo.set_defaults(handler=command_demo)

    visualize = subparsers.add_parser("visualize", help="Create an animated 3D HTML from particles_3d.csv.")
    visualize.add_argument("particles_csv")
    visualize.add_argument("--output", default="particles_3d.html")
    visualize.add_argument("--title", default="HoloD3 particle measurements")
    visualize.add_argument("--cdn", action="store_true", help="Use Plotly from a CDN instead of embedding it.")
    visualize.set_defaults(handler=command_visualize)

    reproduce = subparsers.add_parser("reproduce", help="Validate or execute the exact training lineage.")
    reproduce.add_argument(
        "--config",
        default="configs/training/reproduce-production.json",
        help="Repository-relative reproduction configuration.",
    )
    reproduce.add_argument(
        "--stage",
        required=True,
        choices=[
            "check",
            "yolo-baseline",
            "yolo-production",
            "depth-fallback",
            "depth-primary",
            "diameter",
            "evaluate",
            "all",
        ],
    )
    reproduce.add_argument("--dry-run", action="store_true")
    reproduce.add_argument(
        "--yolo-baseline-source",
        choices=["packaged", "reproduced"],
        default="packaged",
        help="Choose the exact packaged baseline or the baseline trained by this reproduction plan.",
    )
    reproduce.set_defaults(handler=command_reproduce)

    web = subparsers.add_parser("web", help="Start the local HoloD3 Web UI.")
    web.add_argument("--preset", default="portable-torch")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=7860)
    web.add_argument("--device", default=None)
    web.add_argument("--run-root", default="runs/web")
    web.add_argument("--debug", action="store_true")
    web.set_defaults(handler=command_web)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"holod3: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
