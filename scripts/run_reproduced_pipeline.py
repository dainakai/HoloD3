#!/usr/bin/env python3
"""Run the holdout pipeline with checkpoints produced by the reproduction plan."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from holod3.config import PipelineConfig
from holod3.pipeline import HoloD3Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Torch holdout inference using reproduced checkpoints.")
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--depth-primary", required=True)
    parser.add_argument("--depth-fallback", required=True)
    parser.add_argument("--diameter", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = PipelineConfig.preset("portable-torch")
    config = replace(
        config,
        models=replace(
            config.models,
            yolo=args.yolo,
            depth_primary=args.depth_primary,
            depth_fallback=args.depth_fallback,
            diameter=args.diameter,
        ),
        runtime=replace(
            config.runtime,
            device=args.device,
            yolo_device=args.yolo_device,
            strict_backend=False,
        ),
    )
    HoloD3Pipeline(config).run(
        acquisition=args.acquisition,
        run_dir=args.run_dir,
        limit=args.limit,
        check_artifacts=False,
    )


if __name__ == "__main__":
    main()
