"""Audit every depth score and benchmark output-preserving CUDA optimizations.

Run as a module with installed production checkpoints. Outputs are written to a
new directory. The first eager/graph pair is an untimed correctness audit; later
pairs include per-frame graph capture but exclude engine building/preprocessing.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from holod3.acquisition import AcquisitionConfig
from holod3.config import PipelineConfig, repository_root
from holod3.pipeline import HoloD3Pipeline
from holod3.reconstruction import build_propagation_setup, load_wavefront, minimum_intensity_projection
from src.common.cuda_graph import CudaGraphScorer
from src.detection import depth_and_slice_fused as core


def memoize(function):
    cache = {}

    @wraps(function)
    def call(*args, **kwargs):
        key = (str(args), str(kwargs))
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cache[key]

    return call


def memoize_optimizer(function):
    cache = {}

    @wraps(function)
    def call(model, *args, **kwargs):
        # One backend/config/device per process; the same loaded model owns the
        # exact same compiled engine in every eager and graph comparison.
        if id(model) not in cache:
            cache[id(model)] = function(model, *args, **kwargs)
        return cache[id(model)]

    return call


def bench_minip(acquisition: AcquisitionConfig, device: torch.device, repeats: int, limit: int) -> dict:
    setup = build_propagation_setup(acquisition, device)
    records = acquisition.frame_records()
    if limit:
        records = records[:limit]
    timings = {"torch": [], "cuda": []}
    with torch.inference_mode():
        for record in records:
            wave = load_wavefront(acquisition, record, setup, device)
            kwargs = {"slice_step": acquisition.reconstruction.minip_slice_step}
            expected = minimum_intensity_projection(wave, setup, backend="torch", **kwargs)
            actual = minimum_intensity_projection(wave, setup, backend="cuda", **kwargs)
            assert torch.equal(actual, expected), record.stem
            for repeat in range(repeats):
                order = ("torch", "cuda") if repeat % 2 == 0 else ("cuda", "torch")
                for backend in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    actual = minimum_intensity_projection(wave, setup, backend=backend, **kwargs)
                    torch.cuda.synchronize(device)
                    timings[backend].append(time.perf_counter() - start)
                    assert torch.equal(actual, expected), (record.stem, backend)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "frames": len(records), "all_pixels_equal": True, "seconds_per_frame": timings,
        "median_seconds_per_frame": medians, "speedup": medians["torch"] / medians["cuda"],
        "scope": "MinIP only; warmed FFT plan and kernel; excludes wavefront, I/O, and first NVRTC compilation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, default=repository_root() / "data/demo/experimental/acquisition.yaml")
    parser.add_argument("--backend", choices=("torch", "tensorrt"), default="tensorrt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 audits all acquisition frames")
    parser.add_argument("--repeats", type=int, default=3, help="At least 2; first pair audits all scores")
    args = parser.parse_args()
    if args.repeats < 2 or args.limit < 0:
        parser.error("repeats must be at least 2 and limit must be nonnegative")
    acquisition = AcquisitionConfig.load(args.acquisition)
    if acquisition.background_removal.enabled:
        parser.error("For background-corrected inputs, use the prepared _inputs/background/acquisition.yaml")
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    summary = {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "backend": args.backend,
        "acquisition": acquisition.name, "slice_count": acquisition.optics.slice_count,
        "repeats": args.repeats, "results": [],
    }
    summary["minip"] = bench_minip(acquisition, device, args.repeats, args.limit)
    print(json.dumps({"minip": summary["minip"]}), flush=True)
    config = PipelineConfig.preset("portable-torch").with_runtime(
        device=str(device), yolo_device=str(args.device), keep_intermediate_csv=True,
    )
    with (output / "preprocessing.log").open("w") as log:
        HoloD3Pipeline(config).run(
            acquisition=acquisition, run_dir=output / "inputs", limit=args.limit,
            stop_after_preprocessing=True, create_visualization=False, output_stream=log,
        )
    source = output / "inputs/tracks_with_diameter.csv"
    assert source.is_file(), source
    scores: list[torch.Tensor] = []
    audit = {"enabled": False, "graph": False, "calls": 0, "scores": 0}
    scorer_call = CudaGraphScorer.__call__

    def audited_call(self, crops, diameters):
        result = scorer_call(self, crops, diameters)
        if audit["enabled"]:
            current = result.cpu()
            if audit["graph"]:
                assert torch.equal(current, scores[audit["calls"]]), audit["calls"]
                audit["scores"] += current.numel()
            else:
                scores.append(current)
            audit["calls"] += 1
        return result

    with contextlib.ExitStack() as stack:
        for name in ("load_depth_model", "load_slice_model"):
            stack.enter_context(patch.object(core, name, memoize(getattr(core, name))))
        for name in ("optimize_model_backend", "optimize_slice_model_backend"):
            stack.enter_context(patch.object(core, name, memoize_optimizer(getattr(core, name))))
        stack.enter_context(patch.object(CudaGraphScorer, "__call__", audited_call))
        if args.backend == "tensorrt":
            import torch_tensorrt

            compile_model = torch_tensorrt.compile

            def compile_local(*a, **kw):
                kw["timing_cache_path"] = str(output / "tensorrt-timing.bin")
                kw["engine_cache_dir"] = str(output / "tensorrt-engines")
                return compile_model(*a, **kw)

            stack.enter_context(patch.object(torch_tensorrt, "compile", compile_local))
        for repeat in range(args.repeats):
            # Audit baseline must run first. Reverse alternate measured pairs
            # to limit bias from thermal state and run order.
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for use_graph in order:
                mode = "graph" if use_graph else "eager"
                run = output / f"{mode}-{repeat}"
                run.mkdir()
                audit.update(enabled=repeat == 0, graph=use_graph, calls=0)
                argv = [
                    str(core.__file__), "--run-dir", str(run), "--acquisition-config", str(args.acquisition.resolve()),
                    "--input", str(source), "--frames", str(args.limit), "--device", str(device),
                    "--model-backend", args.backend, "--diam-model-backend", args.backend,
                    "--final-output", str(run / "particles.csv"),
                    "--depth-cuda-graphs" if use_graph else "--no-depth-cuda-graphs",
                ]
                torch.cuda.reset_peak_memory_stats(device)
                with (run / "core.log").open("w") as log, contextlib.redirect_stdout(log), patch.object(sys, "argv", argv):
                    core.main()
                if repeat == 0 and use_graph:
                    assert audit["calls"] == len(scores)
                    summary["score_audit"] = {"calls": audit["calls"], "scores": audit["scores"], "exact": True}
                    scores.clear()
                metrics = json.loads((run / "fused_depth_slice_metrics.json").read_text())
                fields = (
                    "elapsed_sec", "mean_holo_sec_per_frame", "mean_depth_sec_per_frame", "mean_slice_infer_sec_per_frame",
                    "rows", "frames", "depth_crops_scored", "depth_router_fallback_count", "unique_frame_slices",
                    "depth_cuda_graph_captures", "depth_cuda_graph_replays", "depth_cuda_graph_capture_sec",
                    "depth_cuda_graph_fallbacks",
                )
                item = {name: metrics[name] for name in fields}
                item.update(mode=mode, repeat=repeat, audited=repeat == 0,
                            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device))
                summary["results"].append(item)
                print(json.dumps(item), flush=True)
            for filename in ("particles.csv", "tracks_with_diam_depth.csv", "tracks_with_estdepth_slice_diameter.csv"):
                left, right = output / f"eager-{repeat}" / filename, output / f"graph-{repeat}" / filename
                pd.testing.assert_frame_equal(pd.read_csv(left), pd.read_csv(right), check_exact=True)
        measured = [item for item in summary["results"] if not item["audited"]]
        medians = {
            mode: statistics.median(item["elapsed_sec"] for item in measured if item["mode"] == mode)
            for mode in ("eager", "graph")
        }
        summary.update(all_csv_values_equal=True, median_core_seconds=medians,
                       core_speedup=medians["eager"] / medians["graph"],
                       scope="All configured depth planes, same compiled engines; core time includes per-frame graph capture, excludes engine compilation, detection and subprocess startup")
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"core_speedup": summary["core_speedup"], "score_audit": summary["score_audit"]}), flush=True)


if __name__ == "__main__":
    main()
