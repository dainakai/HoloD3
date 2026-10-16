"""Audit v0.2.2 against an unmodified Git baseline with shared TensorRT engines.

All reconstruction planes and inference batches are retained. Engine compilation
is outside the core timer; per-run IFFT plan preflight and graph capture are in
it. Run on an idle GPU host, one campaign at a time, with a fixed input ROI CSV.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from holod3.config import repository_root
from scripts.validate_cuda_optimizations import memoize, memoize_optimizer
from src.common.cuda_fft import ReconstructionIFFT
from src.common.cuda_graph import CudaGraphScorer
from src.detection import depth_and_slice_fused as core

CSV_NAMES = ("particles.csv", "tracks_with_diam_depth.csv", "tracks_with_estdepth_slice_diameter.csv")


def csv_diff(expected, actual):
    assert list(actual.columns) == list(expected.columns) and len(actual) == len(expected)
    differences = {}
    for col in expected:
        a, b = actual[col], expected[col]
        mask = ~((a == b) | (a.isna() & b.isna()))
        if mask.any():
            item = {"different_rows": int(mask.sum())}
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                delta = a.to_numpy(float) - b.to_numpy(float)
                assert np.isfinite(delta).all(), col
                item.update(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta**2))))
            differences[col] = item
    return {"rows": len(actual), "exact": not differences, "differences": differences}


def check_idle():
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True,
    )
    others = [line for line in listing.splitlines() if line and int(line.split(",", 1)[0]) != os.getpid()]
    if others:
        raise RuntimeError("Other GPU workloads appeared; discard overlapping timings: " + "; ".join(others))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline-ref", default="v0.2.1")
    parser.add_argument("--acquisition", type=Path, default=repository_root()/"data/demo/experimental/acquisition.yaml")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three measured repetitions")
    check_idle()
    root = repository_root()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "sources").mkdir()
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:src/detection/depth_and_slice_fused.py"], cwd=root, text=True,
    )
    baseline_path = out / "sources/reference_core.py"
    baseline_path.write_text(baseline_source)
    baseline = types.ModuleType("v021_reference")
    # Preserve asset-root discovery while executing the saved, unmodified source.
    baseline.__file__ = core.__file__
    exec(compile(baseline_source, str(baseline_path), "exec"), vars(baseline))
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    results = {"baseline_ref": args.baseline_ref,
               "baseline_commit": subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=root, text=True).strip(),
               "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
               "torch_tensorrt": importlib.metadata.version("torch_tensorrt"),
               "tensorrt": importlib.metadata.version("tensorrt"), "source_sha256": {},
               "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "acquisition_sha256": hashlib.sha256(args.acquisition.read_bytes()).hexdigest(),
               "audit": {}, "runs": [], "median_core_seconds": {},
               "scope": "Six-frame core with production I/O; includes FFT plan preflight and graph capture; excludes engine builds, detector, MinIP and process startup"}
    sources = [Path(__file__), Path(core.__file__), root/"src/detection/depth_runtime.py", root/"src/common/cuda_fft.py"]
    for source in sources:
        data = source.read_bytes()
        results["source_sha256"][source.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
        (out/"sources"/source.name).write_bytes(data)
    results["checkpoint_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"models/production").glob("*.pt")}
    state = {"audit": False, "variant": "baseline", "scores": 0, "crop_calls": 0, "fft_checks": 0}
    reference_scores, reference_crops = [], []
    original_score, original_predict, original_fft = CudaGraphScorer.__call__, core.predict_diameter_batch, ReconstructionIFFT.__call__

    def score(self, crops, diameters):
        values = original_score(self, crops, diameters)
        if state["audit"]:
            if state["variant"] == "baseline":
                reference_scores.append(values.cpu())
            else:
                assert torch.equal(values, reference_scores[state["scores"]].to(device)), "Depth scores changed"
            state["scores"] += 1
        return values

    def diameter(model, crops, *a, **kw):
        if state["audit"]:
            if state["variant"] == "baseline":
                reference_crops.append(crops.cpu())
            else:
                expected = reference_crops[state["crop_calls"]].to(device)
                delta = (crops - expected).double()
                quantized = ((crops.clamp(0, 1)*255).round() - (expected.clamp(0, 1)*255).round()).abs()
                assert torch.isfinite(delta).all()
                state["crop_max"] = max(state["crop_max"], float(delta.abs().max()))
                state["crop_squared"] += float(delta.square().sum())
                state["crop_values"] += delta.numel()
                state["quantized_different"] += int(quantized.count_nonzero())
                state["quantized_max"] = max(state["quantized_max"], float(quantized.max()))
            state["crop_calls"] += 1
        return original_predict(model, crops, *a, **kw)

    def inverse(self, frequency):
        actual = original_fft(self, frequency)
        if state["audit"] and self.active:
            assert torch.equal(actual, torch.fft.ifft2(frequency)), "Normalized IFFT changed values"
            state["fft_checks"] += 1
        return actual

    def save():
        (out/"results.json").write_text(json.dumps(results, indent=2)+"\n")

    variants = {"baseline": (False, False, False), "disabled": (False, False, False),
                "inputs_store": (False, True, True), "reuse": (True, False, False), "v022": (True, True, True)}
    with contextlib.ExitStack() as stack:
        for name in ("load_depth_model", "load_slice_model", "optimize_model_backend", "optimize_slice_model_backend"):
            original = getattr(core, name)
            cached = memoize_optimizer(original) if name.startswith("optimize") else memoize(original)
            stack.enter_context(patch.object(core, name, cached))
            stack.enter_context(patch.object(baseline, name, cached))
        stack.enter_context(patch.object(CudaGraphScorer, "__call__", score))
        stack.enter_context(patch.object(ReconstructionIFFT, "__call__", inverse))
        for module in (core, baseline):
            stack.enter_context(patch.object(module, "predict_diameter_batch", diameter))
        import torch_tensorrt
        compile_model = torch_tensorrt.compile

        def compile_local(*a, **kw):
            kw["timing_cache_path"] = str(out/"timing.bin")
            kw["engine_cache_dir"] = str(out/"engines")
            return compile_model(*a, **kw)

        stack.enter_context(patch.object(torch_tensorrt, "compile", compile_local))

        def run(variant, repeat, audit=False):
            check_idle()
            state.update(variant=variant, audit=audit, scores=0, crop_calls=0, fft_checks=0,
                         crop_max=0., crop_squared=0., crop_values=0, quantized_different=0, quantized_max=0.)
            path = out/f"{variant}-{repeat}"
            path.mkdir()
            argv = [str(core.__file__), "--run-dir", str(path), "--input", str(args.input.resolve()),
                    "--acquisition-config", str(args.acquisition.resolve()), "--frames", str(args.limit),
                    "--device", str(device), "--model-backend", "tensorrt", "--diam-model-backend", "tensorrt",
                    "--depth-cuda-graphs", "--prefetch-holo-workers", "4", "--prefetch-holo-frames", "16",
                    "--stream-csv-writes", "--final-output", str(path/"particles.csv")]
            if variant != "baseline":
                for name, enabled in zip(("reuse-depth-crops", "depth-input-reuse", "fuse-ifft-normalization"), variants[variant], strict=True):
                    argv.append(f"--{'' if enabled else 'no-'}{name}")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            with patch.object(sys, "argv", argv), (path/"core.log").open("w") as log, contextlib.redirect_stdout(log):
                (baseline if variant == "baseline" else core).main()
            torch.cuda.synchronize(device)
            check_idle()
            metrics = json.loads((path/"fused_depth_slice_metrics.json").read_text())
            assert all(metrics[n] == "tensorrt" for n in ("depth_model_backend", "depth_fallback_model_backend", "slice_model_backend"))
            assert metrics["depth_cuda_graph_fallbacks"] == 0
            if variant != "baseline":
                for n, wanted in zip(("reuse_depth_crops", "depth_input_reuse", "fuse_ifft_normalization"), variants[variant], strict=True):
                    assert metrics[n] == wanted, (variant, n, metrics[n])
            csv_checks = {n: csv_diff(pd.read_csv(out/"baseline-audit"/n), pd.read_csv(path/n)) for n in CSV_NAMES}
            assert csv_checks["tracks_with_diam_depth.csv"]["exact"]
            if variant in ("baseline", "disabled", "inputs_store"):
                assert all(c["exact"] for c in csv_checks.values())
            if variant == "v022":
                for n in CSV_NAMES:
                    pd.testing.assert_frame_equal(pd.read_csv(out/"reuse-audit"/n), pd.read_csv(path/n), check_exact=True)
            if audit:
                if variant != "baseline":
                    assert state["scores"] == len(reference_scores) and state["crop_calls"] == len(reference_crops)
                results["audit"][variant] = {k: v for k, v in state.items() if k not in ("variant", "audit")}
                results["audit"][variant]["csv"] = csv_checks
                if state["crop_values"]:
                    results["audit"][variant]["crop_rmse"] = (state["crop_squared"]/state["crop_values"])**0.5
                if variant in ("disabled", "inputs_store"):
                    assert state["crop_max"] == 0
                else:
                    assert state["crop_max"] < 1e-5 and state["quantized_max"] <= 1
            else:
                for n in CSV_NAMES:
                    pd.testing.assert_frame_equal(pd.read_csv(out/f"{variant}-audit"/n), pd.read_csv(path/n), check_exact=True)
            fields = ("elapsed_sec", "frames", "rows", "depth_crops_scored", "unique_frame_slices",
                      "mean_holo_sec_per_frame", "mean_depth_sec_per_frame", "mean_slice_infer_sec_per_frame",
                      "depth_cuda_graph_captures", "depth_cuda_graph_replays", "depth_cuda_graph_capture_sec",
                      "depth_cuda_graph_fallbacks", "ifft_plan_setup_sec", "fuse_ifft_normalization")
            entry = {k: metrics[k] for k in fields if k in metrics}
            entry.update(variant=variant, repeat=repeat, audit=audit,
                         peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device))
            results["runs"].append(entry)
            if variant == "baseline" and audit:
                results["compile_seconds"] = {k: v for k, v in metrics.items() if "compile_sec" in k}
            save()
            print(json.dumps(entry), flush=True)

        for v in variants:
            run(v, "audit", audit=True)
        reference_scores.clear()
        reference_crops.clear()
        timed = [v for v in variants if v != "disabled"]
        for repeat in range(args.repeats):
            order = timed[repeat % len(timed):]+timed[:repeat % len(timed)]
            if repeat % 2:
                order.reverse()
            for v in order:
                run(v, repeat)
        results["median_core_seconds"] = {v: statistics.median(r["elapsed_sec"] for r in results["runs"]
            if not r["audit"] and r["variant"] == v) for v in timed}
        save()
        print(json.dumps(results["median_core_seconds"]), flush=True)


if __name__ == "__main__":
    main()
