from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.detection import depth_and_slice_fused as core
from src.detection.depth_runtime import retain_winning_crops


@pytest.mark.parametrize("recenter,frame_batch", [(False, True), (True, True), (False, False)])
def test_frame_loop_retains_routed_crops_and_preserves_unsupported_paths(tmp_path: Path, monkeypatch, recenter, frame_batch):
    image_dir = tmp_path / "holograms"
    image_dir.mkdir()
    for i in range(2):
        image = (np.arange(64).reshape(8, 8) + i * 20).astype(np.uint8)
        assert cv2.imwrite(str(image_dir / f"frame{i}.png"), image)
    acquisition = tmp_path / "acquisition.yaml"
    acquisition.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "reuse-fixture", "description": "two frames", "mode": "single_gabor",
        "frames": {"primary_holograms": "holograms", "minip": None},
        "optics": {"wavelength_um": 0.63, "pixel_pitch_um": 8., "image_size_px": 8,
                   "reconstruction_start_um": 200., "slice_spacing_um": 20., "slice_count": 10,
                   "phase_retrieval_distance_um": None},
        "reconstruction": {"phase_retrieval_iterations": 0, "fft_padding_side": 12, "minip_slice_step": 1},
        "calibration": {"secondary_distortion_coefficients": None}, "transforms": {"primary": [], "minip": []},
    }))
    source = tmp_path / "input.csv"
    rows = []
    for frame in range(2):
        for i, diameter in enumerate((25., 100., 50.)):
            rows.append(dict(frame=frame, file=f"frame{frame}.png", seg_xc=2+i, seg_yc=3+i,
                             diameter_um=diameter, diameter_method="fallback" if i != 1 else "contrast_area",
                             x1=i, y1=i, x2=i+4, y2=i+4, w=4, h=4))
    pd.DataFrame(rows).to_csv(source, index=False)

    class Model(torch.nn.Module):
        def __init__(self, fallback):
            super().__init__()
            self.fallback = fallback

        def forward(self, image, cond):
            # Fallback reverses score ordering; one particle always ties.
            values = image[:, 0].mean((1, 2)) * (-1 if self.fallback else 1)
            return torch.where(cond[:, 0] < -0.9, torch.zeros_like(values), values)

    def load_depth(path, *_):
        return Model("fallback" in str(path)), {
            "model_backend": "torch", "input_mode": "raw_diam_scalar", "mask_radius_diam_scale": 2.,
            "mask_min_radius_px": 1., "mask_softness_px": 3.,
        }

    monkeypatch.setattr(core, "load_depth_model", load_depth)
    monkeypatch.setattr(core, "optimize_model_backend", lambda model, *a: (model, 0.))
    monkeypatch.setattr(core, "load_slice_model", lambda *a: (torch.nn.Identity(), {}, 1., {}))
    monkeypatch.setattr(core, "optimize_slice_model_backend", lambda model, **kw: (
        model, {"model_backend": "torch", "inference_precision": "fp32", "trt_dynamic_max_batch": 2}, 0.,
    ))
    kept, diameter_inputs, inverse_counts = [], [], []

    def retain(*a):
        result = retain_winning_crops(*a)
        kept.append(result.clone())
        return result

    def predict(model, crops, *a):
        diameter_inputs.append(crops.clone())
        pred = crops.mean((1, 2)).numpy() * 100
        return pred, pred / 10, pred / 100, pred / 1000

    original_ifft = core.ReconstructionIFFT.__call__

    def inverse(self, frequency):
        inverse_counts.append(1)
        return original_ifft(self, frequency)

    monkeypatch.setattr(core, "retain_winning_crops", retain)
    monkeypatch.setattr(core, "predict_diameter_batch", predict)
    monkeypatch.setattr(core.ReconstructionIFFT, "__call__", inverse)
    outputs = []
    for reuse in (False, True):
        run = tmp_path / str(reuse)
        argv = [str(core.__file__), "--run-dir", str(run), "--input", str(source),
                "--acquisition-config", str(acquisition), "--device", "cpu", "--model-backend", "torch",
                "--diam-model-backend", "torch", "--crop-size", "4", "--slice-start", "2", "--slice-step", "3",
                "--slice-end", "8", "--slice-block", "2", "--depth-batch-size", "2", "--diam-batch-size", "2",
                "--no-depth-cuda-graphs", "--no-depth-input-reuse", "--no-fuse-ifft-normalization",
                "--reuse-depth-crops" if reuse else "--no-reuse-depth-crops",
                "--recenter-on-slice" if recenter else "--no-recenter-on-slice",
                "--slice-diam-frame-batch" if frame_batch else "--no-slice-diam-frame-batch"]
        kept.clear()
        diameter_inputs.clear()
        inverse_counts.clear()
        with patch.object(sys, "argv", argv):
            core.main()
        metrics = json.loads((run / "fused_depth_slice_metrics.json").read_text())
        depth = pd.read_csv(run / "tracks_with_diam_depth.csv")
        outputs.append((depth, len(inverse_counts), torch.cat(diameter_inputs)))
        active = reuse and not recenter and frame_batch
        assert metrics["reuse_depth_crops"] == active
        assert metrics["slice_crops_scored"] == 6
        if active:
            # Three depths (2,5,8), two blocks per frame. Stable sort must retain
            # the original diameter batch/particle order after routing and ties.
            for frame, best_crops in enumerate((kept[1], kept[3])):
                order = depth.iloc[frame*3:(frame+1)*3]["slice"].to_numpy().argsort(kind="stable")
                assert torch.equal(torch.cat(diameter_inputs)[frame*3:(frame+1)*3], best_crops[order])
            assert len(inverse_counts) == 6
        else:
            assert not kept
    pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0], check_exact=True)
    if recenter or not frame_batch:
        assert outputs[0][1] == outputs[1][1]
        assert torch.equal(outputs[0][2], outputs[1][2])
    else:
        assert outputs[0][1] > outputs[1][1]
