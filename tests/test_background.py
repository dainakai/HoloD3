from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from holod3.acquisition import AcquisitionConfig, BackgroundRemovalConfig
from holod3.background import prepare_background_holograms
from holod3.reconstruction import build_propagation_setup, load_frame_arrays, prepare_minip_images
from holod3.transforms import apply_transforms
from src.preprocessing.backrem_adaptive import AdaptiveConfig, correct_with_backgrounds, load_gray
from src.preprocessing.backrem_streaming import (
    estimate_anchor_backgrounds_cuda_to_disk,
    estimate_anchor_backgrounds_rolling_cuda_to_disk,
    estimate_anchor_backgrounds_to_disk,
    load_anchor,
    resolve_median_backend,
)


def sequence_acquisition(tmp_path: Path, *, dual: bool = False) -> AcquisitionConfig:
    rows, cols = np.indices((24, 24))
    pattern = (80 + rows + 2 * cols).astype(np.int16)
    for role in (("primary", "secondary") if dual else ("primary",)):
        folder = tmp_path / role
        folder.mkdir(parents=True)
        for frame in range(7):
            value = pattern.copy() + frame * 3 + (20 if role == "secondary" else 0)
            value[4 + frame : 6 + frame, 8:10] -= 40
            assert cv2.imwrite(str(folder / f"frame-{frame + 1}.bmp"), value.astype(np.uint8))
    mapping = {
        "schema_version": 1,
        "name": "temporal-fixture",
        "description": "Moving dark object on a fixed pattern with brightness drift.",
        "mode": "dual_phase_retrieval" if dual else "single_gabor",
        "frames": {"primary_holograms": "primary", "secondary_holograms": "secondary" if dual else None},
        "optics": {
            "wavelength_um": 0.63,
            "pixel_pitch_um": 10.0,
            "image_size_px": 24,
            "reconstruction_start_um": 100.0,
            "slice_spacing_um": 25.0,
            "slice_count": 3,
            "phase_retrieval_distance_um": 500.0 if dual else None,
        },
        "reconstruction": {"phase_retrieval_iterations": 1, "fft_padding_side": 32, "minip_slice_step": 1},
        "background_removal": {"enabled": True, "window": 5, "anchor_stride": 2, "batch_size": 2, "save_workers": 2},
    }
    path = tmp_path / "acquisition.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return AcquisitionConfig.load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("enabled", "true"), ("window", 0), ("window", 4), ("anchor_stride", -1), ("batch_size", True),
     ("lowpass", -1), ("save_workers", 1.5), ("target_level", float("nan")), ("target_level", 256),
     ("median_backend", "unknown")],
)
def test_invalid_background_settings_fail_early(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="background_removal"):
        BackgroundRemovalConfig(**{field: value}).validate()


def test_background_schema_is_opt_in_and_rejects_stale_minip(tmp_path: Path) -> None:
    config = sequence_acquisition(tmp_path)
    mapping = config.to_dict()
    mapping.pop("background_removal")
    assert not AcquisitionConfig.from_mapping(mapping).background_removal.enabled
    mapping["background_removal"] = {"enabled": True, "typo": 1}
    with pytest.raises(ValueError, match="Unknown background_removal"):
        AcquisitionConfig.from_mapping(mapping)
    mapping["background_removal"] = {"enabled": True}
    mapping["frames"]["minip"] = "minip"
    with pytest.raises(ValueError, match="frames.minip: null"):
        AcquisitionConfig.from_mapping(mapping)


def test_temporal_median_uses_lower_middle_value_at_sequence_edges(tmp_path: Path) -> None:
    paths = []
    for i, value in enumerate((10, 40, 90, 150)):
        path = tmp_path / f"{i}.png"
        assert cv2.imwrite(str(path), np.full((16, 16), value, np.uint8))
        paths.append(path)
    anchors, means, _ = estimate_anchor_backgrounds_to_disk(
        paths, AdaptiveConfig(window=3, anchor_stride=2), torch.device("cpu"), tmp_path / "anchors",
    )
    assert anchors == [0, 2, 3]
    assert means == [10.0, 90.0, 90.0]
    np.testing.assert_array_equal(load_anchor(tmp_path / "anchors", 0, 0), np.full((16, 16), 10))


def test_subtracts_fixed_pattern_and_preserves_moving_dark_object() -> None:
    rows, cols = np.indices((24, 24))
    background = (70 + rows * 2 + cols).astype(np.float32)
    raw = np.stack([background + 7, background - 11]).astype(np.uint8)
    raw[0, 9:12, 9:12] -= 40
    raw[1, 14:17, 14:17] -= 40
    expected = np.full_like(raw, 120)
    expected[0, 9:12, 9:12] = 80
    expected[1, 14:17, 14:17] = 80
    corrected, _, target = correct_with_backgrounds(
        raw, np.repeat(background[None], 2, axis=0), AdaptiveConfig(target_level=120), torch.device("cpu"),
    )
    np.testing.assert_array_equal(corrected, expected)
    assert target == 120


def test_row_and_column_profiles_remove_smooth_stripes() -> None:
    rows, cols = np.indices((64, 64))
    stripes = 8 * np.sin(rows * 2 * np.pi / 64) + 5 * np.cos(cols * 2 * np.pi / 64)
    raw = np.rint(120 + stripes).astype(np.uint8)[None]
    background = np.full_like(raw, 120, dtype=np.float16)
    result, _, _ = correct_with_backgrounds(raw, background, AdaptiveConfig(target_level=120), torch.device("cpu"))
    assert float(result.std()) < 0.3 * float(raw.std())
    assert abs(float(np.median(result)) - 120) <= 1


def test_selected_frame_uses_full_sequence_and_matches_full_run(tmp_path: Path) -> None:
    config = sequence_acquisition(tmp_path / "input")
    all_records = config.frame_records()
    _, full, full_summary = prepare_background_holograms(config, all_records, tmp_path / "full", device="cpu")
    selected = config.selected_records(limit=1, start_index=3)
    prepared, one, summary = prepare_background_holograms(config, selected, tmp_path / "selected", device="cpu")
    assert summary["source_frames"] == 7 and summary["selected_frames"] == 1
    assert summary["cameras"]["primary"]["target_level"] == full_summary["cameras"]["primary"]["target_level"]
    assert not prepared.background_removal.enabled
    assert [record.stem for record in one] == ["frame-4"]
    np.testing.assert_array_equal(load_gray(one[0].primary), load_gray(full[3].primary))
    assert len(list(prepared.primary_dir.glob("*.png"))) == 1
    assert (tmp_path / "selected/background_summary.json").is_file()
    assert np.unique(load_gray(one[0].primary)).size > 1
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_background_holograms(config, selected, tmp_path / "selected", device="cpu")
    _, overwritten, _ = prepare_background_holograms(config, selected, tmp_path / "full", device="cpu", overwrite=True)
    assert len(overwritten) == 1
    assert sorted(path.stem for path in (tmp_path / "full/primary").glob("*.png")) == ["frame-4"]


def test_dual_preparation_preserves_file_transforms_and_calibration(tmp_path: Path) -> None:
    config = sequence_acquisition(tmp_path / "input", dual=True)
    (config.base_dir / "instrument.py").write_text(
        "import numpy as np\ndef adjust(image, *, amount):\n"
        "    return np.ascontiguousarray(np.clip(image + amount, 0, 1)[:, ::-1])\n",
        encoding="utf-8",
    )
    coefficients = np.zeros(12)
    coefficients[1] = coefficients[8] = 1
    np.savetxt(config.base_dir / "calibration.txt", coefficients)
    mapping = config.to_dict()
    mapping["calibration"]["secondary_distortion_coefficients"] = "calibration.txt"
    mapping["transforms"] = {
        "primary": [{"function": "instrument.py:adjust", "kwargs": {"amount": 0.03}}],
        "secondary": [{"function": "flip_vertical"}],
    }
    config = AcquisitionConfig.from_mapping(mapping, source_path=config.source_path)
    records = config.selected_records(limit=2)
    prepared, output, summary = prepare_background_holograms(config, records, tmp_path / "prepared", device="cpu")
    assert set(summary["cameras"]) == {"primary", "secondary"}
    assert prepared.distortion_coefficients_path == config.distortion_coefficients_path
    setup = build_propagation_setup(prepared, torch.device("cpu"))
    primary, secondary = load_frame_arrays(prepared, output[0], setup)
    for role, image in (("primary", primary), ("secondary", secondary)):
        corrected = load_gray(getattr(output[0], role)).astype(np.float32) / 255
        expected = apply_transforms(corrected, config.transform_steps(role), base_dir=config.base_dir)
        np.testing.assert_allclose(image, expected)
    assert config.background_removal.enabled  # The user's acquisition is unchanged.


def test_failed_replacement_keeps_existing_corrected_images(tmp_path: Path) -> None:
    config = sequence_acquisition(tmp_path / "input", dual=True)
    records = config.selected_records(limit=1)
    destination = tmp_path / "prepared"
    _, output, _ = prepare_background_holograms(config, records, destination, device="cpu")
    original = output[0].primary.read_bytes()
    bad = config.secondary_dir / "frame-2.bmp"
    bad.write_bytes(b"invalid image")
    with pytest.raises((OSError, ValueError)):
        prepare_background_holograms(config, records, destination, device="cpu", overwrite=True)
    assert output[0].primary.read_bytes() == original
    assert not list(tmp_path.glob(".background-*"))


def test_background_disabled_does_no_io_and_non_uint8_input_is_rejected(tmp_path: Path) -> None:
    config = sequence_acquisition(tmp_path / "input")
    disabled = replace(config, background_removal=BackgroundRemovalConfig())
    records = disabled.frame_records()
    result = prepare_background_holograms(disabled, records, tmp_path / "unused", device="cpu")
    assert result == (disabled, records, {"enabled": False})
    assert not (tmp_path / "unused").exists()
    path = tmp_path / "sixteen-bit.png"
    assert cv2.imwrite(str(path), np.full((16, 16), 4096, np.uint16))
    with pytest.raises(ValueError, match="8-bit"):
        load_gray(path)
    original_path = config.primary_dir / "frame-1.bmp"
    original_path.unlink()
    assert cv2.imwrite(str(config.primary_dir / "frame-1.png"), np.full((24, 24), 4096, np.uint16))
    with pytest.raises(ValueError, match="8-bit"):
        config.validate_image_contracts()


def test_minip_entry_point_applies_backgrounds_and_clears_stale_frames(tmp_path: Path) -> None:
    config = sequence_acquisition(tmp_path / "input")
    output = tmp_path / "minip"
    summary = prepare_minip_images(config, config.selected_records(limit=2), output, device="cpu")
    assert summary["background_removal"]["enabled"]
    prepared = AcquisitionConfig.load(output / "_background/acquisition.yaml")
    expected_dir = tmp_path / "expected"
    prepare_minip_images(prepared, prepared.frame_records(), expected_dir, device="cpu")
    for path in expected_dir.glob("*.png"):
        np.testing.assert_array_equal(load_gray(path), load_gray(output / path.name))
    prepare_minip_images(config, config.selected_records(limit=1), output, device="cpu", overwrite=True)
    assert sorted(path.name for path in output.glob("*.png")) == ["frame-1.png"]


def test_auto_backend_works_without_optional_cuda_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_median_backend("auto", torch.device("cpu"), 129) == "torch"
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    assert resolve_median_backend("auto", torch.device("cuda:0"), 129) == "torch"
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_median_backend("cuda-rolling", torch.device("cpu"), 129)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA median equivalence requires a GPU")
@pytest.mark.parametrize(("window", "stride"), [(5, 2), (3, 8), (257, 2)])
def test_cuda_background_medians_match_torch(tmp_path: Path, window: int, stride: int) -> None:
    pytest.importorskip("cuda.bindings.nvrtc")
    config = sequence_acquisition(tmp_path / "input")
    paths = [record.primary for record in config.frame_records()]
    if window > 255:
        # Exercise the wider histogram counter with more than 255 equal values.
        paths = [paths[0]] * 259
    algorithm = AdaptiveConfig(window=window, anchor_stride=stride)
    device = torch.device("cuda:0")
    anchors, means, _ = estimate_anchor_backgrounds_to_disk(paths, algorithm, device, tmp_path / "torch")
    methods = [("window", estimate_anchor_backgrounds_cuda_to_disk)]
    if window <= 255:
        methods.append(("rolling", estimate_anchor_backgrounds_rolling_cuda_to_disk))
    for name, method in methods:
        actual_anchors, actual_means, _ = method(paths, algorithm, device, tmp_path / name)
        assert actual_anchors == anchors and actual_means == means
        for i, pos in enumerate(anchors):
            np.testing.assert_array_equal(load_anchor(tmp_path / name, i, pos), load_anchor(tmp_path / "torch", i, pos))
