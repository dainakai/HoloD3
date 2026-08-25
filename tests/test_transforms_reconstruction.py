from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from holod3.acquisition import AcquisitionConfig, TransformStep
from holod3.reconstruction import (
    build_propagation_setup,
    gabor_wavefront,
    phase_retrieval_wavefront,
    prepare_minip_images,
    quadratic_distortion_correction,
)
from holod3.transforms import apply_transforms, load_transformed_image, normalize_grayscale


def test_integer_normalization_and_built_in_transform_order(tmp_path: Path) -> None:
    source = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), source)
    value = load_transformed_image(
        image_path,
        [TransformStep("invert"), TransformStep("flip_horizontal")],
        base_dir=tmp_path,
    )
    expected = np.fliplr(1.0 - source.astype(np.float32) / 255.0)
    np.testing.assert_allclose(value, expected, atol=1e-7)
    assert value.dtype == np.float32 and value.flags.c_contiguous


def test_file_transform_receives_kwargs_and_is_relative_to_yaml(tmp_path: Path) -> None:
    module = tmp_path / "instrument.py"
    module.write_text(
        "import numpy as np\n"
        "def shift(image, *, amount):\n"
        "    return np.clip(image + amount, 0.0, 1.0).astype(np.float32)\n",
        encoding="utf-8",
    )
    image = np.full((3, 3), 0.25, dtype=np.float32)
    result = apply_transforms(
        image,
        [TransformStep("instrument.py:shift", {"amount": 0.2})],
        base_dir=tmp_path,
    )
    np.testing.assert_allclose(result, 0.45)


def test_transform_output_contract_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_grayscale(np.asarray([[np.nan]], dtype=np.float32))


def test_gabor_wavefront_is_zero_phase_square_root_amplitude() -> None:
    intensity = torch.tensor([[0.0, 0.25], [1.0, 0.81]], dtype=torch.float32)
    wavefront = gabor_wavefront(intensity)
    torch.testing.assert_close(wavefront.real, torch.sqrt(intensity))
    torch.testing.assert_close(wavefront.imag, torch.zeros_like(intensity))
    assert wavefront.dtype == torch.complex64


def test_zero_iteration_phase_retrieval_preserves_primary_amplitude() -> None:
    primary = torch.tensor([[0.04, 0.25], [0.64, 1.0]], dtype=torch.float32)
    secondary = torch.full_like(primary, 0.3)
    transfer = torch.ones_like(primary, dtype=torch.complex64)
    wavefront = phase_retrieval_wavefront(primary, secondary, transfer, transfer, iterations=0)
    torch.testing.assert_close(wavefront.real, torch.sqrt(primary))
    torch.testing.assert_close(wavefront.imag, torch.zeros_like(primary))


def test_identity_quadratic_distortion_coefficients() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    coefficients = np.zeros(12, dtype=np.float64)
    coefficients[1] = 1.0  # x = one-based column j
    coefficients[8] = 1.0  # y = one-based row i
    np.testing.assert_array_equal(quadratic_distortion_correction(image, coefficients), image)


def test_single_gabor_minip_generation_runs_on_cpu(tmp_path: Path) -> None:
    hologram_dir = tmp_path / "holograms"
    hologram_dir.mkdir()
    source = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
    assert cv2.imwrite(str(hologram_dir / "frame-a.png"), source)
    mapping = {
        "schema_version": 1,
        "name": "tiny-gabor",
        "description": "CPU reconstruction fixture",
        "mode": "single_gabor",
        "frames": {"primary_holograms": "holograms", "minip": None},
        "optics": {
            "wavelength_um": 0.63,
            "pixel_pitch_um": 10.0,
            "image_size_px": 8,
            "reconstruction_start_um": 100.0,
            "slice_spacing_um": 10.0,
            "slice_count": 3,
            "phase_retrieval_distance_um": None,
        },
        "reconstruction": {
            "phase_retrieval_iterations": 0,
            "fft_padding_side": 12,
            "minip_slice_step": 1,
        },
        "calibration": {"secondary_distortion_coefficients": None},
        "transforms": {"primary": [], "minip": []},
    }
    config_path = tmp_path / "acquisition.yaml"
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    config = AcquisitionConfig.load(config_path)
    setup = build_propagation_setup(config, torch.device("cpu"))
    assert setup.phase_forward is None and setup.phase_inverse is None
    output = tmp_path / "prepared"
    summary = prepare_minip_images(config, config.frame_records(), output, device="cpu")
    image = cv2.imread(str(output / "frame-a.png"), cv2.IMREAD_GRAYSCALE)
    assert summary["source"] == "reconstructed"
    assert summary["frames"] == 1
    assert image is not None and image.shape == (8, 8)
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        prepare_minip_images(config, config.frame_records(), output, device="cpu")
    replaced = prepare_minip_images(config, config.frame_records(), output, device="cpu", overwrite=True)
    assert replaced["frames"] == 1
