from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from holod3.acquisition import AcquisitionConfig


def write_image(path: Path, value: int = 127, *, side: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((side, side), value, dtype=np.uint8))


def dual_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "fixture",
        "description": "dual fixture",
        "mode": "dual_phase_retrieval",
        "frames": {
            "primary_holograms": "holograms/primary",
            "secondary_holograms": "holograms/secondary",
            "minip": "minip",
        },
        "optics": {
            "wavelength_um": 0.63,
            "pixel_pitch_um": 10.0,
            "image_size_px": 8,
            "reconstruction_start_um": 1000.0,
            "slice_spacing_um": 25.0,
            "slice_count": 4,
            "phase_retrieval_distance_um": 500.0,
        },
        "reconstruction": {
            "phase_retrieval_iterations": 2,
            "fft_padding_side": 12,
            "minip_slice_step": 1,
        },
        "calibration": {"secondary_distortion_coefficients": None},
        "transforms": {"primary": [], "secondary": [], "minip": []},
    }


def materialize(tmp_path: Path, mapping: dict[str, object], stems: tuple[str, ...] = ("10", "2")) -> Path:
    for stem in stems:
        write_image(tmp_path / "holograms/primary" / f"{stem}.png")
        if mapping["mode"] == "dual_phase_retrieval":
            write_image(tmp_path / "holograms/secondary" / f"{stem}.tif")
        frames = mapping["frames"]
        assert isinstance(frames, dict)
        if frames.get("minip") is not None:
            write_image(tmp_path / "minip" / f"{stem}.jpg")
    config_path = tmp_path / "acquisition.yaml"
    config_path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return config_path


def test_dual_acquisition_resolves_relative_paths_and_natural_order(tmp_path: Path) -> None:
    config = AcquisitionConfig.load(materialize(tmp_path, dual_mapping()))
    records = config.frame_records()
    assert [record.stem for record in records] == ["2", "10"]
    assert all(record.secondary is not None and record.minip is not None for record in records)
    assert config.primary_dir == (tmp_path / "holograms/primary").resolve()
    assert config.to_dict()["optics"]["slice_spacing_um"] == 25.0


def test_frame_stem_mismatch_is_rejected(tmp_path: Path) -> None:
    path = materialize(tmp_path, dual_mapping(), stems=("1", "2"))
    (tmp_path / "holograms/secondary/2.tif").unlink()
    write_image(tmp_path / "holograms/secondary/3.tif")
    with pytest.raises(ValueError, match="stems do not match"):
        AcquisitionConfig.load(path).frame_records()


def test_selection_limit_zero_means_all_and_end_is_inclusive(tmp_path: Path) -> None:
    config = AcquisitionConfig.load(materialize(tmp_path, dual_mapping(), stems=("1", "2", "3", "4")))
    assert [record.stem for record in config.selected_records(limit=0, start_index=1, end_index=2)] == ["2", "3"]
    assert [record.stem for record in config.selected_records(limit=1)] == ["1"]
    with pytest.raises(ValueError, match="limit"):
        config.selected_records(limit=-1)


def test_single_gabor_requires_only_one_hologram_directory(tmp_path: Path) -> None:
    mapping = dual_mapping()
    mapping["mode"] = "single_gabor"
    frames = mapping["frames"]
    optics = mapping["optics"]
    assert isinstance(frames, dict) and isinstance(optics, dict)
    frames.pop("secondary_holograms")
    frames["minip"] = None
    optics["phase_retrieval_distance_um"] = None
    config = AcquisitionConfig.load(materialize(tmp_path, mapping, stems=("1",)))
    record = config.frame_records()[0]
    assert record.secondary is None and record.minip is None


def test_single_gabor_rejects_secondary_directory() -> None:
    mapping = dual_mapping()
    mapping["mode"] = "single_gabor"
    optics = mapping["optics"]
    assert isinstance(optics, dict)
    optics["phase_retrieval_distance_um"] = None
    with pytest.raises(ValueError, match="accepts one hologram directory"):
        AcquisitionConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("wavelength_um", 0, "must be positive"),
        ("wavelength_um", float("nan"), "must be positive"),
        ("reconstruction_start_um", float("inf"), "must be finite"),
        ("slice_count", 0, "must be positive"),
        ("slice_count", 3.5, "must be an integer"),
        ("image_size_px", 9, "fft padding minus image size must be even"),
    ],
)
def test_invalid_optical_values_are_rejected(field: str, value: object, message: str) -> None:
    mapping = dual_mapping()
    optics = mapping["optics"]
    assert isinstance(optics, dict)
    optics[field] = value
    with pytest.raises(ValueError, match=message):
        AcquisitionConfig.from_mapping(mapping)


def test_calibration_and_transformed_image_contracts_are_validated(tmp_path: Path) -> None:
    mapping = dual_mapping()
    calibration = mapping["calibration"]
    assert isinstance(calibration, dict)
    calibration["secondary_distortion_coefficients"] = "calibration/coefficients.txt"
    path = materialize(tmp_path, mapping, stems=("1",))
    coefficients = tmp_path / "calibration/coefficients.txt"
    coefficients.parent.mkdir()
    coefficients.write_text("\n".join(["0"] * 11 + ["nan"]) + "\n", encoding="utf-8")
    config = AcquisitionConfig.load(path)
    with pytest.raises(ValueError, match="NaN or infinite"):
        config.frame_records()

    coefficients.write_text("\n".join(["0"] * 12) + "\n", encoding="utf-8")
    write_image(tmp_path / "minip/1.jpg", side=7)
    with pytest.raises(ValueError, match="expected .* after transforms"):
        config.validate_image_contracts()


def test_reconstruction_iteration_count_must_be_an_integer() -> None:
    mapping = dual_mapping()
    reconstruction = mapping["reconstruction"]
    assert isinstance(reconstruction, dict)
    reconstruction["phase_retrieval_iterations"] = float("nan")
    with pytest.raises(ValueError, match="must be an integer"):
        AcquisitionConfig.from_mapping(mapping)
