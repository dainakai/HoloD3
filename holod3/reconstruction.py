"""Shared dual-camera and single-hologram reconstruction primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from holod3.acquisition import AcquisitionConfig, FrameRecord
from holod3.transforms import load_transformed_image, save_grayscale_png


@dataclass(frozen=True)
class PropagationSetup:
    image_size: int
    padding_side: int
    slice_count: int
    slice_spacing_um: float
    reconstruction_start_um: float
    phase_forward: torch.Tensor | None
    phase_inverse: torch.Tensor | None
    initial_transfer: torch.Tensor
    slice_transfer: torch.Tensor
    distortion_coefficients: np.ndarray | None


def transfer_sqrt_array(size: int, wavelength_um: float, pixel_pitch_um: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(size, device=device, dtype=torch.float32) - (size / 2)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    scale = float(wavelength_um) / float(size) / float(pixel_pitch_um)
    value = 1.0 - (xx * scale) ** 2 - (yy * scale) ** 2
    if bool((value < 0).any().item()):
        raise ValueError(
            "The optical sampling produces evanescent frequencies. Increase pixel_pitch_um or adjust image_size_px."
        )
    return value


def transfer(distance_um: float, wavelength_um: float, sqrt_part: torch.Tensor) -> torch.Tensor:
    phase = (2.0 * math.pi * float(distance_um) / float(wavelength_um)) * torch.sqrt(sqrt_part)
    return torch.exp(1j * phase).to(torch.complex64)


def build_propagation_setup(config: AcquisitionConfig, device: torch.device) -> PropagationSetup:
    optics = config.optics
    numerical = config.reconstruction
    base_sqrt = transfer_sqrt_array(optics.image_size_px, optics.wavelength_um, optics.pixel_pitch_um, device)
    padded_sqrt = transfer_sqrt_array(
        numerical.fft_padding_side,
        optics.wavelength_um,
        optics.pixel_pitch_um,
        device,
    )
    phase_forward = phase_inverse = None
    if config.mode == "dual_phase_retrieval":
        assert optics.phase_retrieval_distance_um is not None
        phase_forward = transfer(optics.phase_retrieval_distance_um, optics.wavelength_um, base_sqrt)
        phase_inverse = transfer(-optics.phase_retrieval_distance_um, optics.wavelength_um, base_sqrt)

    coefficients = config.load_distortion_coefficients()

    return PropagationSetup(
        image_size=optics.image_size_px,
        padding_side=numerical.fft_padding_side,
        slice_count=optics.slice_count,
        slice_spacing_um=optics.slice_spacing_um,
        reconstruction_start_um=optics.reconstruction_start_um,
        phase_forward=phase_forward,
        phase_inverse=phase_inverse,
        initial_transfer=torch.fft.ifftshift(
            transfer(-optics.reconstruction_start_um, optics.wavelength_um, padded_sqrt)
        ),
        slice_transfer=torch.fft.ifftshift(transfer(-optics.slice_spacing_um, optics.wavelength_um, padded_sqrt)),
        distortion_coefficients=coefficients,
    )


def quadratic_distortion_correction(image: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the original 12-coefficient, one-based quadratic calibration."""

    height, width = image.shape
    if height != width:
        raise ValueError(f"Quadratic calibration requires a square image, got {image.shape}.")
    i, j = np.indices((height, width), dtype=np.float64)
    i += 1.0
    j += 1.0
    ref_x = np.rint(
        coefficients[0]
        + coefficients[1] * j
        + coefficients[2] * i
        + coefficients[3] * j * j
        + coefficients[4] * i * j
        + coefficients[5] * i * i
    ).astype(np.int64)
    ref_y = np.rint(
        coefficients[6]
        + coefficients[7] * j
        + coefficients[8] * i
        + coefficients[9] * j * j
        + coefficients[10] * i * j
        + coefficients[11] * i * i
    ).astype(np.int64)
    valid = (ref_x >= 1) & (ref_x <= width) & (ref_y >= 1) & (ref_y <= height)
    output = np.full_like(image, float(image.mean()), dtype=np.float32)
    output[valid] = image[ref_y[valid] - 1, ref_x[valid] - 1]
    return output


def phase_retrieval_wavefront(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    forward: torch.Tensor,
    inverse: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    primary_amplitude = torch.sqrt(torch.clamp(primary, min=0.0))
    secondary_amplitude = torch.sqrt(torch.clamp(secondary, min=0.0))
    wavefront = primary_amplitude.to(torch.complex64)
    forward_unshifted = torch.fft.ifftshift(forward)
    inverse_unshifted = torch.fft.ifftshift(inverse)
    for _ in range(int(iterations)):
        at_secondary = torch.fft.ifft2(torch.fft.fft2(wavefront) * forward_unshifted)
        at_secondary = secondary_amplitude * torch.exp(1j * torch.angle(at_secondary))
        wavefront = torch.fft.ifft2(torch.fft.fft2(at_secondary) * inverse_unshifted)
        wavefront = primary_amplitude * torch.exp(1j * torch.angle(wavefront))
    return wavefront


def gabor_wavefront(primary: torch.Tensor) -> torch.Tensor:
    """Create the zero-phase inline-Gabor wavefront from one intensity image."""

    return torch.sqrt(torch.clamp(primary, min=0.0)).to(torch.complex64)


def load_wavefront(
    config: AcquisitionConfig,
    record: FrameRecord,
    setup: PropagationSetup,
    device: torch.device,
) -> torch.Tensor:
    primary_np, secondary_np = load_frame_arrays(config, record, setup)
    return wavefront_from_arrays(config, primary_np, secondary_np, setup, device)


def load_frame_arrays(
    config: AcquisitionConfig,
    record: FrameRecord,
    setup: PropagationSetup,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load, transform, calibrate, and shape-check one frame on the CPU."""

    primary_np = load_transformed_image(
        record.primary,
        config.transform_steps("primary"),
        base_dir=config.base_dir,
    )
    if primary_np.shape != (setup.image_size, setup.image_size):
        raise ValueError(
            f"Primary frame {record.primary.name} has shape {primary_np.shape}; "
            f"expected {(setup.image_size, setup.image_size)} after transforms."
        )
    if config.mode == "single_gabor":
        return primary_np, None

    if record.secondary is None or setup.phase_forward is None or setup.phase_inverse is None:
        raise RuntimeError("Dual-camera reconstruction is missing a synchronized secondary image or transfer.")
    secondary_np = load_transformed_image(
        record.secondary,
        config.transform_steps("secondary"),
        base_dir=config.base_dir,
    )
    if setup.distortion_coefficients is not None:
        secondary_np = quadratic_distortion_correction(secondary_np, setup.distortion_coefficients)
    if secondary_np.shape != (setup.image_size, setup.image_size):
        raise ValueError(
            f"Secondary frame {record.secondary.name} has shape {secondary_np.shape}; "
            f"expected {(setup.image_size, setup.image_size)} after transforms."
        )
    return primary_np, secondary_np


def wavefront_from_arrays(
    config: AcquisitionConfig,
    primary_np: np.ndarray,
    secondary_np: np.ndarray | None,
    setup: PropagationSetup,
    device: torch.device,
) -> torch.Tensor:
    """Move prepared arrays to a device and construct the configured wavefront."""

    primary = torch.from_numpy(primary_np).to(device=device, dtype=torch.float32)
    if config.mode == "single_gabor":
        return gabor_wavefront(primary)
    if secondary_np is None or setup.phase_forward is None or setup.phase_inverse is None:
        raise RuntimeError("Dual-camera reconstruction is missing a secondary image or transfer.")
    secondary = torch.from_numpy(secondary_np).to(device=device, dtype=torch.float32)
    return phase_retrieval_wavefront(
        primary,
        secondary,
        setup.phase_forward,
        setup.phase_inverse,
        config.reconstruction.phase_retrieval_iterations,
    )


def mean_pad_wavefront(wavefront: torch.Tensor, side: int) -> torch.Tensor:
    input_side = int(wavefront.shape[-1])
    padded = torch.empty((side, side), device=wavefront.device, dtype=torch.complex64)
    padded.fill_(wavefront.mean())
    offset = (side - input_side) // 2
    padded[offset : offset + input_side, offset : offset + input_side] = wavefront
    return padded


def minimum_intensity_projection(
    wavefront: torch.Tensor,
    setup: PropagationSetup,
    *,
    slice_step: int = 1,
) -> torch.Tensor:
    """Reconstruct selected planes and return their per-pixel minimum intensity."""

    offset = (setup.padding_side - setup.image_size) // 2
    frequency = torch.fft.fft2(mean_pad_wavefront(wavefront, setup.padding_side)) * setup.initial_transfer
    projection: torch.Tensor | None = None
    previous_slice = 0
    for slice_index in range(0, setup.slice_count, max(1, int(slice_step))):
        gap = slice_index - previous_slice
        if gap:
            frequency = frequency * (setup.slice_transfer**gap)
        previous_slice = slice_index
        field = torch.fft.ifft2(frequency)
        intensity = (field.real.square() + field.imag.square()).clamp_(0.0, 1.0)
        central = intensity[offset : offset + setup.image_size, offset : offset + setup.image_size]
        projection = central if projection is None else torch.minimum(projection, central)
    if projection is None:
        raise RuntimeError("No reconstruction slices were selected.")
    return projection


def prepare_minip_images(
    config: AcquisitionConfig,
    records: list[FrameRecord],
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize model-ready MinIP PNGs from supplied projections or raw holograms."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    setup = build_propagation_setup(config, torch_device)
    generated = config.minip_dir is None
    outputs: list[Path] = []
    with torch.inference_mode():
        for record in records:
            output = destination / f"{record.stem}.png"
            if output.exists() and not overwrite:
                raise FileExistsError(
                    f"Prepared MinIP already exists: {output}. Choose a new run directory or pass --overwrite."
                )
            if record.minip is not None:
                image = load_transformed_image(
                    record.minip,
                    config.transform_steps("minip"),
                    base_dir=config.base_dir,
                )
            else:
                wavefront = load_wavefront(config, record, setup, torch_device)
                image = (
                    minimum_intensity_projection(
                        wavefront,
                        setup,
                        slice_step=config.reconstruction.minip_slice_step,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                if config.transform_steps("minip"):
                    from holod3.transforms import apply_transforms

                    image = apply_transforms(image, config.transform_steps("minip"), base_dir=config.base_dir)
            if image.shape != (setup.image_size, setup.image_size):
                raise ValueError(
                    f"MinIP frame {record.stem!r} has shape {image.shape}; "
                    f"expected {(setup.image_size, setup.image_size)} after transforms."
                )
            outputs.append(save_grayscale_png(output, image))
    return {
        "mode": config.mode,
        "source": "reconstructed" if generated else "provided",
        "frames": len(outputs),
        "output_dir": str(destination),
        "device": str(torch_device),
    }
