# Acquisition configuration

`acquisition.yaml` is the complete, model-independent description of one movable acquisition. All relative paths are resolved from the YAML file, so the directory may be copied without editing absolute paths.

## Recommended layout

```text
my-acquisition/
├── acquisition.yaml
├── holograms/
│   ├── primary/
│   │   ├── 000001.png
│   │   └── 000002.png
│   └── secondary/          # dual_phase_retrieval only
│       ├── 000001.png
│       └── 000002.png
├── minip/                  # optional; generated when omitted
│   ├── 000001.png
│   └── 000002.png
└── calibration/
    └── secondary_distortion_coefficients.txt
```

Files are synchronized by stem. For example, `primary/000001.png`, `secondary/000001.tif`, and `minip/000001.png` form one frame even though their extensions differ. Missing, extra, or duplicate stems are rejected. Directory and file prefixes have no camera-specific naming requirement.

## Complete dual-camera example

```yaml
schema_version: 1
name: my-dual-camera-acquisition
description: Background-corrected synchronized holograms.
mode: dual_phase_retrieval

frames:
  primary_holograms: holograms/primary
  secondary_holograms: holograms/secondary
  minip: minip                    # omit to reconstruct MinIP images

optics:
  wavelength_um: 0.6328
  pixel_pitch_um: 10.0
  image_size_px: 1024
  reconstruction_start_um: 80200.0
  slice_spacing_um: 100.0
  slice_count: 1024
  phase_retrieval_distance_um: 33050.0

reconstruction:
  phase_retrieval_iterations: 3
  fft_padding_side: 1536
  minip_slice_step: 2

calibration:
  secondary_distortion_coefficients: calibration/secondary_distortion_coefficients.txt

transforms:
  primary: []
  secondary: []
  minip: []
```

## Field meanings and units

| Field | Meaning |
| --- | --- |
| `mode` | `dual_phase_retrieval` or `single_gabor`. |
| `frames.primary_holograms` | Directory for the wavefront-reference intensity images. |
| `frames.secondary_holograms` | Synchronized second-plane images; required only for dual-camera phase retrieval. |
| `frames.minip` | Optional precomputed detector projections. When omitted, HoloD3 reconstructs them from the raw holograms. |
| `wavelength_um` | Illumination wavelength in micrometres. |
| `pixel_pitch_um` | Effective object/sensor-plane pixel pitch represented by one input pixel, in micrometres. This value scales output `x_um` and `y_um`. |
| `image_size_px` | Expected square image side after transforms. |
| `reconstruction_start_um` | Physical depth of slice 1. |
| `slice_spacing_um` | Distance between adjacent reconstructed slices. |
| `slice_count` | Number of reconstruction planes. |
| `phase_retrieval_distance_um` | Propagation distance from the primary to secondary camera plane. |
| `phase_retrieval_iterations` | Alternating amplitude-constraint iterations for dual-camera mode. |
| `fft_padding_side` | Padded FFT side. It must be at least `image_size_px`, with an even difference. |
| `minip_slice_step` | Plane stride used only when HoloD3 generates a MinIP. A larger value is faster but samples fewer planes. |

For a predicted one-based `slice`, HoloD3 reports:

```text
z_um = (slice - 1) * slice_spacing_um
depth_um = reconstruction_start_um + z_um
x_um = x_px * pixel_pitch_um
y_um = y_px * pixel_pitch_um
```

## Secondary-camera distortion calibration

`secondary_distortion_coefficients` is optional. The text file must contain 12 floating-point coefficients for the validated quadratic coordinate mapping. Omit the field when images have already been registered or when the instrument uses no secondary plane.

The calibration is applied after the secondary image transforms and before phase retrieval.

## Image transforms

Every image is loaded as two-dimensional grayscale `float32` in `[0, 1]`. Transform steps execute in listed order. Built-ins are:

- `identity`
- `invert`
- `flip_horizontal`
- `flip_vertical`
- `rotate_quarter_turns` with `turns: 1`, `2`, or `3`
- `crop` with `x`, `y`, `width`, and `height`

Custom functions may be imported from an installed module or a file relative to `acquisition.yaml`:

```yaml
transforms:
  primary:
    - function: my_package.holograms:correct_primary
      kwargs:
        gain: 1.08
  secondary:
    - function: transforms/instrument.py:correct_secondary
  minip:
    - function: invert
```

The function contract is:

```python
import numpy as np

def correct_primary(image: np.ndarray, *, gain: float) -> np.ndarray:
    assert image.ndim == 2 and image.dtype == np.float32
    return np.clip(image * gain, 0.0, 1.0).astype(np.float32)
```

HoloD3 rejects non-finite output, non-grayscale output, and values outside `[0, 1]`. If a transform changes image dimensions, the final dimensions must match `image_size_px`.

Reported `seg_xc/seg_yc` and `x_um/y_um` use the final transformed-image coordinate frame. Built-in crop, flip, and rotation operations do not retain an affine mapping back to the original sensor frame. If downstream work needs original-sensor coordinates, either invert the known transform in that downstream step or use a custom transform package that records its own mapping alongside the acquisition.

## Single-hologram inline Gabor mode

Use [single-gabor-template.yaml](../configs/acquisitions/single-gabor-template.yaml). The `frames` block contains only `primary_holograms`; `secondary_holograms` and `phase_retrieval_distance_um` must be omitted.

HoloD3 interprets each input as intensity and creates the initial zero-phase complex wavefront:

```text
wavefront = sqrt(max(intensity, 0)) + 0i
```

It then uses the same angular-spectrum propagation and slice sampling as dual-camera mode. This is true single-image reconstruction: no duplicate or placeholder second-camera image is read.

The packaged learned checkpoints are not Gabor-domain calibrated. Supply domain-matched checkpoints for quantitative work.

## Validation before inference

```bash
uv run holod3 validate-acquisition my-acquisition/acquisition.yaml
```

Validation checks schema keys, finite units, required mode fields, input directories, image-stem synchronization, all 12 calibration values, and frame count. It also executes the configured transforms for every supplied input and verifies the final image shape. It does not reconstruct holograms, run a model, or write output. Custom transform code is therefore trusted local code and runs during validation as well as inference.
