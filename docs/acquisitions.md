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

## Temporal background removal

Set `background_removal.enabled: true` for raw 8-bit image sequences. Omitted settings default to disabled, so existing acquisitions retain their preprocessing. The setting applies independently to each camera, and supports both dual-camera and single-Gabor sequences. At least two synchronized source frames are required; leave it disabled for an isolated hologram.

```yaml
background_removal:
  enabled: true
  window: 129
  anchor_stride: 32
  batch_size: 16
  median_backend: auto
  lowpass: 0
  target_level: null
  save_workers: 4
```

Set `frames.minip: null` when enabling background removal. Supplied MinIP images are rejected in this mode so detection and learned reconstruction use the same corrected input.

The processing order is:

1. Read each camera's raw grayscale images in sensor coordinates.
2. Estimate per-pixel temporal median backgrounds at the first frame, every `anchor_stride` frames, and the last frame. Each centred window is clipped at the sequence boundaries; even-length boundary windows use the lower middle value.
3. Save background anchors as float16 arrays and linearly interpolate the background for each selected frame.
4. Subtract the background, remove the residual global median, then remove centred row and column median profiles smoothed with a Gaussian of sigma 3 pixels.
5. Optionally subtract a spatial low-pass residual, add the target level, round, and clip to 8-bit PNG.
6. Apply configured image transforms, then secondary-camera distortion calibration, and reconstruct the wavefront.

The background estimator and interpolation operate in two passes, keeping one temporal window, two anchors, and one correction batch in memory. Source frames are naturally sorted and synchronized by stem. Background estimation always uses the complete acquisition; `--start-index`, `--end-index`, and `--limit` select which corrected frames are saved and passed to inference. This makes a selected frame's correction consistent with a full run. Background estimation still reads the source sequence when inference selects only one frame.

| Setting | Meaning |
| --- | --- |
| `enabled` | Boolean; default `false`. |
| `window` | Positive odd temporal window length; default `129`. |
| `anchor_stride` | Positive number of frames between background anchors; default `32`. |
| `batch_size` | Positive correction-pass batch size; default `16`. |
| `median_backend` | `auto`, `torch`, `cuda-window`, or `cuda-rolling`. |
| `lowpass` | Optional spatial mean-filter width; `0` disables it. Values above `1` are rounded up to an odd width. The padding must be smaller than both input dimensions. |
| `target_level` | Target intensity in `[0, 255]`. `null` uses the median of the per-anchor spatial means, separately for each camera. |
| `save_workers` | PNG writer threads; default `4`. `0` and `1` write synchronously. |

`auto` uses the rolling CUDA histogram median for windows up to 255 frames when CUDA and the optional `cuda-python` bindings are available. Larger windows use the CUDA window median. Without the bindings it uses Torch on the selected device; CPU uses Torch as well. Explicit CUDA backends require CUDA. The temporal median is exact in every backend. For narrow test images the profile-smoothing radius is reduced to fit the image; normal images use radius 9.

Background removal preserves arbitrary image stems and requires 8-bit source images; it rejects higher-bit-depth inputs instead of silently truncating them. Image transforms run after background removal. Convert other camera encodings to 8-bit before enabling this step.

Corrected inputs and provenance are retained inside the run:

```text
_inputs/background/
├── acquisition.yaml           # points to corrected images; correction disabled to avoid a second pass
├── background_summary.json    # settings, frame counts, anchor positions, target levels, timing
├── primary/
│   ├── frame-name.png
│   └── _anchors/*.npy
└── secondary/                 # dual-camera only
    ├── frame-name.png
    └── _anchors/*.npy
```

The generated acquisition preserves the original optics, transforms, and calibration references. Both MinIP generation and the depth/diameter subprocess use it. `pipeline_summary.json` retains the original acquisition hash and adds the background settings, selected backend, and camera-specific timings. Existing prepared images require `--overwrite`; replacement clears obsolete selected frames after both cameras succeed.

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

Validation checks schema keys, finite units, required mode fields, input directories, image-stem synchronization, all 12 calibration values, and frame count. With background removal enabled, it also checks the settings, 8-bit source format, consistent sensor dimensions, and low-pass size. It executes the configured transforms for every supplied input and verifies the final image shape. It does not estimate backgrounds, reconstruct holograms, run a model, or write output. Custom transform code is therefore trusted local code and runs during validation as well as inference.
