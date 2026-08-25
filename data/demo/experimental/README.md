# Experimental demo acquisition

This directory contains six consecutive frames from a real dual-camera
hologram acquisition. It is the small, directly runnable example shipped with
HoloD3. Frame identifiers were shortened to acquisition-local numbers and the
directory names describe their role instead of the original instrument layout.

The MinIP images, raw background-removed holograms, optical geometry, and
secondary-camera distortion calibration are included. The data has no physical
particle ground truth and must not be used as an accuracy benchmark.

Run the complete demo with:

```bash
uv run holod3 demo --preset portable-torch
```

The command writes `runs/demo/particles_3d.csv` and the offline interactive
animation `runs/demo/particles_3d.html`.
