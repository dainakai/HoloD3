# Web UI

## Start the trusted-local server

```bash
uv run hf auth login
uv run holod3 fetch-models --scope production
uv run holod3 web
```

Open <http://127.0.0.1:7860>. The server defaults to `portable-torch` and the included experimental acquisition.

The UI has no user accounts, authentication, or path sandbox. It can read paths visible to the HoloD3 process and execute GPU work. Detector checkpoints can contain executable pickle data, so use only checkpoints you created or obtained from a trusted source. Keep the default loopback binding. Do not use `--host 0.0.0.0` on an untrusted network.

The server accepts at most one queued/running full-pipeline job and rejects decoded detector images above 16,777,216 pixels. These resource guards are not an authentication boundary.

## Detector-only workflow

1. In **MinIP particle detection**, either select an image upload or enter a local image path.
2. If both are supplied, the upload is used.
3. Select **Run YOLO detection**.
4. Wait for the status to report a detection count.
5. Inspect the annotated image and detection JSON shown below the form.

This route runs only the detector. It cannot infer depth or diameter from a MinIP image alone.

## Included full-pipeline demo

1. Leave **Acquisition YAML** set to the included `data/demo/experimental/acquisition.yaml` path.
2. Leave **Output directory** empty for a unique timestamped directory below `runs/web/`, or enter a new path.
3. Keep **Frame limit** at `1` for a quick check. Enter `0` for all six demo frames.
4. Leave all three fallback controls enabled to match the validated production policy.
5. Leave the four checkpoint paths unchanged.
6. Select **Start full pipeline**.
7. The page polls the background job and shows `queued`, `running`, `complete`, or `failed` with a message.
8. On completion, use **Download particle CSV**, **Download run summary**, **Open animated 3D scatter**, and **Download run log**.

The browser may be closed after a job starts, but jobs are held only in server memory. Restarting the Web process clears the job list; files already written in the run directory remain.

## Custom acquisition

Create and validate an acquisition YAML from a template first:

```bash
cp configs/acquisitions/dual-camera-template.yaml my-acquisition/acquisition.yaml
uv run holod3 validate-acquisition my-acquisition/acquisition.yaml
```

Enter the resulting YAML path in **Acquisition YAML**. The YAML supplies image directories, optics, calibration, holography mode, and optional transforms; the Web form does not duplicate those scientific parameters.

## Custom checkpoints

Expand **Model checkpoint paths** and enter server-visible paths for any or all of:

- detector;
- primary depth scorer;
- fallback depth scorer; and
- diameter regressor.

Paths may be repository-relative or absolute. HoloD3 checks every checkpoint needed by the enabled routes before inference; the fallback depth checkpoint is optional when the depth router is disabled. The Web UI does not upload checkpoint files; large or untrusted model uploads are intentionally excluded.

## Gabor workflow

Use a validated `single_gabor` acquisition YAML in the same full-pipeline form. A warning in CLI validation and documentation indicates that packaged dual-camera-trained depth/diameter checkpoints are not scientifically calibrated for Gabor-domain crops. The Web UI executes the selected weights exactly; it does not suppress that domain limitation.

## Failure diagnosis

- **Missing checkpoint:** authenticate and run `holod3 fetch-models`, or correct the expanded checkpoint path fields.
- **Acquisition schema/path error:** run `holod3 validate-acquisition` in a terminal for a focused diagnostic.
- **Existing output directory:** choose a new directory. Web jobs never overwrite prior results.
- **Backend error:** restart with `--preset portable-torch`; use `production` only with a compatible CUDA/TensorRT environment.
- **Long runtime:** reduce the frame limit. A full 1,024-plane reconstruction evaluates many particle crops.

The downloadable log contains internal command output but never needs an access token. Review logs before sharing because acquisition paths can reveal your own local directory choices.
