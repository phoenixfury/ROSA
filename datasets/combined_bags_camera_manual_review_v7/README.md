# Reviewed v7 for ROSA: offline canonical images

**Superseded:** the current v7 launcher uses native images with runtime resize.
See `projects/ROSA/NATIVE_V7.md`. The instructions below describe the earlier
optional offline export only; they are NOT required for the current experiment.

The exported TRAIN/VAL files preserve **every** v7 GT annotation, removal,
class, cuboid, track ID, token, ordering, split and LiDAR extrinsic. Only camera
paths, image dimensions and image intrinsics change. Native images are never
overwritten. ROSA class order is Car, Van, Truck, Bus, Pedestrian, Cyclist.

9959 training frames / 109831 boxes; 1034 validation frames / 9295 boxes.
43972 camera images must be processed. The metadata has been generated, but
only **12 smoke-test images** have been converted locally. This is NOT a fully
converted training dataset yet. `images_complete.json` is deliberately absent.

Conversion uses each camera's calibrated OpenCV-fisheye rays and the released
800x800 OCam lens (`calib_results.txt`, FOV 220 degrees), bilinear sampling,
black invalid pixels, JPEG quality 95. It is not an aspect-ratio distortion or
a resize of the native 1920x1536 frame. Four-camera order is front/left/right/rear.
Resampling is lossy; invalid/cropped lens regions cannot recover native content.
This preprocessing is **only for ROSA**. RayBEV-YOLOX still reads native images.

## Server conversion and verification

From the repository root, activate the existing ROSA environment. Transfer this
dataset's metadata, both preparation scripts below, the new training config
and launcher, and the original reviewed RayBEV v7 PKLs (for exact verification).
The server must already have the ROSA project and its dependencies.

```bash
cd ~/mmdetection3d
export NATIVE_ROOT="$PWD/data/combined_bags_camera_ray_dedup_v2"
python tools/custom_data/prepare_rosa_reviewed_v7.py --stage images \
  --native-root "$NATIVE_ROOT" --workers 4
python tools/custom_data/prepare_rosa_reviewed_v7.py --stage verify \
  --native-root "$NATIVE_ROOT" --deep
bash projects/ROSA/tools/launch_manual_review_v7.sh
```

`images` resumes from an SHA256 journal, checks native image dimensions,
and refuses changed source pixels or different existing output JPEGs. Rerun
the same command after interruption. Do not run two writers concurrently.
`--limit-frames 3` is a smoke test only and never marks conversion complete.
Reserve disk space for 43972 new JPEGs. No point clouds are copied; ROSA is
camera-only. Copied `pc_info` paths remain native-dataset-relative and are not
intended for a point-cloud loader in this export.

To regenerate metadata into a NEW empty output directory rather than using
this prepared export:

```bash
python tools/custom_data/prepare_rosa_reviewed_v7.py --stage prepare \
  --native-root "$NATIVE_ROOT" --output /path/to/new/rosa_v7
```

Use the same `--output` with later stages and set `ROSA_DATA_ROOT` to it when
training. `prepare_fisheye3dod_inference.py` supplies the shared calibrated
remap implementation; keep it beside `prepare_rosa_reviewed_v7.py`.
Source/camera/output identities and class totals are in `prepared.json`.

## Training

Config: `projects/ROSA/configs/fisheye3dod/rosa_vov99_manual_review_v7_24e.py`.
Launcher refuses incomplete images, changed reviewed labels, missing init
weights or a nonempty output run. Defaults: single-frame VoV99, two GPUs,
batch 1/GPU, accumulation 4, effective batch 8, up to 24 epochs with inherited
plateau stopping. No training has been launched or GPU-memory-tested here.

Default full-model warm-start is
`projects/ROSA/work_dirs/combined_4bags/best_fisheye_bev_mAP_epoch_6.pth`.
Set `ROSA_INIT_CKPT` to its actual server location. Initialization uses a fresh
optimizer/schedule, not resume. TRAIN-only MiniBatchKMeans anchors are exported
for the constructor/fresh experiments; loading a full checkpoint restores its
learned anchors too. Do not interpret this as fresh 3D training or a clean
unseen-track comparison without auditing the old checkpoint's training split.

`DRY_RUN=1 bash projects/ROSA/tools/launch_manual_review_v7.sh` prints the command
only **after** dataset and checkpoint preflight have passed.
