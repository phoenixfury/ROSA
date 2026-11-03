# ROSA coordinates and ONNX deployment

## Output coordinates: already x-forward, y-left, z-up

The custom-data model predicts in **CLIDAR**, the LiDAR coordinate frame at the
current synchronized frame, not a camera frame, odometry/world frame, or the
vehicle base-link origin. Its convention is right-handed:

| Field | Meaning |
|---|---|
| x, y, z | Forward, left, up; meters; relative to the LiDAR origin |
| length, width, height | Local box dimensions, meters; length lies along its heading |
| yaw | Radians about +z; 0 points along +x, positive turns toward +y |
| ONNX/portable JSON origin | **Geometric/gravity center**, not the bottom of the box |

No axis converter or 90/180-degree correction is necessary. In particular, do
not convert the final boxes using the internal OCam camera basis. The projection
adapter changes camera coordinates from `[right, down, forward]` to
`[down, right, back]`; it does not change the model's output LiDAR axes.

Evidence in the current pipeline:

- `tools/custom_data/append_improved_detzero_rosa_dataset.py` copies the DetZero
  `[x,y,z,length,width,height,yaw]` center-origin boxes without rotating axes.
- `rosa/integrations/fisheye3dod/dataset.py` loads them with
  `LiDARInstance3DBoxes(..., origin=(.5,.5,.5))`; `model.py` restores gravity
  centers before loss and wraps decoded center-origin boxes consistently.
- `projects/mmdet3d_plugin/models/detection3d/decoder.py` decodes center XYZ,
  exponentiated dimensions and `atan2(sin_yaw,cos_yaw)` without an axis rotation.
- Calibration independently confirms orientation: the front optical axis in
  CLIDAR is approximately `[+0.989,+0.018,-0.147]`, left is
  `[-0.025,+0.713,-0.701]`, right is `[-0.011,-0.704,-0.710]`, and rear is
  `[-0.895,0,-0.446]`. The cameras tilt down; front/left/right directions agree
  with +x/+y/-y. The recorder retains the original CLIDAR point coordinates.

**MMDetection API pitfall:** `pred_instances_3d.bboxes_3d.tensor` uses its standard
bottom-center storage even though the detector state is center-origin. To obtain
canonical boxes from that API:

```python
boxes = sample.pred_instances_3d.bboxes_3d
canonical = boxes.tensor.detach().cpu().numpy().copy()
canonical[:, :3] = boxes.gravity_center.detach().cpu().numpy()
```

Alternatively, `deployment.runtime.convert_box_origin(array, 'bottom_center',
'gravity_center')` adds height/2 to z without changing x, y, dimensions or yaw.
The ONNX inference script exposes `--box-origin bottom_center` if a consumer
needs that origin; the default remains `gravity_center`. Existing portable JSON
already marked `box_origin=gravity_center` must **not** be converted again.

If a consumer needs `base_link`, the LiDAR mounting transform must be applied;
the two origins are not interchangeable. No guessed mounting transform is used.
Velocities and persistent track IDs are intentionally not exported: this model
is single-frame and its baseline velocity targets were zeros.

## Prepared model

The default checkpoint is the overall detection candidate from E007:

```text
projects/ROSA/work_dirs/singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth
```

Generated bundle:

```text
projects/ROSA/work_dirs/onnx_singleframe_e1/
  rosa_e1.onnx                 # FP32, standard ONNX opset 17, ~309 MiB
  rosa_e1.json                 # Model/checkpoint hashes and coordinate contract
  calibration.npz             # First validation sample's converted rig calibration
  calibrations/*.npz          # Correct per-rig calibration selected by the manifest
  validation_samples.json     # All 601 validation tokens/images/calibration paths
  example_{0,200,400}_input.npz
  example_{0,200,400}_reference.npz
  validation_cuda.json        # Numerical validation, generated separately
```

This is a correctness-first static export: batch 1, four **canonical 800x800**
fisheye images, 300 queries. It is not a quantized model or a TensorRT engine.
The FP32 graph need not be faster than the original optimized CUDA/AMP model.
Full ONNX validation-set AP/NDS has not been recomputed; numerical checks use
real frames from all three bags, plus a CPU test.

## Dependency setup

On the local machine dependencies were installed into an isolated directory,
not into the existing training environment. From the mmdetection3d root:

```bash
conda run --no-capture-output -n openmmlab python -m pip install \
  --no-deps --target projects/ROSA/work_dirs/onnx_deps_py38 \
  -r projects/ROSA/deployment/requirements-py38.txt
```

The tools automatically use that directory under Python 3.8. The tested stack
is PyTorch 2.4.0+cu121, CUDA 12.1, cuDNN 9.1, ONNX 1.16.2 and ONNX Runtime GPU
1.19.2. Existing NumPy/protobuf/packaging/sympy remain untouched.

On a different server, check CUDA and cuDNN versions before choosing a runtime
wheel. In particular, cuDNN major versions must match; see the
[official ONNX Runtime CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
The inference tool can optionally import PyTorch to preload its matching CUDA
libraries, as described in the
[official interoperability guidance](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch).
CPU inference does not import PyTorch or OpenMMLab. An independent CPU runtime
needs NumPy, OpenCV and ONNX Runtime; it does not need the checkpoint or detector
repositories. CUDA initialization failure raises an error instead of silently
running the entire model on CPU.

## Quick GPU test

The bundle is already exported locally. From the mmdetection3d root:

```bash
BUNDLE=projects/ROSA/work_dirs/onnx_singleframe_e1

conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/infer_fisheye_onnx.py \
  --model "$BUNDLE/rosa_e1.onnx" \
  --samples "$BUNDLE/validation_samples.json" \
  --provider cuda --max-frames 10 \
  --output "$BUNDLE/predictions_10_frames.json"
```

Remove `--max-frames` to process the full validation set. `--start-frame` selects
a later manifest row. On the server, add `--data-root /your/copied/dataset/root`;
the sample manifest has relative image paths and does not require editing PKLs.
Copy the model **and its matching JSON**, calibration files and sample manifest.

For a single new synchronized image set, use matching canonical images in the
fixed order **front, left, right, rear**:

```bash
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/infer_fisheye_onnx.py \
  --model "$BUNDLE/rosa_e1.onnx" \
  --images /path/front.jpg /path/left.jpg /path/right.jpg /path/rear.jpg \
  --calibration "$BUNDLE/calibration.npz" \
  --provider cuda --output "$BUNDLE/new_frame.json"
```

Use the appropriate manifest calibration for a different rig; `calibration.npz`
is specifically the first sample's rig. Do not blindly resize full-resolution
fisheye images: the crop/remap and calibration must match the training export.
Calibration values are graph inputs, but shapes, camera count, lens model and
10-term inverse polynomial are fixed; a different layout needs a new export.

For CPU testing change `--provider cuda` to `--provider cpu`.

The output JSON contains `frames[].boxes_3d`, `scores`, `labels`, `names`,
`box_origin`, explicit coordinate metadata, providers and per-frame network
timings. Scores are quality-weighted class probabilities. The default display
threshold is 0.10; `--score-threshold` changes it. There is no distance filter
by default; use `--max-distance 60` for an explicit XY radius if desired.

## Export / verify again

Use trusted local PyTorch checkpoints only. No training is started:

```bash
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/export_fisheye_onnx.py \
  --checkpoint projects/ROSA/work_dirs/singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth \
  --data-root data/Custom/combined_improved_detzero_rosa \
  --output-dir projects/ROSA/work_dirs/onnx_singleframe_e1_NEW
```

Export tracing and references run on CPU to avoid competing with the local
6-GB GPU. Existing model files are protected unless `--force` is explicit.
Export still uses the existing training environment and built ROSA host; only
standalone ONNX inference is independent of those detector/CUDA-extension packages.
The exporter retains all learned camera-ray fusion, geometry attention,
point attention and refinement. An export-scoped tensor adapter replaces only
custom CUDA feature reading with standard bilinear `GridSample` and grouped
reductions. No permanent source/weight changes or ONNX custom plugins are used.
The exporter checks the model with ONNX and rejects nonstandard operator domains.

Class-wise **rotated BEV NMS at IoU 0.5** remains explicit NumPy/OpenCV host
postprocessing. It is not dropped or replaced by axis-aligned/image NMS. Its
suppression footprint matches the existing MMCV CPU/CUDA decoder convention;
the physical returned yaw is unchanged. Equal-score tie ordering and values
exactly at an IoU threshold can differ across implementations.

Validate the exported graph against its real FP32 PyTorch references:

```bash
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/validate_fisheye_onnx.py \
  --model "$BUNDLE/rosa_e1.onnx" --provider cuda \
  --report "$BUNDLE/validation_cuda.json"
```

Or test one saved input and reference through the inference CLI:

```bash
conda run --no-capture-output -n openmmlab python \
  projects/ROSA/tools/infer_fisheye_onnx.py \
  --model "$BUNDLE/rosa_e1.onnx" \
  --input "$BUNDLE/example_0_input.npz" \
  --reference "$BUNDLE/example_0_reference.npz" \
  --provider cuda --output "$BUNDLE/example_prediction.json"
```

Numerical checks compare all 300 decoded boxes, scores and classes before NMS,
with circular yaw differences. The original native GPU evaluator uses AMP;
FP32 ONNX is not expected to be bit-identical to that FP16/mixed-precision path.
`--native-predictions` on the validator optionally records a separate
class-aware, nearest-center diagnostic against native gravity-center JSON.

`export_fisheye_onnx.py --prepare-only` refreshes sample/calibration files without
changing the graph; `--references-only` regenerates references for an existing
matching checkpoint/model without re-exporting. The validation manifest is
portable via the runtime's `--data-root` override.

Tests, including the native GPU reader comparison:

```bash
ROSA_DEPLOY_CUDA_TESTS=1 conda run --no-capture-output -n openmmlab \
  python -m pytest -q projects/ROSA/tests/test_onnx_deployment.py
```
