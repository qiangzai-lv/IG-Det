# VGGT-Omega scene reconstruction

Reconstruct all scene directories under `/data/data/scannet_vis`. The script
uses only RGB images and the original VGGT-Omega checkpoint. Predicted cameras
and depth define the output frame; neither GT depth nor pose `.txt`, intrinsic
files, annotations or GT scale are read. Point coordinates are not metric.

Run from the project root:

```bash
cd /data/working/IG-Det-IPMSR
conda run --no-capture-output -n mmcv python vis/reconstruct_vggt_omega.py \
  --input-root /data/data/scannet_vis \
  --num-images 40 \
  --image-resolution 512 \
  --save-dense \
  --output-dir vis/outputs/vggt_omega
```

`--num-images` controls both the reconstruction's input count and the exported
RGB count per scene. Frames are naturally sorted by filename, then selected
uniformly using rounded linspace including the first/last frame. If a scene
has fewer frames than requested, all its frames are used exactly once. `-1`
selects all frames. Images are discovered non-recursively in a scene directory
or its `color`, `rgb`, or `images` subdirectory, so `depth/*.png` is excluded.

To list scenes/counts without loading the model:

```bash
conda run -n mmcv python vis/reconstruct_vggt_omega.py --list-scenes
```

To reconstruct specific scenes with a different frame count:

```bash
conda run --no-capture-output -n mmcv python vis/reconstruct_vggt_omega.py \
  --input-root /data/data/scannet_vis \
  --scenes scene0427_00 scene0638_00 \
  --num-images 80 --image-resolution 512 \
  --save-dense --output-dir vis/outputs/vggt_omega_80
```

A single image directory can also be passed as `--input-root`. Relative output
paths are relative to the current working directory; the default checkpoint
and output directory are anchored to the repository.

## Output for each scene

- `reconstruction.ply`: binary RGB point cloud in the model's predicted frame,
  directly usable in CloudCompare, MeshLab or Open3D.
- `reconstruction.png`, `reconstruction.pdf`: orthographic point-cloud preview.
- `images/`: exactly the selected RGB frames after model preprocessing, named
  by sampling index and source frame stem. Pixel coordinates match the exported
  camera intrinsics.
- `sampled_views.jpg`: contact sheet with source filenames.
- `cameras.npz`: predicted `extrinsics_w2c [V,3,4]`, `intrinsics [V,3,3]`, camera
  centers, original frame indices/names and processed image dimensions.
- `camera_centers.ply`: predicted camera centers in the point cloud's frame.
- `depth_confidence.npz`: optional (`--save-dense`) predicted depths, raw depth
  confidences and valid-pixel masks for future projection or re-rendering.
- `manifest.json`: exact frame selection, preprocessing, confidence threshold,
  filtering, preview settings, checkpoint identity/path and point counts.

The output root also contains `index.json` with all completed scene manifests.
Each scene is inferred jointly, never split into independently scaled batches.

## Density, filtering and preview settings

- `--preprocess-mode max_size` (default): longest image side is
  `--image-resolution`, rounded to patch multiples; `balanced` uses the bundled
  loader's approximately constant image-area convention. Resolution must be a
  positive multiple of 16. Mixed image shapes are center-padded and padding
  pixels are excluded from the cloud.
- `--pixel-stride 2` (default): take every other pixel in both directions when
  exporting the cloud; use `1` for full density. Model inference is unchanged.
- `--confidence-percentile 20` (default): remove the lowest 20% of finite valid
  sampled confidence values in the scene. Use `0` to keep all finite
  positive-depth points. Raw confidence values are not probabilities.
- `--max-points 1000000` (default): deterministically cap export size using
  evenly spaced indices; `0` disables this cap.
- `--preview-elev 25 --preview-azim -65`: adjust the preview camera.
- `--no-preview`: export data without Matplotlib preview rendering.
- `--overwrite`: explicitly replace existing scene outputs. A fresh output
  directory is preferable when comparing frame counts.

Preview orientation is a display-only `(x,z,-y)` basis change. Axis limits use
1%-99% point quantiles with equal coordinate scale; extreme outliers can lie
outside the preview frame. The PLY and camera data are not rotated or clipped
by this preview operation. Previews subsample at most 200,000 exported points.
For final paper camera placement, open the PLY in a 3D viewer.

More views and larger resolutions increase memory use. On OOM, select fewer
views or lower resolution; the script does not silently change the requested
reconstruction protocol. Original datasets/checkpoints are left untouched.

## Tests

```bash
conda run -n mmcv python -m pytest -q vis/test_reconstruct_vggt_omega.py
```

Tests cover uniform no-repeat sampling, natural sorting, excluding depth
images, mixed-shape padding masks, RGB conventions, confidence filtering,
binary PLY format, output counts/overwrite and synthetic-camera reconstruction.

## Farthest-point sampling (FPS)

Sample the exported reconstruction PLYs and create new PLYs combining each
scene with visible red sample markers:

```bash
conda run --no-capture-output -n mmcv python vis/sample_fps.py \
  --input-root vis/outputs/vggt_omega \
  --num-points 256 \
  --output-dir vis/outputs/vggt_omega_fps
```

Default: all five reconstructed scenes, exact greedy FPS over **all input PLY
vertices**, starting at source index 0. There is no random candidate pool or
additional downsampling before FPS. Each step selects the largest nearest-
selected squared Euclidean distance. The input is isotropically normalized
only during distance calculation for float32 stability; original exported
coordinates are retained. CPU and CUDA are supported (`--device auto` defaults
to CUDA when available). The sampling is deterministic for a fixed device and
start index, apart from potential floating-point tie boundaries. Selected
source indices are unique; if requested K exceeds N, all N indices are used.

For one PLY or a different count:

```bash
conda run --no-capture-output -n mmcv python vis/sample_fps.py \
  --input-root vis/outputs/vggt_omega/scene0427_00/reconstruction.ply \
  --num-points 128 \
  --output-dir vis/outputs/fps_128
```

Each scene output contains:

- `scene_with_fps.ply`: all original context vertices plus exact sampled
  centers and dense sphere-surface markers. Original context colors are
  retained by default. No original input PLY is modified.
- `fps_points.ply`: exactly the K selected source vertices, colored red.
- `fps_indices.npy`: K source indices into the input PLY's vertex array.
- `scene_with_fps.png` / `.pdf`: preview with the samples explicitly overlaid
  on the scene (including samples occluded in this 2D viewpoint).
- `manifest.json`: sampling settings, source path, point counts and radius.

Combined PLY vertex properties include `point_type` (0=context, 1=exact FPS
center, 2=display-only sphere surface), `fps_index` (-1 for context, otherwise
the selected point's ordinal), and `source_index` (input vertex index). Sphere
vertices are visualization geometry, not additional FPS samples. The combined
file is a point cloud without mesh faces and can be opened in CloudCompare,
MeshLab or Open3D. Isolate `point_type==1` or use `fps_points.ply` when measuring
sample locations.

Useful options:

```text
--scenes scene0427_00 scene0638_00  # process chosen scene directories
--num-points 256                   # number of FPS source vertices
--start-index 0                    # deterministic initial source index
--marker-color '#EF3340'           # highlight color
--marker-radius-ratio 0.006        # sphere radius / robust scene diagonal
--marker-radius 0.02               # alternatively set radius in source units
--marker-points 192                # sphere-surface density; 0 disables spheres
--context-gray                    # gray context for stronger contrast
--preview-elev 25 --preview-azim -65
--no-preview                      # PLY/indices only
--overwrite                       # replace existing generated outputs
```

The marker radius defaults to 0.006 times the scene's 1%-99% diagonal. FPS
still uses all vertices, including outliers; that robust statistic only
controls marker size. Preview axes are `(x,z,-y)` and show the complete source
bounds, so all selected centers can be seen. FPS on a reconstructed cloud
retains its arbitrary scale; it does not invoke metric scale recovery.

```bash
conda run -n mmcv python -m pytest -q vis/test_sample_fps.py
```

FPS tests compare CPU/CUDA choices with an independent NumPy greedy reference,
check duplicate-coordinate behavior and count limits, preserve exact source
coordinates/indices, validate sphere radius and binary PLY properties, and
confirm original reconstruction files remain unchanged.

## 验证集 Query 框与检测结果

`visualize_val_predictions.py` 加载当前 IGDet 配置和完整训练 checkpoint，默认保存验证集全部场景。保留现有模型的 GT 尺度与轴对齐路径，不引入尺度恢复模块。Query 为实例聚合后、decoder 前的实际初始化框；检测结果为现有推理后处理输出，不改模型 NMS 或推理阈值。

输出遵循 `/data/working/VGGT_Det/work_dirs/visualization_all` 的目录和命名方式：

```text
visualization_all/
├── queries/
│   ├── <scene>_initial_queries.ply       # Query 框与中心
│   ├── <scene>_gt_initial_queries.ply    # 场景点云 + Query 框与中心
│   ├── <scene>_initial_queries.png
│   ├── <scene>_initial_queries.npz
│   └── <scene>_initial_queries.json
├── predictions/
│   ├── <scene>_predictions_boxes.ply     # 检测框
│   ├── <scene>_gt_predictions.ply        # 场景点云 + 检测框
│   ├── <scene>_predictions.png
│   ├── <scene>_predictions.npz
│   └── <scene>_predictions.json
└── index.json
```

图片采用白底、灰色场景点云、正交投影、俯仰角 55°/方位角 -65°、300 DPI；检测框按 tab20 类别配色，并标注类别与分数。PLY 框以细圆柱三角网格保存，默认半径 0.01 米。`gt_` 文件名表示叠加原始 ScanNet 场景点云，不包含 GT 框。

**PLY 和图片按阈值过滤，NPZ 和 JSON 始终保存全部原始预测。** JSON 的 `queries` / `predictions` 含每个框的中心、尺寸、类别、分数；另记录显示子集和索引。检测原始数据指完整 post-NMS 输出，显示阈值不会回写或删减模型结果。无观测的原点补齐 Query 仅从显示中排除，仍保留在原始数据里。非有限分数在 JSON 中写作 null，NPZ 保留原值。

在仓库根目录执行真实推理：

```bash
conda run --no-capture-output -n mmcv python vis/visualize_val_predictions.py \
  --config work_dirs/igdet_scannet/igdet_scannet.py \
  --checkpoint work_dirs/igdet_scannet/epoch_1.pth \
  --output-dir vis/outputs/visualization_all \
  --score-thr 0.01 \
  --verify-predictions
```

也可以直接从已保存的原始预测重新导出，不加载模型或使用 GPU：

```bash
conda run --no-capture-output -n mmcv python vis/visualize_val_predictions.py \
  --from-results vis/outputs/val_predictions_epoch1 \
  --output-dir vis/outputs/visualization_all \
  --score-thr 0.01
```

`--from-results` 同时支持旧版逐场景目录和新版 `queries/`、`predictions/` 目录，输出必须使用另一个目录以保护源数据。已有非空输出目录需要显式 `--overwrite`。`--score-thr` 默认 0.3，作用于两类框；`--query-score-thr` 可以单独设置 Query 显示阈值。`--query-topk` 仅限制显示数量，原始数据不会截断。默认全部场景，可用 `--scenes` 或 `--max-scenes` 选取子集。`--max-points` 仅限制 PLY/图片背景点数，NPZ 保留收到的完整背景点云。

`--verify-predictions` 对首场景增加一次无采集前向，严格比较全部检测框、分数和类别。脚本固定随机种子，不强制模型 CUDA cumsum 不支持的全局确定性算法。

2026-11-13 已完成 epoch-1 权重 10/10 场景 × 128 视图验证，首场景采集开关前后输出完全一致。新版格式通过全部 10 场景原始数组/JSON 对照检查。当前 epoch-1 分数约 0.011–0.015，0.01 仅用于检查几何位置。

```bash
conda run -n mmcv python -m pytest -q vis/test_visualize_val_predictions.py
```

测试涵盖采集透明性、坐标变换、重力中心约定、PLY 几何、空框、阈值与保存数据隔离、JSON 全量条目以及无推理的重新过滤。
