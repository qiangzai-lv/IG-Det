# ARKitScenes 2D 标注生成

本工具只支持当前 ARKitScenes_processed 格式，生成普通水平 COCO 框。
输入 3D 框为几何中心 `[cx,cy,cz,dx,dy,dz,yaw]`；正 yaw 从场景 +X 转向 +Y。
点云和框共享场景坐标；PKL 中 `lidar2cam` 实际为 C2W，相机轴为右、下、前。
深度 PNG 为相机 Z，默认毫米转米；不额外翻转相机轴，不做 VGGT 归一化。
非单位 axis_align_matrix 会报错，需要先核查其数据来源，避免重复对齐。

## 生成与合并

在项目根目录运行：

```bash
conda activate mmcv
DATA_ROOT=data/ARKitScenes_processed \
OUTPUT_ROOT=data/arkitscenes_coco \
MAX_PARALLEL=4 SPLIT=all \
bash utils/generate_arkitscenes_2d_all_shards.sh

OUTPUT_ROOT=data/arkitscenes_coco SPLIT=all \
bash utils/merge_arkitscenes_2d_shards.sh
```

`SPLIT` 支持 `train`、`val`、`all`。默认处理所有场景、所有 PKL 索引视图。
`NUM_VIEWS=50` 可以均匀抽取每场景最多 50 帧，用独立 OUTPUT_ROOT 保存开发结果。
`SHARD_START`、`SHARD_END` 是当前机器负责的半开分片区间，MAX_PARALLEL 是总分片数。
设置 `VISUALIZE=1 VISUALIZATION_MAX_IMAGES=8` 可以限制每个 worker 输出的可视化数。
可视化只绘制最终被接受的训练框，被拒绝实例不画框；空标注视图同样输出图像。
完整可视化请设置 `VISUALIZE=1 VISUALIZATION_MAX_IMAGES=-1`。

单进程调试支持额外 CLI 参数：

```bash
OUTPUT_ROOT=work_dirs/arkit_label_debug \
bash utils/generate_arkitscenes_2d_train.sh \
  --max-scenes 1 --num-views 5 --visualize
```

脚本在原始数据旁生成 `arkit_infos_{train,val}_with_ids.pkl`，保留原始 PKL。
场景内实例 ID 首次按原始标注顺序生成，已有 ID 会保留，不能在过滤后重新编号。
COCO category_id=PKL bbox_label_3d+1，17 类映射严格校验。

输出：

```text
data/arkitscenes_coco/
  bbox_train_scenes/<scene>.json
  bbox_val_scenes/<scene>.json
  bbox_train.json
  bbox_val.json
  bbox_train_rejections.json
  bbox_val_rejections.json
```

## 算法与字段

场景点云按 3D OBB 归属实例。重叠区域优先归属体积较小的框，同体积按原始顺序。
点云投影后要求 `abs(point_Z - nearest_surface_Z) <= 0.05 + 0.05 * surface_Z`，
即双侧表面一致性。表面后方为遮挡，表面前方超出容差的点无深度支持；两者均不参与标注。
保留现有 3x3 邻域最近有效深度策略。再对可见实例点取水平外接矩形并裁剪到图像。
可见比例分母是该实例在场景点云中的归属点数；它不是图像面积比例。
默认可见比例阈值 0.05、至少 3 点；保留原算法的底部支撑面过滤和最近实例点回退。
这是由 3D 标注与深度生成的几何伪标注，不是官方 2D 真值。

`bbox` 是可见框；`bbox_amodal` 是完整 OBB 投影的水平外接矩形，仅作诊断。
`center_depth` 来自靠近 bbox 中心的可见实例点的相机 Z 均值；
`center_3d` 是 bbox 中心按该深度反投影得到的场景点，不是 3D 框中心。
无可见框的图像仍保留在 images 中，annotations 为空。

逐场景 JSON 记录生成版本、源标注 SHA256、几何参数、选帧参数和场景清单。
断点续跑只跳过相同来源和参数的结果；参数改变需新输出目录。
合并时拒绝参数混用、重复场景和未完成的分片。不要手工修改已生成的 JSON。
源指纹覆盖标注，不覆盖图像/深度/点云内容；替换这些文件后请使用新目录重新生成。
目前仅完成标注工具迁移，训练配置与数据加载器另行适配。

## 验证

```bash
python -m pytest -q tests/test_arkitscenes_annotations.py tests/test_arkitscenes_subset.py
```

本机 6+2 场景：训练 3445 帧 / 3340 框，验证 1200 帧 / 1002 框，
共 4645 张逐视图可视化。双侧判断生成版本为 2，
旧的单侧判断结果不应继续用于训练。
小子集没有覆盖所有 17 类，不能据此判断完整数据的类别覆盖率或训练精度。
