脚本 `extract_arkitscenes_subset.py` 从服务器上的 processed ARKitScenes 提取完整场景。只依赖 Python、NumPy，无需安装本项目或 MMDetection。

每个选中场景完整复制 `processed/3dod/{train,val}/{scene_id}/` 下的所有文件及子目录（包括 pkl 没有引用的帧/辅助文件），并复制完整 `points/{scene_id}_points.bin`。场景记录、所有帧、17 类标签、7D 旋转框、相机参数和原始 metainfo 均保留；不会采样帧、点云或转换坐标。源 metainfo.dataset 为 scannet，这是源数据的内容，脚本不自动改写它。

训练场景来自 train，`--test-scenes` 指定的开发测试场景来自 val，输出标注仍使用 `arkit_infos_val.pkl`，不制造新的正式 test split。

将本机脚本复制到存有完整数据的服务器后，在脚本目录执行：

```bash
python extract_arkitscenes_subset.py \
  --source /home/Newdisk/lxq/data/ARKitScenes_processed \
  --output /home/Newdisk/lxq/data/ARKitScenes_dev \
  --train-scenes 2 \
  --test-scenes 1 \
  --seed 2026 \
  --dry-run
```

去掉 `--dry-run` 重新执行即可实际提取。修改两个 scenes 参数控制场景数量，可以把其中一个设为 0。固定 seed 使选择可复现；场景数相同且源数据不变时，dry-run 和实际提取选取相同场景。`--skip-missing` 可跳过缺少必要文件的场景，改选其他场景；缺失与选中情况会记录在 manifest。默认缺失文件直接报错，已有输出目录始终拒绝覆盖，失败不会留下半成品输出目录。

输出结构：

```text
ARKitScenes_dev/
  arkit_infos_train.pkl
  arkit_infos_val.pkl
  subset_manifest.json
  points/<selected_scene>_points.bin
  processed/3dod/train/<selected_scene>/...全部文件...
  processed/3dod/val/<selected_scene>/...全部文件...
```

如果某个 split 请求数量为 0，不输出该 split 的 pkl。summary 中 copied_file_mib 是待复制数据文件的体积，不含新生成的标注与 manifest。选中场景 ID、源记录索引、文件清单和字节数记录在 subset_manifest.json。相对路径保持原布局，整个目录迁移到本机后，将数据根目录指向新目录即可找到这些文件；训练配置、ARKitScenes 旋转框与 2D 标注/实例 ID 的接入需另行适配。

打包与传输，在数据服务器执行：

```bash
tar -C /home/Newdisk/lxq/data \
  -czf /home/Newdisk/lxq/data/ARKitScenes_dev.tar.gz ARKitScenes_dev
```

在本机执行（替换服务器 IP 和本机目标路径）：

```bash
scp lxq@SERVER_IP:/home/Newdisk/lxq/data/ARKitScenes_dev.tar.gz /LOCAL_STORAGE/
tar -xzf /LOCAL_STORAGE/ARKitScenes_dev.tar.gz -C /LOCAL_STORAGE/
```

该子集用于代码开发和 smoke test，不能代表完整数据集精度。当前完整数据位于远端，本机只基于用户提供的结构用模拟数据验证；未在远端真实数据上运行。
