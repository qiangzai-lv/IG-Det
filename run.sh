cd /data/working/IG-Det-IPMSR

conda run --no-capture-output -n mmcv python vis/visualize_val_predictions.py \
  --config work_dirs/igdet_scannet/igdet_scannet.py \
  --checkpoint work_dirs/igdet_scannet/epoch_1.pth \
  --output-dir vis/outputs/visualization_all \
  --score-thr 0.01 \
  --verify-predictions \
  --overwrite

python vis/filter_predictions_with_gt.py \
  --input-dir vis/outputs/visualization_all \
  --config configs/igdet/igdet_scannet.py \
  --output-dir vis/outputs/filtered_predictions \
  --score-thr 0.013 \
  --overwrite

  python vis/filter_predictions_by_gt_iou.py \
  --input-dir vis/outputs/visualization_all \
  --config configs/igdet/igdet_scannet.py \
  --output-dir vis/outputs/gt_iou_predictions \
  --min-iou 0 \
  --overwrite


cd /Users/lvxueqiang/Desktop/CVPR2027_v3/visualize

python3 download_scannet_mesh.py --accept-terms \
  --scene-ids scene0427_00

set "BLENDER_EXE=D:\Blender\blender.exe"
render_predictions_windows.bat --scene scene0025_02


python vis/filter_query_boxes_nms_topk.py \
  --input-dir vis/outputs/visualization_all/queries \
  --output-dir vis/outputs/query_nms_topk \
  --nms-iou-thr 0.5 \
  --top-k 64 \
  --overwrite


python3 visualize/project_query_boxes.py --scene scene0015_00