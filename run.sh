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