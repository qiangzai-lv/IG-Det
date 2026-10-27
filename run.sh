bash tools/dist_train.sh configs/igdet/igdet_scannet.py 1


bash tools_mmdet/dist_train.sh configs/gdino/grounding_dino_swin-t_pretrain_obj365_ori.py 1

bash tools_mmdet/dist_test.sh configs/gdino/grounding_dino_swin-t_pretrain_obj365_ori.py /root/shared-nvme/data/pretrain/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 1 --show-dir /root/shared-nvme/code/Recondet_up/vis


MAX_PARALLEL=8 \
SHARD_START=0 \
SHARD_END=8 \
SPLIT=test \
bash utils/generate_scannet_2d_all_shards.sh

python utils/visualize_query_identity.py \
  --scene-id scene0294_00 \
  --num-queries 100 \
  --query-color '#2F80ED' \
  --query-size 30



# 生成全部训练场景标注，8 个进程并行
DATA_ROOT=data/ScanNet_processed_v2 \
OUTPUT_ROOT=data/scannet_coco_v2 \
MAX_PARALLEL=8 \
SPLIT=train \
bash utils/generate_scannet_2d_all_shards.sh

# 合并逐场景标注
OUTPUT_ROOT=data/scannet_coco_v2 \
SPLIT=train \
bash utils/merge_scannet_2d_shards.sh


conda run -n mmcv python /data/working/IG-Det/utils/visualize_training_instance_points.py \
  --num-views -1 \
  --output-dir /data/working/IG-Det/work_dirs/train_instance_points_all_views