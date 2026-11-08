bash tools/dist_train.sh configs/igdet/igdet_scannet.py 1


bash tools_mmdet/dist_train.sh configs/gdino/grounding_dino_swin-t_pretrain_obj365_ori.py 1

bash tools_mmdet/dist_test.sh configs/gdino/grounding_dino_swin-t_pretrain_obj365_ori.py /root/shared-nvme/data/pretrain/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 1 --show-dir /root/shared-nvme/code/Recondet_up/vis


python tools/build_instance_size_priors.py \
  configs/igdet/igdet_scannet.py \
  --output data/pretrain/scannet_instance_size_priors.pt \
  --max-modes 4 \
  --min-samples-per-mode 8 \
  --seed 0