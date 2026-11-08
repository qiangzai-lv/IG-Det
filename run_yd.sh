export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

pip install fairscale

cd /l50062047/yundao/IG-Det_V2

bash tools/dist_train_yd.sh configs/igdet/igdet_scannet_2d.py
