"""Joint fine-tuning of the existing detector and iterative IP-MSR.

Build priors first; pass the trained baseline checkpoint via
--cfg-options load_from=/absolute/path/to/baseline.pth (not --resume).
"""
_base_ = ['./igdet_scannet.py']

load_from = ''

model = dict(ipmsr_cfg=dict(
    prior_file='data/pretrain/scannet_instance_size_priors.pt',
    num_layers=4,
    num_scale_tokens=8,
    num_heads=8,
    ffn_dims=1024,
    dropout=0.,
    spatial_downsample_ratio=2,
    detection_query_cross_attn_layers=[1, 2],
    score_thr=.1,
    fallback_topk=10,
    evidence_nms_iou_thr=.25,
    ema_momentum=.99,
    loss_weight=1.,
    layer_loss_weights=[.25, .5, .75, 1.]))

# Original model, data, trainability, and optimization settings are inherited.
# Fresh work directory separates joint fine-tuning from the original run.
work_dir = 'work_dirs/igdet_scannet_ipmsr'
