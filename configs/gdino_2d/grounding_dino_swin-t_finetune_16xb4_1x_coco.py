_base_ = './grounding_dino_swin-t_pretrain_obj365.py'

# Single RGB images and saved 2D boxes; no IGDet multi-view/3D branches.
dataset_type = 'CocoDataset'
data_root = 'data/'
scannet_ann_root = 'scannet_coco_v2_new/'
scannet_image_root = 'ScanNet_processed_v2/'
# Names must match categories[].name in the generated COCO JSON exactly.
classes = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin')
metainfo = dict(classes=classes)

model = dict(
    # Set to e.g. 64; loading a larger query embedding keeps its first N rows.
    num_queries=128,
    freeze_modules=['language_model', 'backbone'],
    language_model=dict(name='data/pretrain/bert-base-uncased'),
    backbone=dict(with_cp=False, init_cfg=None),
    encoder=dict(num_cp=0))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        ann_file=scannet_ann_root + 'keypoints_bbox_train.json',
        metainfo=metainfo,
        data_prefix=dict(img=scannet_image_root),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

# learning policy
max_epochs = 12
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# Explicit validation settings prevent inheritance of COCO-80 annotations.
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
         backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
        'text', 'custom_entities'))
]
val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        ann_file=scannet_ann_root + 'keypoints_bbox_val.json',
        data_prefix=dict(img=scannet_image_root),
        metainfo=metainfo,
        return_classes=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=data_root + scannet_ann_root + 'keypoints_bbox_val.json',
    metric='bbox')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(
    max_keep_ckpts=4, save_best='coco/bbox_mAP', rule='greater'))

# Local full detector weights also initialize the Swin backbone.
load_from = ('data/pretrain/'
             'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_'
             '20231204_095047-b448804b.pth')
auto_scale_lr = dict(enable=False, base_batch_size=64)
find_unused_parameters = True

# Use the available backend on the training server.
import torch
try:
    import torch_npu  # noqa: F401
    _dist_backend_ = 'hccl' if torch.npu.is_available() else 'nccl'
except ImportError:
    _dist_backend_ = 'nccl'
env_cfg = dict(dist_cfg=dict(backend=_dist_backend_))
