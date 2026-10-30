_base_ = ['./grounding_dino_swin-t_pretrain_obj365.py']

model = dict(
    backbone=dict(_delete_=True, type='ExternalFeatureBackbone'),
    neck=None,
    encoder=dict(num_cp=0),
    reconstruction_decoder=None,
    freeze_modules=['language_model'])
