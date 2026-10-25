"""Save activations to avoid VGGT and cross-view reconstruction recomputation.

All model, data, dtype, LoRA and optimization settings inherit the base config.
Use a separate work directory when comparing with the baseline.
"""

_base_ = ['./igdet_scannet.py']

model = dict(
    vggt_lora_cfg=dict(gradient_checkpointing=False),
    g_dino_cfg=dict(cross_view_checkpointing=False),
)
