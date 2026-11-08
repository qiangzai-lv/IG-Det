"""Save activations to avoid cross-view reconstruction recomputation.

All model, data, dtype and optimization settings inherit the base config.
Use a separate work directory when comparing with the baseline.
"""

_base_ = ['./igdet_scannet.py']

model = dict(
    g_dino_cfg=dict(cross_view_checkpointing=False),
)
