"""Disable only the project's cross-view reconstruction recomputation.

Native VGGT still checkpoints its attention blocks in training mode. This
variant does not override that behavior or change its source code.
"""

_base_ = ['./igdet_scannet.py']

model = dict(
    g_dino_cfg=dict(cross_view_checkpointing=False),
)
