"""Validate and restore the frozen 2D detector used by 3D training."""
from pathlib import Path

from mmengine.config import Config
from mmengine.runner.checkpoint import CheckpointLoader


RECONSTRUCTION_PREFIXES = ('reconstruction_decoder.', 'bbox_head.reconstruction_head.')


def is_2d_detector_state(name):
    return not name.startswith(RECONSTRUCTION_PREFIXES)


def require_complete_state(source, expected, label):
    missing = sorted(set(expected) - set(source))
    mismatched = [f'{key}: {tuple(source[key].shape)} != {tuple(value.shape)}'
                  for key, value in expected.items()
                  if key in source and source[key].shape != value.shape]
    if missing or mismatched:
        raise ValueError(
            f'Incomplete {label} checkpoint; missing={missing[:10]}, '
            f'shape mismatches={mismatched[:10]}. '
            'Use the complete VGGT 2D pretraining checkpoint.')


def read_2d_checkpoint(path, classes=None, pyramid_cfg=None,
                       exchange_cfg=None, backbone_checkpoint=None):
    payload = CheckpointLoader.load_checkpoint(str(path), map_location='cpu')
    source_config = payload.get('meta', {}).get('cfg')
    if source_config is None:
        raise ValueError('VGGT 2D checkpoint requires meta.cfg to verify the frozen backbone and classes')
    source = Config.fromstring(source_config, '.py').model
    if not source.get('train_2d_only', False) or not source.get('detection_pyramid_cfg'):
        raise ValueError('Checkpoint must come from VGGT 2D pretraining')
    state = payload.get('state_dict', payload)
    state = {name.removeprefix('module.'): value for name, value in state.items()}
    if (source.get('vggt_lora_cfg', {}).get('enabled', False) or
            any('.lora_a.' in name or '.lora_b.' in name for name in state)):
        raise ValueError('Checkpoint uses VGGT LoRA; rerun 2D pretraining with the frozen original backbone')
    if classes is not None and tuple(source.g_dino_cfg.semantic_classes) != tuple(classes):
        raise ValueError('2D checkpoint semantic class ordering differs from 3D training')
    if backbone_checkpoint is not None:
        original = source.get('vggt_omega_checkpoint')
        if (original is None or Path(original).expanduser().resolve() !=
                Path(backbone_checkpoint).expanduser().resolve()):
            raise ValueError('2D checkpoint uses a different frozen VGGT backbone')
    for key, expected in (('detection_pyramid_cfg', pyramid_cfg),
                          ('scene_query_exchange_cfg', exchange_cfg)):
        if expected is not None and dict(source.get(key) or {}) != dict(expected):
            raise ValueError(f'2D checkpoint {key} differs from 3D training')
    return state


def restore_detection_pyramid(pyramid, state):
    prefix = 'detection_feature_projector.'
    detection = {key[len(prefix):]: value for key, value in state.items()
                 if key.startswith(prefix)}
    require_complete_state(detection, pyramid.state_dict(), '2D detection pyramid')
    if set(detection) != set(pyramid.state_dict()):
        raise ValueError('Unexpected detection pyramid keys in 2D checkpoint')
    pyramid.load_state_dict(detection, strict=True)
