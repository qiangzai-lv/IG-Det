"""2D-only checkpoint restoration and loss scaling for joint 3D training."""
import math

from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config


RECONSTRUCTION_PREFIXES = ('reconstruction_decoder.', 'bbox_head.reconstruction_head.')
DETECTION_LOSS_NAMES = {
    'loss_cls', 'loss_bbox', 'loss_iou',
    'enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou',
    'dn_loss_cls', 'dn_loss_bbox', 'dn_loss_iou',
}


def is_2d_detector_state(name):
    return not name.startswith(RECONSTRUCTION_PREFIXES)


def require_complete_state(source, expected, label):
    missing = sorted(set(expected) - set(source))
    mismatched = [f'{k}: {tuple(source[k].shape)} != {tuple(v.shape)}'
                  for k, v in expected.items()
                  if k in source and source[k].shape != v.shape]
    if missing or mismatched:
        raise ValueError(
            f'Incomplete {label} checkpoint; missing={missing[:10]}, '
            f'shape mismatches={mismatched[:10]}. '
            'Use the VGGT 2D pretraining checkpoint, not the old Swin checkpoint.')


def read_2d_checkpoint(path, lora_cfg=None, classes=None,
                       pyramid_cfg=None, exchange_cfg=None,
                       input_image_size=(336, 448)):
    payload = CheckpointLoader.load_checkpoint(str(path), map_location='cpu')
    source_config = payload.get('meta', {}).get('cfg')
    if source_config is None:
        raise ValueError('VGGT 2D checkpoint requires meta.cfg to verify LoRA scaling and classes')
    source = Config.fromstring(source_config, '.py').model
    if not source.get('vggt_checkpoint') or source.get('vggt_omega_checkpoint'):
        raise ValueError('2D checkpoint must use native VGGT, not VGGT-Omega')
    if tuple(source.get('input_image_size', ())) != tuple(input_image_size):
        raise ValueError('2D checkpoint input_image_size differs from joint training')
    if not source.get('train_2d_only', False) or not source.get('detection_pyramid_cfg'):
        raise ValueError('Checkpoint must come from VGGT 2D pretraining')
    if classes is not None and tuple(source.g_dino_cfg.semantic_classes) != tuple(classes):
        raise ValueError('2D checkpoint semantic class ordering differs from joint training')
    if lora_cfg is not None:
        for key in ('enabled', 'block_indices', 'branches', 'target_modules',
                    'rank', 'alpha', 'dropout'):
            original = source.vggt_lora_cfg.get(key)
            current = lora_cfg.get(key)
            if isinstance(original, (tuple, list)):
                original, current = tuple(original), tuple(current or ())
            if original != current:
                raise ValueError(f'2D checkpoint VGGT LoRA {key} mismatch: {original} != {current}')
    for key, expected in (('detection_pyramid_cfg', pyramid_cfg),
                          ('scene_query_exchange_cfg', exchange_cfg)):
        if expected is not None and dict(source.get(key) or {}) != dict(expected):
            raise ValueError(f'2D checkpoint {key} differs from joint training')
    state = payload.get('state_dict', payload)
    return {name.removeprefix('module.'): value for name, value in state.items()}


def restore_vggt_2d_components(vggt_encoder, pyramid, state):
    """Restore only LoRA + detection pyramid; leave camera/3D weights alone."""
    expected_lora = {k: v for k, v in vggt_encoder.state_dict().items()
                     if '.lora_a.' in k or '.lora_b.' in k}
    if not expected_lora:
        raise ValueError('Joint 2D retention requires enabled VGGT LoRA')
    lora = {k.removeprefix('vggt_encoder.'): v for k, v in state.items()
            if k.startswith('vggt_encoder.') and
            ('.lora_a.' in k or '.lora_b.' in k)}
    # Exact adapter topology matters: silently dropping trained adapters changes
    # the frozen detector's input distribution at the start of 3D training.
    if set(lora) != set(expected_lora):
        raise ValueError('VGGT LoRA keys differ from the 2D checkpoint configuration')
    require_complete_state(lora, expected_lora, 'VGGT LoRA')
    prefix = 'detection_feature_projector.'
    detection = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    require_complete_state(detection, pyramid.state_dict(), '2D detection pyramid')
    # Validate both components before modifying either one.
    if set(detection) != set(pyramid.state_dict()):
        raise ValueError('Unexpected detection pyramid keys in 2D checkpoint')
    vggt_encoder.load_state_dict(lora, strict=False)
    pyramid.load_state_dict(detection, strict=True)


def scale_retention_losses(losses, detection_weight):
    """Scale only native G-DINO 2D detection losses, never 3D terms or stats.

    Instance consistency already carries instance_consistency_cfg.loss_weight.
    Keep its loss key and internal weight unchanged (no second factor of 0.1).
    """
    if not math.isfinite(detection_weight) or detection_weight < 0:
        raise ValueError('2D detection weight must be finite and nonnegative')
    result = dict(losses)
    for name, value in losses.items():
        if name.rsplit('.', 1)[-1] in DETECTION_LOSS_NAMES:
            result[name] = ([v * detection_weight for v in value]
                            if isinstance(value, list) else value * detection_weight)
    return result
