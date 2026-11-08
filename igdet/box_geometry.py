"""Scale-independent overlap tests for proposal selection and aggregation."""

import torch

from mmdet3d.structures.ops.iou3d_calculator import (
    axis_aligned_bbox_overlaps_3d)


def axis_aligned_iou(boxes1, boxes2):
    """Compute FP32 IoU without a coordinate-unit-dependent volume floor.

    A fixed epsilon changes the IoU of small boxes when scene units change.
    Positive unions use their actual volume; zero-volume pairs return zero.
    Callers use detached geometry, so no gradient through degenerate ratios is
    needed. The global metric/loss overlap implementation remains unchanged.
    """
    with torch.autocast(device_type=boxes1.device.type, enabled=False):
        iou = axis_aligned_bbox_overlaps_3d(
            boxes1.float(), boxes2.float(), mode='iou',
            is_aligned=False, eps=0.)
        return torch.nan_to_num(iou, nan=0., posinf=0., neginf=0.)
