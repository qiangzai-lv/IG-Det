import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from igdet.rotated_geometry import pack_boxes, rotated_overlaps


def _ensure_finite(name, value):
    finite = torch.isfinite(value)
    if finite.all():
        return
    invalid_count = int((~finite).sum().item())
    finite_values = value[finite]
    finite_range = 'no finite values'
    if finite_values.numel() > 0:
        finite_range = (
            f'finite_min={finite_values.min().item():.6g}, '
            f'finite_max={finite_values.max().item():.6g}')
    raise FloatingPointError(
        f'{name} contains {invalid_count}/{value.numel()} non-finite values; '
        f'{finite_range}')


class RepeatedHungarianMatcher(nn.Module):
    """Hungarian matcher with repeated GTs and explicit log-size cost."""

    def __init__(self,
                 cost_weights=None,
                 gt_repeat_num=5,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        if cost_weights is None:
            cost_weights = dict(cls=2.0, center=1.0, size=1.0, giou=2.0)
        required = {'cls', 'center', 'size', 'giou'}
        if set(cost_weights) != required:
            raise ValueError(f'cost_weights must contain exactly {required}')
        if gt_repeat_num < 1:
            raise ValueError('gt_repeat_num must be positive')
        self.cost_weights = dict(cost_weights)
        self.gt_repeat_num = int(gt_repeat_num)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def _focal_cost(self, logits, gt_labels):
        probability = logits.sigmoid()
        eps = torch.finfo(probability.dtype).eps
        negative = ((1.0 - self.focal_alpha)
                    * probability.pow(self.focal_gamma)
                    * -(1.0 - probability + eps).log())
        positive = (self.focal_alpha
                    * (1.0 - probability).pow(self.focal_gamma)
                    * -(probability + eps).log())
        return positive[:, gt_labels] - negative[:, gt_labels]

    @staticmethod
    def _center_size_pred_to_bbox(centers, sizes):
        half_size = sizes / 2.0
        return torch.cat((centers - half_size, centers + half_size), dim=-1)

    @torch.no_grad()
    def _get_targets(self, pred_centers, pred_sizes, pred_size_logs,
                     pred_logits, gt_centers, gt_sizes, gt_labels, pred_yaw, gt_yaw):
        num_queries = pred_centers.shape[0]
        num_gt = gt_centers.shape[0]
        empty = torch.empty(0, dtype=torch.long, device=pred_centers.device)
        if num_queries == 0 or num_gt == 0:
            return empty, empty

        for name, value in (
                ('predicted centers', pred_centers),
                ('predicted sizes', pred_sizes),
                ('predicted log sizes', pred_size_logs),
                ('classification logits', pred_logits),
                ('ground-truth centers', gt_centers),
                ('ground-truth sizes', gt_sizes),
                ('predicted yaw', pred_yaw), ('ground-truth yaw', gt_yaw)):
            _ensure_finite(name, value)
        if (pred_sizes <= 0).any() or (gt_sizes <= 0).any():
            raise ValueError('predicted and ground-truth sizes must be positive')

        output_device = pred_centers.device
        pred_centers = pred_centers.float()
        pred_sizes = pred_sizes.float()
        pred_size_logs = pred_size_logs.float()
        pred_logits = pred_logits.float()
        gt_centers = gt_centers.float()
        gt_sizes = gt_sizes.float()

        repeat_num = self.gt_repeat_num
        repeated_gt_indices = torch.arange(
            num_gt, device=gt_centers.device).repeat(repeat_num)
        repeated_centers = gt_centers[repeated_gt_indices]
        repeated_sizes = gt_sizes[repeated_gt_indices]
        repeated_labels = gt_labels[repeated_gt_indices]

        cost_center = torch.cdist(pred_centers, repeated_centers, p=1)
        cost_size = torch.cdist(
            pred_size_logs, repeated_sizes.clamp_min(1e-5).log(), p=1)
        cost_class = self._focal_cost(pred_logits, repeated_labels)

        pred_boxes = pack_boxes(pred_centers, pred_sizes, pred_yaw.float())
        gt_boxes = pack_boxes(repeated_centers, repeated_sizes, gt_yaw.float()[repeated_gt_indices])
        giou = rotated_overlaps(pred_boxes, gt_boxes, mode='giou')
        total_cost = (
            self.cost_weights['cls'] * cost_class
            + self.cost_weights['center'] * cost_center
            + self.cost_weights['size'] * cost_size
            - self.cost_weights['giou'] * giou)
        _ensure_finite('Hungarian cost matrix', total_cost)

        pred_indices, repeated_indices = linear_sum_assignment(
            total_cost.cpu().numpy())
        pred_indices = torch.from_numpy(pred_indices).long().to(
            output_device)
        repeated_indices = torch.from_numpy(repeated_indices).long().to(
            output_device)
        return pred_indices, repeated_gt_indices.to(output_device)[
            repeated_indices]
