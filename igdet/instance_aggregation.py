"""Anchor-based multiview instance aggregation without trainable parameters."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from igdet.rotated_geometry import pack_boxes, rotated_overlaps

class InstanceQueryAggregator(nn.Module):
    """Retrieve at most one observation per other view for each scene anchor.

    Matching uses aligned-scene boxes. Pooling uses the existing normalized
    VGGT centers and body-axis dimensions, preserving the decoder's contract.
    Anchors remain independent and may share support observations. All matching
    and pooled outputs are detached, as in the original query initializer.
    """

    def __init__(self, similarity_thr=0.6, iou_thr=0.1, max_supports=8,
                 anchor_weight=0.5):
        super().__init__()
        for name, value in [('similarity_thr', similarity_thr),
                            ('iou_thr', iou_thr)]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and within [0, 1]')
        if isinstance(max_supports, bool) or not isinstance(max_supports, int) \
                or max_supports < 1:
            raise ValueError('max_supports must be a positive integer')
        if not math.isfinite(anchor_weight) or not 0 < anchor_weight <= 1:
            raise ValueError('anchor_weight must be finite and within (0, 1]')
        self.similarity_thr = float(similarity_thr)
        self.iou_thr = float(iou_thr)
        self.max_supports = max_supports
        self.anchor_weight = float(anchor_weight)

    @torch.no_grad()
    def forward(self, selected, observations, num_views,
                anchor_centers_aligned, scene_scale):
        """Return selected fields plus support indices, masks and weights.

        ``selected`` is the NMS/FPS output. ``observations`` also contains the
        aligned NMS geometry and raw instance embeddings. Anchor geometry is
        supplied explicitly because an FPS anchor's center can differ from
        its source observation's predicted center.
        """
        device = selected['detection_query'].device
        with torch.autocast(device_type=device.type, enabled=False):
            return self._aggregate(
                selected, observations, num_views,
                anchor_centers_aligned, scene_scale)

    def _aggregate(self, selected, observations, num_views,
                   anchor_centers_aligned, scene_scale):
        batch_size, anchor_count, dims = selected['detection_query'].shape
        valid_mask = observations['valid_mask']
        if num_views < 1 or valid_mask.shape[0] != batch_size * num_views:
            raise ValueError('observation views must match selected scenes')
        query_count = valid_mask.shape[1]
        if query_count < 1:
            raise ValueError('observations require at least one query per view')
        count = num_views * query_count
        device = selected['detection_query'].device
        if anchor_centers_aligned.shape != (batch_size, anchor_count, 3):
            raise ValueError('aligned anchor centers must have shape [B, K, 3]')
        scale = torch.as_tensor(
            scene_scale, device=device, dtype=torch.float32).reshape(batch_size)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('scene scale must be finite and positive')
        sources = selected['source_indices']
        if sources.shape != (batch_size, anchor_count) or \
                ((sources < -1) | (sources >= count)).any():
            raise ValueError('anchor source indices must be -1 or valid observations')

        def scene_field(name):
            value = observations[name]
            if value.shape[:2] != (batch_size * num_views, query_count):
                raise ValueError(f'{name} must begin with [B * V, Q]')
            return value.detach().float().reshape(batch_size, count, -1)

        queries = scene_field('fused_detection_query')
        if queries.shape[-1] != dims:
            raise ValueError('observation and anchor feature dimensions differ')
        centers = scene_field('bbox_centers_vggt')
        sizes = scene_field('bbox_sizes_vggt')
        aligned_centers = scene_field('nms_centers_aligned')
        aligned_sizes = scene_field('nms_sizes_aligned')
        yaws = scene_field('bbox_yaw_aligned')
        embeddings = scene_field('instance_embeddings')
        foreground = scene_field('class_scores_2d').amax(dim=-1)
        norms = embeddings.norm(dim=-1)
        usable_embeddings = (torch.isfinite(embeddings).all(dim=-1)
                             & torch.isfinite(norms) & (norms > 1e-12))
        normalized = F.normalize(torch.where(
            usable_embeddings[..., None], embeddings, 0.), dim=-1, eps=1e-12)
        candidate_valid = selected['candidate_valid_mask'].bool().clone()
        if candidate_valid.shape != (batch_size, count):
            raise ValueError('candidate mask must have shape [B, V * Q]')
        candidate_valid &= valid_mask.reshape(batch_size, count).bool()
        candidate_valid &= usable_embeddings
        # Protect gather/pooling even if a caller supplies a broader mask.
        for value in [queries, centers, sizes, aligned_centers, aligned_sizes, yaws]:
            candidate_valid &= torch.isfinite(value).all(dim=-1)
        candidate_valid &= (sizes > 0).all(dim=-1)
        candidate_valid &= (aligned_sizes > 0).all(dim=-1)
        candidate_valid &= torch.isfinite(foreground) & (foreground > 0)

        output = dict(selected)
        for key in ['detection_query', 'query_xyz_vggt', 'query_size_vggt']:
            output[key] = selected[key].detach().clone()
        support_indices = sources.new_full(
            (batch_size, anchor_count, self.max_supports), -1)
        support_weights = queries.new_zeros(support_indices.shape)
        anchor_weights = queries.new_ones(batch_size, anchor_count)

        for batch_id in range(batch_size):
            pool = torch.nonzero(candidate_valid[batch_id], as_tuple=False).flatten()
            if pool.numel() == 0 or self.anchor_weight == 1:
                continue
            anchor_sources = sources[batch_id]
            safe_sources = anchor_sources.clamp_min(0)
            anchor_centers = selected['query_xyz_vggt'][batch_id].float()
            anchor_sizes = selected['query_size_vggt'][batch_id].float()
            anchor_queries = selected['detection_query'][batch_id].float()
            anchor_valid = ((anchor_sources >= 0)
                            & usable_embeddings[batch_id, safe_sources]
                            & torch.isfinite(anchor_centers).all(dim=-1)
                            & torch.isfinite(anchor_sizes).all(dim=-1)
                            & torch.isfinite(anchor_queries).all(dim=-1)
                            & torch.isfinite(anchor_centers_aligned[batch_id]).all(dim=-1)
                            & (anchor_sizes > 0).all(dim=-1))
            similarities = (normalized[batch_id, safe_sources]
                            @ normalized[batch_id, pool].t()).clamp(-1, 1)
            anchor_boxes = pack_boxes(
                anchor_centers_aligned[batch_id].float(),
                anchor_sizes * scale[batch_id], selected['query_yaw_aligned'][batch_id])
            anchor_boxes = torch.where(anchor_valid[:, None], anchor_boxes,
                                       anchor_boxes.new_tensor([0,0,0,1,1,1,0]))
            candidate_boxes = pack_boxes(
                aligned_centers[batch_id,pool], aligned_sizes[batch_id,pool], yaws[batch_id,pool])
            ious = rotated_overlaps(anchor_boxes, candidate_boxes)
            different_view = (anchor_sources[:, None] // query_count
                              != pool[None] // query_count)
            matches = (anchor_valid[:, None] & different_view
                       & (similarities >= self.similarity_thr)
                       & (ious >= self.iou_thr))
            affinity = foreground[batch_id, pool][None] * similarities.clamp_min(0) * ious
            affinity = torch.where(matches, affinity, 0.)
            # Restore [view, query] indexing for a vectorized top-1 per view.
            per_query = queries.new_zeros(anchor_count, count)
            per_query.scatter_(1, pool[None].expand(anchor_count, -1), affinity)
            view_scores, view_queries = per_query.reshape(
                anchor_count, num_views, query_count).max(dim=-1)
            limit = min(self.max_supports, num_views)
            view_order = view_scores.argsort(
                dim=-1, descending=True, stable=True)[:, :limit]
            scores = view_scores.gather(1, view_order)
            indices = (view_order * query_count
                       + view_queries.gather(1, view_order))
            mask = scores > 0
            support_indices[batch_id, :, :limit] = torch.where(mask, indices, -1)
            active = mask.any(dim=-1)
            denominator = torch.where(
                active[:, None], scores.sum(dim=-1, keepdim=True), 1.)
            weights = scores / denominator
            weights *= 1. - self.anchor_weight
            support_weights[batch_id, :, :limit] = weights
            anchor_weights[batch_id, active] = self.anchor_weight
            if not active.any():
                continue

            # Only update matched anchors. No-match outputs stay bitwise
            # identical, without a needless log/exp roundtrip on their sizes.
            indices = indices[active]
            mask = mask[active, :, None]
            weights = weights[active, :, None]
            support_queries = torch.where(mask, queries[batch_id, indices], 0.)
            support_centers = torch.where(mask, centers[batch_id, indices], 0.)
            support_sizes = torch.where(mask, sizes[batch_id, indices], 1.)
            pooled_queries = (self.anchor_weight * anchor_queries[active]
                              + (weights * support_queries).sum(dim=1))
            pooled_centers = (self.anchor_weight * anchor_centers[active]
                              + (weights * support_centers).sum(dim=1))
            pooled_sizes = (self.anchor_weight * anchor_sizes[active].log()
                            + (weights * support_sizes.log()).sum(dim=1)).exp()
            for key, value in [('detection_query', pooled_queries),
                               ('query_xyz_vggt', pooled_centers),
                               ('query_size_vggt', pooled_sizes)]:
                output[key][batch_id, active] = value.to(output[key].dtype)

        output['instance_aggregation'] = dict(
            support_indices=support_indices,
            support_mask=support_indices >= 0,
            support_weights=support_weights,
            anchor_weights=anchor_weights,
            support_counts=(support_indices >= 0).sum(dim=-1))
        return output
