import numpy as np
import torch
from igdet.box_geometry import axis_aligned_iou


@torch.no_grad()
def _aligned_3d_nms(centers, sizes, scores, classes, threshold):
    """Class-aware volume NMS using device-native PyTorch AABB overlaps.

    Keep the legacy ascending-argsort/reverse traversal, including score ties.
    Compute IoUs in row chunks and perform greedy suppression on CPU masks,
    avoiding a device synchronization for every selected box. Only geometry and
    indices participate; query features stay on their original device.
    Geometry is float32 with no unit-dependent union-volume floor. The
    axis-aligned formula has no rotated-BEV intersection tolerance.
    """
    if centers.shape[0] == 0:
        return centers.new_empty((0,), dtype=torch.long)
    order = scores.argsort().flip(0)
    # Use the same min/max convention as final detection NMS. Convert before
    # computing corners so half/bfloat16 inputs do not lose more precision.
    centers = centers.float()
    half_sizes = sizes.float() / 2.0
    boxes = torch.cat([
        centers - half_sizes, centers + half_sizes
    ], dim=-1)[order].contiguous()
    sorted_classes = classes[order].cpu().numpy()
    kept = np.zeros(len(centers), dtype=np.bool_)
    for label in np.unique(sorted_classes):
        positions = np.flatnonzero(sorted_classes == label)
        class_boxes = boxes[torch.as_tensor(positions, device=boxes.device)]
        count = len(positions)
        suppression = np.empty((count, count), dtype=np.bool_)
        # Bound float32 IoU intermediates, even if all 8192 candidates agree
        # on the class (as happens at initialization).
        for start in range(0, count, 512):
            stop = min(start + 512, count)
            iou = axis_aligned_iou(class_boxes[start:stop], class_boxes)
            suppression[start:stop] = (iou > threshold).cpu().numpy()
        suppressed = np.zeros(count, dtype=np.bool_)
        for index in range(count):
            if not suppressed[index]:
                kept[positions[index]] = True
                suppressed |= suppression[index]

    keep_positions = torch.as_tensor(np.flatnonzero(kept), device=order.device)
    return order[keep_positions]


def _weighted_fps(points, scores, selected_points, count):
    """Select distant, high-score candidates, deterministically with repeats."""
    if count <= 0:
        return points.new_empty((0,), dtype=torch.long)
    if points.shape[0] == 0:
        raise ValueError('weighted FPS requires at least one candidate')
    weights = scores.float().clamp_min(0)
    selected = selected_points.float()
    remaining = torch.ones(
        points.shape[0], dtype=torch.bool, device=points.device)
    chosen = []
    min_distance = torch.full((points.shape[0],), float('inf'),
                              device=points.device, dtype=torch.float32)
    if selected.numel():
        min_distance = torch.cdist(
            points.float(), selected).square().amin(dim=1)
    else:
        first = weights.argmax()
        chosen.append(first)
        remaining[first] = False
        min_distance = torch.cdist(
            points.float(), points[first:first + 1]).square().squeeze(1)
    for _ in range(min(count, points.shape[0])):
        if len(chosen) >= min(count, points.shape[0]):
            break
        priority = min_distance * weights
        priority[~remaining] = -torch.inf
        index = priority.argmax()
        chosen.append(index)
        remaining[index] = False
        min_distance = torch.minimum(
            min_distance,
            torch.cdist(
                points.float(), points[index:index + 1]).square().squeeze(1))
    if len(chosen) < count:
        repeats = torch.tensor(chosen, device=points.device, dtype=torch.long)
        chosen.extend(repeats[torch.arange(
            count - len(chosen), device=points.device) % len(repeats)].tolist())
    return torch.tensor(chosen, device=points.device, dtype=torch.long)


def select_reconstruction_boxes(
        reconstruction_outputs, batch_size, num_views, num_queries,
        nms_iou_thr, logger, scene_ids=None,
        foreground_score_thr=0.1):
    """Select fixed-count scene proposals from reconstruction AABBs."""
    per_query_keys = (
        'bbox_centers_vggt', 'bbox_sizes_vggt', 'bbox_scores',
        'bbox_labels', 'class_scores_2d', 'fused_detection_query',
        'points_vggt', 'valid_mask', 'iou_centers_normalized',
        'iou_sizes_normalized')
    required = per_query_keys
    missing = [key for key in required if key not in reconstruction_outputs]
    if missing:
        raise KeyError(f'Missing reconstruction outputs: {missing}')
    if min(batch_size, num_views, num_queries) <= 0:
        raise ValueError('batch_size, num_views and num_queries must be positive')
    if not 0 <= nms_iou_thr <= 1:
        raise ValueError('NMS IoU threshold must be within [0, 1]')
    if not 0 <= foreground_score_thr <= 1:
        raise ValueError('foreground score threshold must be within [0, 1]')

    valid_mask = reconstruction_outputs['valid_mask']
    expected_views = batch_size * num_views
    if valid_mask.ndim != 2 or valid_mask.shape[0] != expected_views:
        raise ValueError('valid_mask must have shape [B * V, Q]')
    query_count = valid_mask.shape[1]
    query_dims = reconstruction_outputs['fused_detection_query'].shape[-1]
    expected_prefix = (expected_views, query_count)
    for key in per_query_keys:
        if reconstruction_outputs[key].shape[:2] != expected_prefix:
            raise ValueError(f'{key} must begin with [B * V, Q]')

    centers = reconstruction_outputs['bbox_centers_vggt'].reshape(
        batch_size, num_views * query_count, 3)
    sizes = reconstruction_outputs['bbox_sizes_vggt'].reshape_as(centers)
    nms_centers = reconstruction_outputs['iou_centers_normalized'].reshape_as(
        centers)
    nms_sizes = reconstruction_outputs['iou_sizes_normalized'].reshape_as(centers)
    scores = reconstruction_outputs['bbox_scores'].reshape(
        batch_size, num_views * query_count)
    labels = reconstruction_outputs['bbox_labels'].reshape_as(scores).long()
    queries = reconstruction_outputs['fused_detection_query'].reshape(
        batch_size, num_views * query_count, query_dims)
    reconstruction_points = reconstruction_outputs['points_vggt'].reshape(
        batch_size, num_views * query_count, 3)
    class_scores_2d = reconstruction_outputs['class_scores_2d'].reshape(
        batch_size, num_views * query_count, -1)
    if class_scores_2d.shape[-1] == 0:
        raise ValueError('class_scores_2d must contain at least one class')
    foreground_scores_2d = class_scores_2d.amax(dim=-1)
    valid = valid_mask.reshape(
        batch_size, num_views * query_count).bool().clone()

    valid &= torch.isfinite(centers).all(dim=-1)
    valid &= torch.isfinite(sizes).all(dim=-1)
    valid &= torch.isfinite(nms_centers).all(dim=-1)
    valid &= torch.isfinite(nms_sizes).all(dim=-1)
    valid &= torch.isfinite(scores)
    valid &= torch.isfinite(queries).all(dim=-1)
    valid &= torch.isfinite(reconstruction_points).all(dim=-1)
    valid &= (sizes > 0).all(dim=-1)
    valid &= (nms_sizes > 0).all(dim=-1)
    valid &= torch.isfinite(foreground_scores_2d)
    valid &= foreground_scores_2d >= foreground_score_thr
    fallback_valid = valid_mask.reshape(
        batch_size, num_views * query_count).bool().clone()
    fallback_valid &= torch.isfinite(sizes).all(dim=-1)
    fallback_valid &= (sizes > 0).all(dim=-1)
    fallback_valid &= torch.isfinite(queries).all(dim=-1)
    fallback_valid &= torch.isfinite(reconstruction_points).all(dim=-1)
    fallback_valid &= torch.isfinite(foreground_scores_2d)

    output_centers = []
    output_sizes = []
    output_queries = []
    output_scores = []
    output_labels = []
    output_source_indices = []
    output_fallback_masks = []
    for batch_id in range(batch_size):
        candidate_indices = torch.nonzero(
            valid[batch_id], as_tuple=False).flatten()
        candidate_scores = scores[batch_id, candidate_indices].detach()
        if candidate_indices.numel():
            local_keep = _aligned_3d_nms(
                nms_centers[batch_id, candidate_indices].detach(),
                nms_sizes[batch_id, candidate_indices].detach(),
                candidate_scores,
                labels[batch_id, candidate_indices].detach(), nms_iou_thr)
            local_keep = local_keep[candidate_scores[local_keep].argsort(
                descending=True)]
            selected_indices = candidate_indices[local_keep[:num_queries]]
        else:
            local_keep = candidate_indices
            selected_indices = candidate_indices

        real_count = len(selected_indices)
        fallback_count = num_queries - real_count
        selected_centers = centers[batch_id, selected_indices].detach()
        selected_sizes = sizes[batch_id, selected_indices].detach()
        selected_queries = queries[batch_id, selected_indices].detach()
        selected_scores = scores[batch_id, selected_indices].detach()
        selected_labels = labels[batch_id, selected_indices].detach()
        source_indices = selected_indices.detach()
        fallback_mask = torch.zeros(real_count, dtype=torch.bool,
                                    device=centers.device)

        if fallback_count:
            fallback_pool = candidate_indices
            if fallback_pool.numel() == 0:
                fallback_pool = torch.nonzero(
                    fallback_valid[batch_id], as_tuple=False).flatten()
            remaining = fallback_pool[
                ~torch.isin(fallback_pool, selected_indices)]
            if remaining.numel() == 0:
                remaining = fallback_pool
            if remaining.numel() == 0:
                scene_id = (
                    batch_id if scene_ids is None else scene_ids[batch_id])
                raise RuntimeError(
                    'No usable reconstruction observations for weighted FPS: '
                    f'scene={scene_id}')
            fallback_indices = _weighted_fps(
                reconstruction_points[batch_id, remaining],
                foreground_scores_2d[batch_id, remaining],
                reconstruction_points[batch_id, selected_indices],
                fallback_count)
            fallback_source_indices = remaining[fallback_indices]
            fallback_centers = reconstruction_points[
                batch_id, fallback_source_indices].detach()
            fallback_sizes_for_scene = sizes[
                batch_id, fallback_source_indices].detach()
            fallback_queries = queries[
                batch_id, fallback_source_indices].detach()
            fallback_scores = scores[
                batch_id, fallback_source_indices].detach()
            fallback_scores = torch.where(
                torch.isfinite(fallback_scores), fallback_scores,
                foreground_scores_2d[
                    batch_id, fallback_source_indices].detach())
            fallback_labels = labels[
                batch_id, fallback_source_indices].detach()
            selected_centers = torch.cat(
                [selected_centers, fallback_centers], dim=0)
            selected_sizes = torch.cat(
                [selected_sizes, fallback_sizes_for_scene], dim=0)
            selected_queries = torch.cat([
                selected_queries, fallback_queries
            ], dim=0)
            selected_scores = torch.cat([
                selected_scores, fallback_scores
            ])
            selected_labels = torch.cat([
                selected_labels, fallback_labels
            ])
            source_indices = torch.cat([
                source_indices, fallback_source_indices.detach()
            ])
            fallback_mask = torch.cat([fallback_mask, torch.ones(
                fallback_count, dtype=torch.bool, device=centers.device)])
            scene_id = batch_id if scene_ids is None else scene_ids[batch_id]
            logger.warning(
                'Reconstruction proposals required weighted FPS fallback: '
                f'scene={scene_id} valid_candidates={len(candidate_indices)} '
                f'nms_keep={len(local_keep)} '
                f'weighted_fps_fallback={fallback_count} '
                f'random_fallback=0 num_queries={num_queries}')

        output_centers.append(selected_centers)
        output_sizes.append(selected_sizes)
        output_queries.append(selected_queries)
        output_scores.append(selected_scores)
        output_labels.append(selected_labels)
        output_source_indices.append(source_indices)
        output_fallback_masks.append(fallback_mask)

    return {
        'query_xyz_vggt': torch.stack(output_centers),
        'query_size_vggt': torch.stack(output_sizes),
        'detection_query': torch.stack(output_queries),
        'scores': torch.stack(output_scores),
        'labels': torch.stack(output_labels),
        'source_indices': torch.stack(output_source_indices),
        'candidate_valid_mask': valid,
        'fallback_mask': torch.stack(output_fallback_masks),
    }
