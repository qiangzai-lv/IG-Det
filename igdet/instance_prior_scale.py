"""Instance-prior metric scale recovery in the detector's normalized gauge."""
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from igdet.camera_alignment import rotate_vggt_points_to_scene_axes
from igdet.query_correspondence import _aligned_3d_nms


class ScaleDecoderLayer(nn.Module):
    def __init__(self, dims, heads, ffn_dims, dropout, detection_attention):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dims, heads, dropout=dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dims, heads, dropout=dropout, batch_first=True)
        self.detection_attn = (nn.MultiheadAttention(dims, heads, dropout=dropout, batch_first=True)
                               if detection_attention else None)
        self.self_norm = nn.LayerNorm(dims)
        self.spatial_norm = nn.LayerNorm(dims)
        self.detection_norm = nn.LayerNorm(dims) if detection_attention else None
        self.ffn_norm = nn.LayerNorm(dims)
        self.ffn = nn.Sequential(nn.Linear(dims, ffn_dims), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dims, dims))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, detection_query, detection_padding_mask):
        normalized = self.self_norm(query)
        query = query + self.dropout(self.self_attn(
            normalized, normalized, normalized, need_weights=False)[0])
        query = query + self.dropout(self.spatial_attn(
            self.spatial_norm(query), memory, memory, need_weights=False)[0])
        if self.detection_attn is not None:
            query = query + self.dropout(self.detection_attn(
                self.detection_norm(query), detection_query, detection_query,
                key_padding_mask=detection_padding_mask, need_weights=False)[0])
        return query + self.dropout(self.ffn(self.ffn_norm(query)))


class InstancePriorScaleRecovery(nn.Module):
    """One shared positive scale per scene, refined by eight tokens/four layers.

    Size priors are fixed metric dimensions in aligned-scene axis order. EMA
    prototypes use final detection queries, grouped once per GT instance. Only
    ``loss`` and ``update_prototypes`` accept ground truth; ``forward`` cannot
    access GT scale. Original detector parameters retain their training policy.
    """
    def __init__(self, prior_file, classes, embed_dims=512, num_layers=4,
                 num_scale_tokens=8, num_heads=8, ffn_dims=1024, dropout=0.,
                 spatial_downsample_ratio=2,
                 detection_query_cross_attn_layers=(1, 2), score_thr=.1,
                 fallback_topk=10, evidence_nms_iou_thr=.25,
                 ema_momentum=.99, loss_weight=1.,
                 layer_loss_weights=(.25, .5, .75, 1.)):
        super().__init__()
        if embed_dims <= 0 or embed_dims % num_heads:
            raise ValueError('embed_dims must be positive and divisible by num_heads')
        if num_layers < 1 or num_scale_tokens < 1 or spatial_downsample_ratio < 1:
            raise ValueError('layers, tokens and downsample ratio must be positive')
        if len(layer_loss_weights) != num_layers or any(w < 0 for w in layer_loss_weights):
            raise ValueError('one nonnegative loss weight is required per scale layer')
        if not 0 <= score_thr <= 1 or fallback_topk < 1 or not 0 <= evidence_nms_iou_thr <= 1:
            raise ValueError('invalid evidence selection settings')
        if not 0 <= ema_momentum < 1 or loss_weight < 0:
            raise ValueError('invalid EMA momentum or loss weight')
        cross_layers = set(detection_query_cross_attn_layers)
        if any(i < 0 or i >= num_layers for i in cross_layers):
            raise ValueError('detection cross-attention layer index is out of range')
        prior = torch.load(Path(prior_file), map_location='cpu', weights_only=True)
        if prior.get('source_split') != 'train':
            raise ValueError('size priors must be built from training annotations only')
        if list(prior.get('classes', [])) != list(classes):
            raise ValueError('size-prior classes/order differ from the detector')
        sizes = torch.as_tensor(prior['metric_size_priors'], dtype=torch.float32)
        valid = torch.as_tensor(prior['valid_modes'], dtype=torch.bool)
        counts = torch.as_tensor(prior['mode_counts'], dtype=torch.long)
        fixed = torch.as_tensor(prior.get('fixed_prior_mask', torch.zeros_like(valid)), dtype=torch.bool)
        replicated = torch.as_tensor(
            prior.get('replicated_prior_mask', torch.zeros_like(valid)), dtype=torch.bool)
        if (sizes.ndim != 3 or sizes.shape[0] != len(classes) or sizes.shape[-1] != 3
                or valid.shape != sizes.shape[:2] or counts.shape != valid.shape
                or fixed.shape != valid.shape or replicated.shape != valid.shape):
            raise ValueError('invalid prior tensor shapes')
        if (not valid.any(dim=1).all() or not torch.isfinite(sizes).all()
                or (sizes <= 0).any() or (counts < 0).any()
                or (counts[valid & ~fixed & ~replicated] <= 0).any()
                or ((fixed | replicated) & ~valid).any()
                or (fixed & replicated).any()
                or (counts[fixed | replicated] != 0).any()):
            raise ValueError('every category requires finite positive populated size priors')
        self.classes = tuple(classes)
        self.register_buffer('metric_size_priors', sizes)
        self.register_buffer('valid_modes', valid)
        self.register_buffer('mode_counts', counts)
        self.register_buffer('fixed_prior_mask', fixed)
        self.register_buffer('replicated_prior_mask', replicated)
        self.register_buffer('query_prototypes', torch.zeros(*valid.shape, embed_dims))
        self.register_buffer('prototype_counts', torch.zeros_like(counts))
        self.embed_dims = embed_dims
        self.spatial_downsample_ratio = spatial_downsample_ratio
        self.score_thr = score_thr
        self.fallback_topk = fallback_topk
        self.evidence_nms_iou_thr = evidence_nms_iou_thr
        self.ema_momentum = ema_momentum
        self.loss_weight = loss_weight
        self.layer_loss_weights = tuple(layer_loss_weights)
        self.scale_tokens = nn.Parameter(torch.randn(num_scale_tokens, embed_dims) * .02)
        self.scale_embedding = nn.Sequential(nn.Linear(1, embed_dims), nn.GELU(),
                                             nn.Linear(embed_dims, embed_dims))
        self.layers = nn.ModuleList([
            ScaleDecoderLayer(embed_dims, num_heads, ffn_dims, dropout, i in cross_layers)
            for i in range(num_layers)])
        self.output_norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in self.layers])
        self.scale_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.GELU(), nn.Linear(embed_dims, 1))
            for _ in self.layers])
        for head in self.scale_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def pool_spatial_features(self, feature_maps):
        """Halve H/W independently on each level/view; retain all view tokens."""
        if not feature_maps:
            raise ValueError('scale decoding requires spatial features')
        tokens = []
        batch, views = feature_maps[0].shape[:2]
        for feature in feature_maps:
            b, v, channels, height, width = feature.shape
            if (b, v, channels) != (batch, views, self.embed_dims):
                raise ValueError('spatial features must share [B, V, embed_dims]')
            pooled = F.adaptive_avg_pool2d(
                feature.float().reshape(b * v, channels, height, width),
                (max(1, height // self.spatial_downsample_ratio),
                 max(1, width // self.spatial_downsample_ratio)))
            tokens.append(pooled.flatten(2).transpose(1, 2).reshape(b, -1, channels))
        return torch.cat(tokens, dim=1)

    def _retrieve_sizes(self, query, labels):
        # Clone protects autograd's saved cosine operands from in-place EMA
        # updates performed after this forward and before backward.
        prototypes = self.query_prototypes.detach().clone()[labels]
        available = self.valid_modes[labels] & (self.prototype_counts[labels] > 0)
        similarities = torch.einsum(
            'bkd,bkmd->bkm', F.normalize(query.float(), dim=-1),
            F.normalize(prototypes, dim=-1))
        modes = similarities.masked_fill(~available, -torch.inf).argmax(-1)
        default_modes = self.mode_counts.masked_fill(~self.valid_modes, -1).argmax(-1)
        modes = torch.where(available.any(-1), modes, default_modes[labels])
        return self.metric_size_priors[labels, modes], modes

    def forward(self, detection_query, centers, sizes, class_logits,
                feature_maps, first_frame_pose, axis_align_matrix):
        with torch.autocast(device_type=detection_query.device.type, enabled=False):
            return self._forward(detection_query.float(), centers.float(), sizes.float(),
                                 class_logits.float(), feature_maps,
                                 first_frame_pose, axis_align_matrix)

    def _forward(self, detection_query, centers, sizes, class_logits,
                 feature_maps, first_frame_pose, axis_align_matrix):
        batch, queries, dims = detection_query.shape
        if (dims != self.embed_dims or queries < 1 or centers.shape != (batch, queries, 3)
                or sizes.shape != centers.shape
                or class_logits.shape != (batch, queries, len(self.classes))):
            raise ValueError('invalid final detection query/box/class shapes')
        valid = (torch.isfinite(detection_query).all(-1) & torch.isfinite(centers).all(-1)
                 & torch.isfinite(sizes).all(-1) & (sizes > 0).all(-1)
                 & torch.isfinite(class_logits).all(-1))
        if not valid.any(-1).all():
            raise RuntimeError('IP-MSR requires at least one valid detection query per scene')
        safe_query = torch.where(valid[..., None], detection_query, 0.)
        scores, labels = class_logits.sigmoid().max(-1)
        priors, modes = self._retrieve_sizes(safe_query, labels)
        safe_sizes = torch.where(valid[..., None], sizes, 1.)
        log_evidence = (priors.log() - safe_sizes.log()).mean(-1)
        iou_centers = rotate_vggt_points_to_scene_axes(
            centers.detach(), first_frame_pose, axis_align_matrix)
        references, evidence_indices = [], []
        topk_fallback = []
        for b in range(batch):
            with torch.no_grad():
                indices = torch.nonzero(valid[b] & (scores[b] >= self.score_thr)).flatten()
                fallback = indices.numel() == 0
                if fallback:
                    candidates = torch.nonzero(valid[b]).flatten()
                    order = scores[b, candidates].argsort(descending=True, stable=True)
                    indices = candidates[order[:self.fallback_topk]]
                keep = _aligned_3d_nms(
                    iou_centers[b, indices], sizes[b, indices].detach(),
                    scores[b, indices].detach(), labels[b, indices], self.evidence_nms_iou_thr)
                indices = indices[keep]
            weights = scores[b, indices]
            # Finite logits can still sigmoid-underflow to zero. Equal weights
            # in that case preserve a defined scale without a GT fallback.
            weight_sum = weights.sum()
            normalized_weights = torch.where(
                weight_sum > 0, weights / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny),
                torch.full_like(weights, 1. / len(indices)))
            references.append((normalized_weights * log_evidence[b, indices]).sum())
            evidence_indices.append(indices)
            topk_fallback.append(fallback)
        reference = torch.stack(references)
        prior_reference = reference
        memory = self.pool_spatial_features(feature_maps)
        if memory.shape[0] != batch or not torch.isfinite(memory).all():
            raise ValueError('scale spatial memory must be finite and match batch size')
        query = self.scale_tokens[None].expand(batch, -1, -1)
        log_scales, deltas = [], []
        for layer, norm, head in zip(self.layers, self.output_norms, self.scale_heads):
            query = query + self.scale_embedding(reference[:, None, None])
            query = layer(query, memory, safe_query, ~valid)
            delta = head(norm(query).mean(dim=1)).squeeze(-1)
            reference = reference + delta
            deltas.append(delta)
            log_scales.append(reference)
        all_log_scales = torch.stack(log_scales)
        if not torch.isfinite(all_log_scales).all():
            raise FloatingPointError('IP-MSR log scales are not finite')
        scale = all_log_scales[-1].exp()
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise FloatingPointError('IP-MSR scales must be finite and positive')
        return dict(scale=scale, log_scale=all_log_scales[-1],
                    all_log_scales=all_log_scales, prior_log_scale=prior_reference,
                    log_scale_deltas=torch.stack(deltas), scale_tokens=query,
                    evidence_indices=evidence_indices, retrieved_modes=modes,
                    topk_fallback=torch.tensor(topk_fallback, device=query.device))

    def loss(self, outputs, gt_scale):
        target = torch.as_tensor(gt_scale, device=outputs['scale'].device,
                                 dtype=torch.float32).reshape(-1)
        if target.shape != outputs['scale'].shape or not torch.isfinite(target).all() or (target <= 0).any():
            raise ValueError('GT scene scale must have shape [B] and be finite positive')
        return {('loss_scale' if i == len(self.layers) - 1 else f'd{i}.loss_scale'):
                F.smooth_l1_loss(prediction, target.log()) * weight * self.loss_weight
                for i, (prediction, weight) in enumerate(zip(outputs['all_log_scales'], self.layer_loss_weights))}

    @torch.no_grad()
    def update_prototypes(self, detection_query, matches, batch_data_samples):
        if not self.training:
            return
        sums = torch.zeros_like(self.query_prototypes)
        counts = torch.zeros_like(self.prototype_counts)
        for b, (query_ids, gt_ids) in enumerate(matches):
            gt = batch_data_samples[b].gt_instances_3d
            sizes = gt.bboxes_3d.tensor[:, 3:6].to(sums.device).float()
            labels = gt.labels_3d.to(sums.device)
            for gt_id in gt_ids.unique():
                label = labels[gt_id]
                size = sizes[gt_id]
                if label < 0 or label >= len(self.classes):
                    continue
                if not torch.isfinite(size).all() or (size <= 0).any():
                    raise ValueError('prototype GT size must be finite and positive')
                distances = (self.metric_size_priors[label].log() - size.log()).square().sum(-1)
                mode = distances.masked_fill(~self.valid_modes[label], torch.inf).argmin()
                feature = detection_query[b, query_ids[gt_ids == gt_id]].detach().float().mean(0)
                if not torch.isfinite(feature).all() or feature.norm() <= 1e-12:
                    continue
                sums[label, mode] += F.normalize(feature, dim=0)
                counts[label, mode] += 1
        # All ranks participate, including ranks with no matched GT instances.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        active = counts > 0
        mean = F.normalize(sums / counts.clamp_min(1)[..., None], dim=-1)
        initialized = self.prototype_counts > 0
        updated = F.normalize(self.ema_momentum * self.query_prototypes
                              + (1. - self.ema_momentum) * mean, dim=-1)
        self.query_prototypes[active] = torch.where(
            initialized[..., None], updated, mean)[active]
        self.prototype_counts.add_(counts)
