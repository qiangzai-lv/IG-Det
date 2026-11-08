# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from igdet.detr3_models.helpers import GenericMLP
from mmdet.utils import reduce_mean
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from igdet.rotated_geometry import pack_boxes, rotated_overlaps, rotated_nms
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from igdet.camera_alignment import denormalize_vggt_boxes
from igdet.matcher import RepeatedHungarianMatcher


def decode_size_residuals(size_residuals, initial_query_sizes,
                          size_logit_range):
    """Decode detached, multiplicative size refinement across layers."""
    if not size_residuals:
        return [], [], []
    if (len(size_logit_range) != 2
            or size_logit_range[0] >= size_logit_range[1]):
        raise ValueError('size_logit_range must be an increasing pair')

    first = size_residuals[0]
    if initial_query_sizes.ndim != 3 or initial_query_sizes.shape[-1] != 3:
        raise ValueError('initial_query_sizes must have shape [B, Q, 3]')
    initial_sizes = initial_query_sizes.to(
        device=first.device, dtype=first.dtype).transpose(1, 2)
    if initial_sizes.shape != first.shape:
        raise ValueError('initial query sizes must match size residuals')
    if not torch.isfinite(initial_sizes).all() or (initial_sizes <= 0).any():
        raise ValueError('initial query sizes must be finite and positive')
    reference_log = initial_sizes.log()

    reference_logs = []
    predicted_logs = []
    predicted_sizes = []
    for residual in size_residuals:
        if residual.shape != first.shape:
            raise ValueError('all size residuals must have the same shape')
        reference_logs.append(reference_log)
        predicted_log = reference_log + residual.float()
        bounded_log = predicted_log.clamp(
            min=size_logit_range[0], max=size_logit_range[1])
        stable_log = predicted_log + (bounded_log - predicted_log).detach()
        predicted_logs.append(stable_log)
        predicted_sizes.append(stable_log.exp())
        reference_log = stable_log.detach()
    return reference_logs, predicted_logs, predicted_sizes


@MODELS.register_module()
class IGDetHead(BaseModule):

    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 cls_loss: ConfigType = dict(
                     type='mmdet.FocalLoss', use_sigmoid=True,
                     gamma=2.0, alpha=0.25, loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 mlp_dropout=0.3,
                 matcher_cost_weights={
                     'cls': 2.0, 'center': 1.0, 'size': 1.0, 'giou': 2.0},
                 loss_weights={'center_loss': 2.0, 'size_loss': 1.0,
                               'cls_loss': 2.0, 'iou_loss': 2.0},
                 if_v2_head=False,
                 matcher='repeated_hungarian',
                 loss_layer_ids=None,
                 gt_repeat_num=5,
                 size_logit_range=(-10.0, 10.0),
                 yaw_weight=0.5,
                 ):
        super(IGDetHead, self).__init__(init_cfg)
        self.n_classes = n_classes
        self.yaw_weight = float(yaw_weight)
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if if_v2_head:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels // 2, n_channels // 4, n_channels // 8],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        else:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)
        if matcher != 'repeated_hungarian':
            raise ValueError('matcher must be repeated_hungarian')
        self.matcher = RepeatedHungarianMatcher(
            cost_weights=matcher_cost_weights,
            gt_repeat_num=gt_repeat_num)
        self.loss_weights = loss_weights
        if len(size_logit_range) != 2 or size_logit_range[0] >= size_logit_range[1]:
            raise ValueError('size_logit_range must be an increasing pair')
        self.size_logit_range = tuple(float(value) for value in size_logit_range)
        self.gt_repeat_num = int(gt_repeat_num)
        if loss_layer_ids is None:
            loss_layer_ids = list(range(n_levels))
        self.loss_layer_ids = sorted(set(loss_layer_ids))
        if not self.loss_layer_ids:
            raise ValueError('loss_layer_ids must contain at least one layer')
        if self.loss_layer_ids[0] < 0 or self.loss_layer_ids[-1] >= n_levels:
            raise ValueError(
                f'loss_layer_ids must be within [0, {n_levels - 1}]')

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        center_head = self.mlp_func(output_dim=3)
        size_mlp_func = partial(
            GenericMLP,
            norm_fn_name=None,
            activation='relu',
            use_conv=True,
            hidden_dims=[n_channels, n_channels],
            dropout=None,
            input_dim=n_channels)
        size_head = size_mlp_func(output_dim=3)
        semcls_head = self.mlp_func(output_dim=n_classes)
        # Same scalar scene-yaw head at every decoder layer, as VGGT-Det.
        self.yaw_head = self.mlp_func(output_dim=1)
        self.center_heads = nn.ModuleList([
            copy.deepcopy(center_head) for _ in range(n_levels)
        ])
        self.size_heads = nn.ModuleList([
            copy.deepcopy(size_head) for _ in range(n_levels)
        ])
        self.semcls_heads = nn.ModuleList([
            copy.deepcopy(semcls_head) for _ in range(n_levels)
        ])
        for center_head in self.center_heads:
            nn.init.constant_(center_head.layers[-1].weight, 0.)
            nn.init.constant_(center_head.layers[-1].bias, 0.)
        for size_head in self.size_heads:
            nn.init.constant_(size_head.layers[-1].weight, 0.)
            nn.init.constant_(size_head.layers[-1].bias, 0.)
        prior_bias = -math.log((1.0 - 0.01) / 0.01)
        for semcls_head in self.semcls_heads:
            nn.init.constant_(semcls_head.layers[-1].bias, prior_bias)

    def forward(self, x, batch_inputs_dict, refined_query_xyz=None,
                initial_query_sizes=None, layer_ids=None):
        if layer_ids is None:
            layer_ids = list(range(len(x)))
        if len(layer_ids) != len(x):
            raise ValueError('Layer ids must match decoder outputs')
        if layer_ids != list(range(len(x))):
            raise ValueError(
                'Size refinement requires all decoder layers in order')
        if refined_query_xyz is None or len(refined_query_xyz) != len(x):
            raise ValueError('Refined references must match decoder outputs')
        if initial_query_sizes is None:
            raise ValueError('Initial query sizes are required')

        center_preds = []
        size_residual_preds = []
        cls_preds = []
        yaw_preds = []
        for feature, center, layer_id in zip(
                x, refined_query_xyz, layer_ids):
            center_preds.append(center.permute(0, 2, 1))
            size_residual_preds.append(self.size_heads[layer_id](feature))
            cls_preds.append(self.semcls_heads[layer_id](feature))
            yaw_preds.append(self.yaw_head(feature).float())
        with torch.autocast(device_type=size_residual_preds[0].device.type,
                            enabled=False):
            size_reference_logs, size_log_preds, size_preds = (
                decode_size_residuals(
                    [residual.float() for residual in size_residual_preds],
                    initial_query_sizes,
                    self.size_logit_range))
        return dict(
            center_preds=center_preds,
            size_preds=size_preds,
            size_residual_preds=size_residual_preds,
            size_reference_logs=size_reference_logs,
            size_log_preds=size_log_preds,
            cls_preds=cls_preds, yaw_preds=yaw_preds)

    @staticmethod
    def _align_detection_outputs(outputs, batch_inputs_dict):
        required = ('pose_matrix', 'axis_align_matrix', 'vggt_gt_scale')
        missing = [key for key in required if key not in batch_inputs_dict]
        if missing:
            raise KeyError(
                f'Missing VGGT-to-aligned transform inputs: {missing}')

        aligned_outputs = dict(outputs)
        aligned_centers = []
        aligned_sizes = []
        aligned_size_logs = []
        for centers_vggt, sizes_vggt in zip(
                outputs['center_preds'], outputs['size_preds']):
            centers, sizes = denormalize_vggt_boxes(
                centers_vggt.transpose(1, 2),
                sizes_vggt.transpose(1, 2),
                batch_inputs_dict['pose_matrix'],
                batch_inputs_dict['axis_align_matrix'],
                batch_inputs_dict['vggt_gt_scale'])
            centers = centers.transpose(1, 2)
            sizes = sizes.transpose(1, 2)
            aligned_centers.append(centers)
            aligned_sizes.append(sizes)
            aligned_size_logs.append(sizes.clamp_min(1e-5).log())
        aligned_outputs['center_preds'] = aligned_centers
        aligned_outputs['size_preds'] = aligned_sizes
        aligned_outputs['size_log_preds'] = aligned_size_logs
        return aligned_outputs

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             batch_inputs_dict: dict, refined_query_xyz=None,
             initial_query_sizes=None, **kwargs) -> dict:
        if refined_query_xyz is None or len(refined_query_xyz) != len(x):
            raise ValueError('Loss requires one refined reference per layer')
        layer_ids = self.loss_layer_ids
        outputs = self(
            x, batch_inputs_dict, refined_query_xyz, initial_query_sizes)
        outputs = self._align_detection_outputs(outputs, batch_inputs_dict)

        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            batch_input_points = [None for i in range(len(batch_data_samples))]

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = (
                       outputs['center_preds'], outputs['size_preds'],
                       outputs['size_log_preds'], outputs['cls_preds'], outputs['yaw_preds'], layer_ids,
                       batch_gt_instances_3d, batch_input_metas,
                       batch_input_points, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     size_log_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     yaw_preds: List[List[Tensor]],
                     layer_ids: List[int],
                     #  objness_preds: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_input_points,
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:

        if layer_ids[-1] >= len(center_preds):
            raise ValueError('Supervised layer id exceeds predictions')

        losses_by_layer = []
        for layer_id in layer_ids:
            center_losses = []
            size_losses = []
            cls_losses = []
            giou_losses = []
            yaw_losses = []
            matches = []
            for batch_id in range(len(batch_input_metas)):
                gt_bboxes = batch_gt_instances_3d[batch_id].bboxes_3d
                gt_sizes = gt_bboxes.tensor[:, 3:6].clamp_min(1e-5)
                matches.append(self.matcher._get_targets(
                    center_preds[layer_id][batch_id].t(),
                    size_preds[layer_id][batch_id].t(),
                    size_log_preds[layer_id][batch_id].t(),
                    cls_preds[layer_id][batch_id].t(),
                    gt_bboxes.gravity_center,
                    gt_sizes,
                    batch_gt_instances_3d[batch_id].labels_3d,
                    yaw_preds[layer_id][batch_id].t(), gt_bboxes.tensor[:,6]))
            local_num_pos = sum(match[0].numel() for match in matches)
            avg_factor = reduce_mean(center_preds[layer_id][0].new_tensor(
                [local_num_pos], dtype=torch.float32)).clamp_min(1.0).item()

            for batch_id in range(len(batch_input_metas)):
                center_loss, size_loss, cls_loss, giou_loss, yaw_loss = \
                    self._loss_by_feat_single(
                        center_pred=center_preds[layer_id][batch_id],
                        size_pred=size_preds[layer_id][batch_id],
                        size_log_pred=size_log_preds[layer_id][batch_id],
                        cls_pred=cls_preds[layer_id][batch_id],
                        yaw_pred=yaw_preds[layer_id][batch_id],
                        input_meta=batch_input_metas[batch_id],
                        gt_bboxes=batch_gt_instances_3d[
                            batch_id].bboxes_3d,
                        gt_labels=batch_gt_instances_3d[
                            batch_id].labels_3d,
                        input_points=batch_input_points[batch_id],
                        match_indices=matches[batch_id],
                        avg_factor=avg_factor)
                center_losses.append(center_loss)
                size_losses.append(size_loss)
                cls_losses.append(cls_loss)
                giou_losses.append(giou_loss)
                yaw_losses.append(yaw_loss)
            losses_by_layer.append((layer_id, dict(
                center_loss=torch.sum(torch.stack(center_losses)),
                size_loss=torch.sum(torch.stack(size_losses)),
                cls_loss=torch.sum(torch.stack(cls_losses)),
                giou_loss=torch.sum(torch.stack(giou_losses)),
                yaw_loss=torch.sum(torch.stack(yaw_losses)))))

        loss_dict = {}
        main_layer_id = layer_ids[-1]
        for layer_id, layer_losses in losses_by_layer:
            prefix = '' if layer_id == main_layer_id else f'd{layer_id}.'
            for name, value in layer_losses.items():
                loss_dict[f'{prefix}{name}'] = value
        return loss_dict

    def _loss_by_feat_single(self, center_pred, size_pred,
                             size_log_pred, cls_pred, yaw_pred, input_meta,
                             gt_bboxes, gt_labels, input_points,
                             match_indices=None, avg_factor=None):
        del input_meta, input_points
        centers = center_pred.t()
        sizes = size_pred.t()
        size_logs = size_log_pred.t()
        cls_scores = cls_pred.t()
        gt_centers = gt_bboxes.gravity_center
        gt_sizes = gt_bboxes.tensor[:, 3:6].clamp_min(1e-5)
        if match_indices is None:
            pred_indices, gt_indices = self.matcher._get_targets(
                centers, sizes, size_logs, cls_scores,
                gt_centers, gt_sizes, gt_labels, yaw_pred.t(), gt_bboxes.tensor[:,6])
        else:
            pred_indices, gt_indices = match_indices
        if avg_factor is None:
            num_pos = reduce_mean(centers.new_tensor(
                [pred_indices.numel()], dtype=torch.float32))
            avg_factor = num_pos.clamp_min(1.0).item()

        cls_target = torch.full(
            (centers.shape[0],), self.n_classes,
            dtype=torch.long, device=centers.device)
        cls_target[pred_indices] = gt_labels[gt_indices]
        cls_loss = self.cls_loss(
            cls_scores, cls_target, avg_factor=avg_factor)
        cls_loss = cls_loss * self.loss_weights['cls_loss']

        if pred_indices.numel() == 0:
            size_loss = size_logs.sum() * 0.0
        else:
            size_loss = F.l1_loss(
                size_logs[pred_indices],
                gt_sizes[gt_indices].clamp_min(1e-5).log(),
                reduction='sum') / max(float(avg_factor), 1.0)
        size_loss = size_loss * self.loss_weights['size_loss']
        if pred_indices.numel() == 0:
            center_loss = centers.sum() * 0.0
            giou_loss = sizes.sum() * 0.0
        else:
            center_loss = F.l1_loss(
                centers[pred_indices], gt_centers[gt_indices],
                reduction='sum') / avg_factor

            pred_boxes = pack_boxes(centers[pred_indices], sizes[pred_indices], yaw_pred.t()[pred_indices])
            gt_boxes = pack_boxes(gt_centers[gt_indices], gt_sizes[gt_indices], gt_bboxes.tensor[gt_indices,6])
            giou = rotated_overlaps(pred_boxes, gt_boxes, mode='giou', aligned=True)
            giou_loss = (1.0 - giou).sum() / avg_factor
        center_loss = center_loss * self.loss_weights['center_loss']
        giou_loss = giou_loss * self.loss_weights['iou_loss']
        if pred_indices.numel() == 0:
            yaw_loss = yaw_pred.sum()*0.
        else:
            yaw_loss = F.mse_loss(yaw_pred.t()[pred_indices,0],
                                  gt_bboxes.tensor[gt_indices,6], reduction='sum') / avg_factor
        return center_loss, size_loss, cls_loss, giou_loss, yaw_loss*self.yaw_weight

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList, batch_inputs_dict,
                refined_query_xyz=None, initial_query_sizes=None,
                layer_ids=None,
                rescale: bool = False) -> InstanceList:

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outputs = self(
            x, batch_inputs_dict, refined_query_xyz,
            initial_query_sizes, layer_ids)
        outputs = self._align_detection_outputs(outputs, batch_inputs_dict)
        predictions = self.predict_by_feat(
            [outputs['center_preds'][-1]],
            [outputs['size_preds'][-1]],
            [outputs['cls_preds'][-1]],
            [outputs['yaw_preds'][-1]],
            batch_input_metas=batch_input_metas,
            rescale=rescale, batch_inputs_dict=batch_inputs_dict, batch_data_samples=batch_data_samples)
        return predictions

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                        size_preds: List[List[Tensor]],
                        cls_preds: List[List[Tensor]],
                        yaw_preds: List[List[Tensor]],
                        batch_input_metas: List[dict], batch_inputs_dict: dict, batch_data_samples,
                        **kwargs) -> List[InstanceData]:

        results = []
        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            batch_input_points = [None for i in range(len(batch_input_metas))]
        for i in range(len(batch_input_metas)):
            results.append(
                self._predict_by_feat_single(
                    center_preds=[x[i] for x in center_preds],
                    size_preds=[x[i] for x in size_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    yaw_preds=[x[i] for x in yaw_preds],
                    input_meta=batch_input_metas[i],
                    input_points=batch_input_points[i],
                    data_samples=batch_data_samples[i]))
        return results

    def _predict_by_feat_single(self, center_preds, size_preds, cls_preds, yaw_preds,
                                input_meta: dict, input_points, data_samples) -> InstanceData:

        mlvl_bboxes, mlvl_scores = [], []
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[
                stage_idx].t()
            angles = yaw_preds[stage_idx].t()
            scores = cls_scores.sigmoid()

            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                centers = centers[ids]
                sizes = sizes[ids]
                scores = scores[ids]
                angles = angles[ids]
            bboxes = pack_boxes(centers, sizes, angles)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes_after_nms, scores, labels = self._nms(bboxes, scores,
                                                     input_meta)  # bboxes(n_box, 6) (x_center, y_center, z_center, w, h, z)

        bboxes = input_meta['box_type_3d'](
            bboxes_after_nms, box_dim=7, with_yaw=True, origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        return results

    def _nms(self, bboxes, scores, img_meta):
        # Reference VGGT-Det runs NMS separately for every foreground class;
        # one proposal may consequently produce more than one class prediction.
        del img_meta
        output_boxes, output_scores, output_labels = [], [], []
        for label in range(scores.shape[1]):
            mask = scores[:,label] > self.test_cfg.score_thr
            boxes, values = bboxes[mask], scores[mask,label]
            labels = torch.full_like(values, label, dtype=torch.long)
            keep = rotated_nms(boxes, values, labels, self.test_cfg.iou_thr, bev=True)
            output_boxes.append(boxes[keep])
            output_scores.append(values[keep])
            output_labels.append(labels[keep])
        return torch.cat(output_boxes), torch.cat(output_scores), torch.cat(output_labels)
