"""Upright, geometric-center [x,y,z,dx,dy,dz,yaw] boxes.

Yaw is CCW X->Y in scene coordinates. GIoU uses the minimum enclosing
BEV rectangle, matching Rotated_IoU.cal_giou_3d(enclosing_type='smallest').
The differentiable implementation uses native PyTorch, without a CUDA extension.
"""
import torch


def pack_boxes(centers, sizes, yaw):
    if yaw.ndim == centers.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    return torch.cat((centers, sizes, yaw), dim=-1)


def bev_corners(boxes):
    signs = boxes.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    local = signs * boxes[..., None, 3:5] / 2
    c, s = boxes[..., 6].cos()[..., None], boxes[..., 6].sin()[..., None]
    x, y = local.unbind(-1)
    return torch.stack((c*x-s*y, s*x+c*y), -1) + boxes[..., None, :2]


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _inside(points, boxes):
    delta = points - boxes[..., None, :2]
    c, s = boxes[..., 6].cos()[..., None], boxes[..., 6].sin()[..., None]
    x, y = delta.unbind(-1)
    local = torch.stack((c*x+s*y, -s*x+c*y), -1)
    return (local.abs() <= boxes[..., None, 3:5]/2 + 1e-6).all(-1)


def _intersection_area(a, b, ca, cb):
    # All polygon vertices are contained corners or edge intersections.
    start_a, edge_a = ca[:, :, None], (ca.roll(-1, 1)-ca)[:, :, None]
    start_b, edge_b = cb[:, None, :], (cb.roll(-1, 1)-cb)[:, None, :]
    denominator = _cross(edge_a, edge_b)
    usable = denominator.abs() > 1e-8
    safe = torch.where(usable, denominator, torch.ones_like(denominator))
    delta = start_b-start_a
    t, u = _cross(delta, edge_b)/safe, _cross(delta, edge_a)/safe
    mask = usable & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    intersections = start_a + t[..., None]*edge_a
    vertices = torch.cat((ca, cb, intersections.flatten(1, 2)), 1)
    valid = torch.cat((_inside(ca, b), _inside(cb, a), mask.flatten(1)), 1)
    # Mask invalid intersections before reductions; no inf * 0 intermediates.
    vertices = torch.where(valid[..., None], vertices, 0.)
    count = valid.sum(1)
    mean = vertices.sum(1)/count.clamp_min(1)[:, None]
    offset = vertices-mean[:, None]
    # Sorting is discrete; gradients flow through the selected vertices.
    angles = torch.atan2(offset[..., 1].detach(), offset[..., 0].detach())
    order = torch.where(valid, angles, torch.full_like(angles, 10.)).argsort(dim=1, stable=True)
    polygon = vertices.gather(1, order[..., None].expand(-1, -1, 2))
    position = torch.arange(vertices.shape[1], device=a.device)[None].expand_as(valid)
    next_index = torch.where(position+1 < count[:, None], position+1, 0)
    following = polygon.gather(1, next_index[..., None].expand(-1, -1, 2))
    terms = _cross(polygon-mean[:, None], following-mean[:, None])
    area = torch.where(position < count[:, None], terms, 0.).sum(1).abs()/2
    return torch.where(count >= 3, area, 0.)


def _enclosing_area(ca, cb):
    corners = torch.cat((ca, cb), 1)
    # Same 24 candidate hull-edge directions as the reference implementation.
    pairs = [(i,j) for i in range(8) for j in range(i+1,8)
             if (i,j) not in ((0,2),(1,3),(4,6),(5,7))]
    indices = torch.tensor(pairs, device=ca.device)
    directions = corners[:, indices[:,1]]-corners[:, indices[:,0]]
    length = directions.square().sum(-1)
    directions = directions/length.clamp_min(1e-14).sqrt()[..., None]
    # Subtract a common origin to improve accuracy for large world coordinates.
    centered = corners-corners.mean(1, keepdim=True)
    parallel = (centered[:, None]*directions[:, :, None]).sum(-1)
    perpendicular = _cross(directions[:, :, None], centered[:, None])
    area = (parallel.amax(-1)-parallel.amin(-1))*(perpendicular.amax(-1)-perpendicular.amin(-1))
    return torch.where(length > 1e-14, area, torch.full_like(area, float('inf'))).amin(-1)


def _aligned(a, b, mode, bev):
    ca, cb = bev_corners(a), bev_corners(b)
    intersection = _intersection_area(a, b, ca, cb)
    area_a, area_b = a[:,3]*a[:,4], b[:,3]*b[:,4]
    intersection = torch.minimum(intersection, torch.minimum(area_a, area_b)).clamp_min(0)
    if bev:
        union = area_a+area_b-intersection
        enclosing = _enclosing_area(ca,cb) if mode=='giou' else None
    else:
        lower = torch.maximum(a[:,2]-a[:,5]/2,b[:,2]-b[:,5]/2)
        upper = torch.minimum(a[:,2]+a[:,5]/2,b[:,2]+b[:,5]/2)
        intersection = intersection*(upper-lower).clamp_min(0)
        union = area_a*a[:,5]+area_b*b[:,5]-intersection
        if mode=='giou':
            z_range = (torch.maximum(a[:,2]+a[:,5]/2,b[:,2]+b[:,5]/2)
                       -torch.minimum(a[:,2]-a[:,5]/2,b[:,2]-b[:,5]/2))
            enclosing = _enclosing_area(ca,cb)*z_range
    iou = intersection/union.clamp_min(1e-8)
    if mode=='iou':
        return iou.clamp(0,1)
    enclosing = torch.maximum(enclosing,union)
    return (iou-(enclosing-union)/enclosing.clamp_min(1e-8)).clamp(-1,1)


def rotated_overlaps(a, b, mode='iou', aligned=False, bev=False, chunk_size=2048):
    """Aligned [N] or pairwise [N,M] overlaps. Geometry runs in FP32/FP64.

    CUDA pairwise IoU uses MMCV's equivalent rotated BEV intersection as a fast
    no-grad path. Training GIoU and other devices use the same native geometry.
    """
    if mode not in ('iou','giou') or a.ndim!=2 or b.ndim!=2 or a.shape[-1]!=7 or b.shape[-1]!=7:
        raise ValueError('Expected [N,7] boxes and iou/giou mode')
    if aligned and len(a)!=len(b):
        raise ValueError('Aligned boxes must have the same count')
    with torch.autocast(device_type=a.device.type, enabled=False):
        dtype = torch.float64 if a.dtype==torch.float64 else torch.float32
        a,b = a.to(dtype),b.to(dtype)
        shape = (len(a),) if aligned else (len(a),len(b))
        if not len(a) or not len(b):
            return (a.sum()+b.sum()).expand(shape)*0
        # NPU migration adapters can remap is_cuda; inspect the actual device.
        if not aligned and mode=='iou' and a.device.type == 'cuda' and not torch.is_grad_enabled():
            from mmcv.ops import box_iou_rotated
            # MMCV clockwise=True uses the numeric Rz(+yaw) matrix. Its
            # name refers to image Y-down, not the scene Y-up convention.
            area = box_iou_rotated(a[:,[0,1,3,4,6]].contiguous(),
                                   b[:,[0,1,3,4,6]].contiguous(),clockwise=True)
            if bev:
                return area
            va=a[:,3]*a[:,4];vb=b[:,3]*b[:,4]
            intersection = area*(va[:,None]+vb[None])/(1+area)
            height=(torch.minimum(a[:,None,2]+a[:,None,5]/2,b[None,:,2]+b[None,:,5]/2)
                    -torch.maximum(a[:,None,2]-a[:,None,5]/2,b[None,:,2]-b[None,:,5]/2)).clamp_min(0)
            volume=intersection*height
            return volume/(va[:,None]*a[:,None,5]+vb[None]*b[None,:,5]-volume).clamp_min(1e-8)
        total = len(a) if aligned else len(a)*len(b)
        values=[]
        for start in range(0,total,chunk_size):
            ids=torch.arange(start,min(start+chunk_size,total),device=a.device)
            aa=a[ids] if aligned else a[ids//len(b)]
            bb=b[ids] if aligned else b[ids%len(b)]
            values.append(_aligned(aa,bb,mode,bev))
        return torch.cat(values).reshape(shape)


@torch.no_grad()
def rotated_nms(boxes,scores,labels,threshold,bev=False):
    """Class-aware greedy suppression; candidates use volume, final boxes BEV."""
    import numpy as np
    order=scores.argsort().flip(0)
    kept=[]
    for label in labels[order].unique():
        positions=torch.nonzero(labels[order]==label,as_tuple=False).flatten()
        group=boxes[order[positions]].float()
        suppression=np.empty((len(group),len(group)),dtype=np.bool_)
        for start in range(0,len(group),128):
            suppression[start:start+128]=(rotated_overlaps(group[start:start+128],group,bev=bev)>threshold).cpu().numpy()
        blocked=np.zeros(len(group),dtype=bool)
        for i in range(len(group)):
            if not blocked[i]:
                kept.append(int(positions[i]));blocked |= suppression[i]
    positions=torch.tensor(sorted(kept),device=order.device,dtype=torch.long)
    return order[positions]
