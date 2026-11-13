#!/usr/bin/env python3
"""Export current IGDet validation queries and detections without changing inference.

Uses the current model's axis-aligned metric convention (including its existing
GT-scale alignment). No IP-MSR or alternative scale recovery is introduced.
Background is the dataset scan, not a second VGGT reconstruction. No GT boxes
are drawn. Run from the repository root in the mmcv environment.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import random
from contextlib import AbstractContextManager, contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
SIGNS = np.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
                  [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]], np.float32)


def resolve_execution_device(requested='auto'):
    """Register NPU before parsing devices; select its index before model init."""
    requested = requested.lower()
    if requested == 'cpu':
        return 'cpu'
    kind, separator, ordinal = requested.partition(':')
    if kind not in ('auto', 'npu', 'cuda') or (kind == 'auto' and separator):
        raise ValueError('Device must be auto, cpu, npu[:index], or cuda[:index]')
    if separator and (not ordinal.isdecimal()):
        raise ValueError('Device index must be a nonnegative integer')
    # torch_npu registers torch.npu and the NPU device type as an import effect.
    if kind in ('auto', 'npu') and not hasattr(torch, 'npu'):
        try:
            importlib.import_module('torch_npu')
        except ImportError as error:
            if kind == 'npu':
                raise RuntimeError('--device npu requires an importable torch_npu installation') from error
    npu = getattr(torch, 'npu', None)
    if kind == 'auto':
        kind = ('npu' if npu is not None and npu.is_available()
                else 'cuda' if torch.cuda.is_available() else 'cpu')
    if kind == 'cpu':
        return 'cpu'
    backend = getattr(torch, kind, None)
    if backend is None or not backend.is_available():
        raise RuntimeError(f'Requested {kind} is unavailable; check the PyTorch backend or use --device auto')
    index = int(ordinal) if separator else backend.current_device()
    if index >= backend.device_count():
        raise ValueError(f'{kind}:{index} is outside the available device count {backend.device_count()}')
    backend.set_device(index)
    return f'{kind}:{index}'


@contextmanager
def preserve_random_state(device):
    """Restore Python/NumPy/CPU and the selected accelerator, even on errors.

    Do not call CUDA RNG APIs on NPU builds. Both accelerator backends expose
    get_rng_state(index) / set_rng_state(state, index).
    """
    kind, _, ordinal = str(device).partition(':')
    backend = getattr(torch, kind) if kind != 'cpu' else None
    index = int(ordinal) if ordinal else (backend.current_device() if backend else None)
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    accelerator_state = backend.get_rng_state(index).clone() if backend else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if backend:
            backend.set_rng_state(accelerator_state, index)
        random.setstate(py_state)
        np.random.set_state(np_state)


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class QueryCapture(AbstractContextManager):
    """Observe actual aggregated queries; return original tensors unchanged.

    Local, single-process instrumentation, restored even on errors. Copies are
    detached before decoder execution to protect against in-place refinement.
    """
    def __init__(self, model):
        self.model = model
        self.payload = None
        self.originals = {}

    def __enter__(self):
        model = self.model
        for name in ('_select_reconstruction_boxes', 'get_box_features'):
            self.originals[name] = (name in model.__dict__, getattr(model, name))
        select = self.originals['_select_reconstruction_boxes'][1]
        get_features = self.originals['get_box_features'][1]

        def capture_select(*args, **kwargs):
            selected = select(*args, **kwargs)
            self.selected = {key: selected[key].detach().clone() for key in (
                'query_xyz_vggt', 'query_size_vggt', 'scores', 'labels',
                'source_indices', 'fallback_mask')}
            return selected

        def capture_features(feature_maps, inputs, images, xyz, query, *args, **kwargs):
            from igdet.camera_alignment import denormalize_vggt_boxes
            if not torch.equal(xyz, self.selected['query_xyz_vggt']):
                raise RuntimeError('Captured queries differ from decoder input')
            centers, sizes = denormalize_vggt_boxes(
                xyz.detach(), self.selected['query_size_vggt'],
                inputs['pose_matrix'], inputs['axis_align_matrix'],
                inputs['vggt_gt_scale'])
            self.payload = {key: numpy(value) for key, value in self.selected.items()}
            self.payload['query_boxes'] = numpy(torch.cat((centers, sizes), -1))
            self.payload['existing_scene_scale'] = numpy(inputs['vggt_gt_scale'])
            return get_features(feature_maps, inputs, images, xyz, query, *args, **kwargs)

        model._select_reconstruction_boxes = capture_select
        model.get_box_features = capture_features
        return self

    def reset(self):
        self.payload = None
        self.selected = None

    def __exit__(self, *exc):
        for name, (was_instance_attribute, original) in self.originals.items():
            if was_instance_attribute:
                setattr(self.model, name, original)
            else:
                delattr(self.model, name)
        self.reset()


def box_segments(boxes):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 6)
    corners = boxes[:, None, :3] + SIGNS[None] * boxes[:, None, 3:6] / 2
    return corners[:, np.asarray(EDGES)].reshape(-1, 2, 3)


def write_overlay(path, points, rgb, boxes, colors, radius):
    """Colored scan vertices and triangle cylinder edges in one PLY."""
    from plyfile import PlyData, PlyElement
    parts, color_parts, faces = [points], [rgb], []
    offset = len(points)
    angles = np.arange(8) * (2 * np.pi / 8)
    for segment, color in zip(box_segments(boxes), np.repeat(colors, 12, axis=0)):
        a, b = segment
        direction = b - a
        length = np.linalg.norm(direction)
        if length <= 1e-10:
            continue
        direction /= length
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        u = np.cross(direction, axis)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        ring = radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        vertices = np.concatenate((a + ring, b + ring))
        parts.append(vertices)
        color_parts.append(np.tile(color, (16, 1)))
        for i in range(8):
            j = (i + 1) % 8
            faces.extend([(offset+i, offset+j, offset+8+i),
                          (offset+j, offset+8+j, offset+8+i)])
        offset += 16
    xyz = np.concatenate(parts).astype(np.float32)
    rgb = np.concatenate(color_parts).astype(np.uint8)
    vertex = np.empty(len(xyz), dtype=[('x','<f4'), ('y','<f4'), ('z','<f4'),
                                      ('red','u1'), ('green','u1'), ('blue','u1')])
    for i, key in enumerate(('x','y','z')):
        vertex[key] = xyz[:, i]
    for i, key in enumerate(('red','green','blue')):
        vertex[key] = rgb[:, i]
    elements = [PlyElement.describe(vertex, 'vertex')]
    if faces:
        face = np.empty(len(faces), dtype=[('vertex_indices','<i4',(3,))])
        face['vertex_indices'] = faces
        elements.append(PlyElement.describe(face, 'face'))
    PlyData(elements, text=False).write(str(path))


def save_preview(directory, points, rgb, query_boxes, det_boxes, det_colors, elev, azim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ids = np.linspace(0, len(points)-1, min(len(points), 60000)).astype(int)
    all_xyz = np.concatenate((points, box_segments(query_boxes).reshape(-1,3),
                              box_segments(det_boxes).reshape(-1,3)))
    lower, upper = all_xyz.min(0), all_xyz.max(0)
    middle, half = (lower+upper)/2, max(float((upper-lower).max())/2, .01)*1.02

    def draw(ax, boxes, colors, title):
        ax.scatter(*points[ids].T, c=rgb[ids]/255., s=.25, alpha=.65, rasterized=True)
        if len(boxes):
            ax.add_collection3d(Line3DCollection(box_segments(boxes),
                               colors=np.repeat(colors/255.,12,axis=0), linewidths=1.0))
        ax.set(xlim=(middle[0]-half,middle[0]+half), ylim=(middle[1]-half,middle[1]+half),
               zlim=(middle[2]-half,middle[2]+half), title=title)
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()

    qcolors = np.tile([255,140,25], (len(query_boxes),1))
    fig = plt.figure(figsize=(14,7))
    draw(fig.add_subplot(121, projection='3d'), query_boxes, qcolors, f'Query initialization ({len(query_boxes)})')
    draw(fig.add_subplot(122, projection='3d'), det_boxes, det_colors, f'Detections ({len(det_boxes)})')
    fig.savefig(directory/'comparison.png', dpi=200, bbox_inches='tight')
    fig.savefig(directory/'comparison.pdf', bbox_inches='tight'); plt.close(fig)
    for name, boxes, colors in [('query_boxes',query_boxes,qcolors), ('detection_boxes',det_boxes,det_colors)]:
        fig = plt.figure(figsize=(7,7)); draw(fig.add_subplot(111,projection='3d'),boxes,colors,name)
        fig.savefig(directory/f'{name}.png', dpi=200, bbox_inches='tight'); plt.close(fig)


def export_scene(directory, points, rgb, query, detections, metadata, *, score_thr=.3,
                 query_topk=None, max_points=200000, radius=None, preview=True, elev=25, azim=-65):
    from igdet.visualization_colors import CLASS_COLORS
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    valid = np.isfinite(points).all(-1)
    points, rgb = points[valid], rgb[valid]
    if not len(points):
        raise ValueError('Empty scene point cloud')
    if len(points) > max_points:
        ids = np.linspace(0,len(points)-1,max_points).astype(int)
        points, rgb = points[ids], rgb[ids]
    qboxes = query['query_boxes']
    # FPS fallback with a real source observation remains visible. Synthetic
    # origin padding has source_index == -1 and is omitted from rendering only.
    qmask = (query['source_indices'] >= 0) & np.isfinite(qboxes).all(-1) & (qboxes[:,3:] > 0).all(-1)
    qids = np.flatnonzero(qmask)
    if query_topk is not None:
        qids = qids[np.argsort(-np.nan_to_num(query['scores'][qids], nan=-np.inf),kind='stable')[:query_topk]]
    dboxes, scores, labels = (detections[key] for key in ('boxes','scores','labels'))
    dmask = np.isfinite(scores) & (scores >= score_thr) & np.isfinite(dboxes).all(-1) & (dboxes[:,3:] > 0).all(-1)
    dids = np.flatnonzero(dmask)
    colors = np.asarray([CLASS_COLORS.get(int(label),(150,150,150)) for label in labels[dids]],np.uint8).reshape(-1,3)
    if radius is None:
        radius = max(float(np.linalg.norm(np.quantile(points,.99,axis=0)-np.quantile(points,.01,axis=0)))*.0008,1e-5)
    empty = np.empty((0,6),np.float32)
    write_overlay(directory/'scene.ply', points,rgb,empty,np.empty((0,3)),radius)
    write_overlay(directory/'query_boxes.ply',points,rgb,qboxes[qids],np.tile([255,140,25],(len(qids),1)),radius)
    write_overlay(directory/'detection_boxes.ply',points,rgb,dboxes[dids],colors,radius)
    np.savez_compressed(directory/'predictions.npz', **{f'query_{k}':v for k,v in query.items()},
                        detection_boxes=dboxes,detection_scores=scores,detection_labels=labels,
                        rendered_query_ids=qids,rendered_detection_ids=dids)
    manifest = dict(metadata, coordinate_system='axis_aligned_metric',
                    scale_policy='unchanged current model alignment; no additional scale recovery',
                    background='dataset scan; axis aligned once; no GT boxes rendered',
                    score_threshold=score_thr, query_count=len(qboxes), rendered_queries=len(qids),
                    detection_count=len(dboxes), rendered_detections=len(dids),
                    cloud_points=len(points), box_edge_radius=radius,
                    box_format='gravity center xyz + axis-aligned size xyz')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if preview:
        save_preview(directory,points,rgb,qboxes[qids],dboxes[dids],colors,elev,azim)
    return manifest


def json_safe(value):
    """Standard JSON retains nonfinite entries as null, never drops rows."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key,item in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_reference_scene(root, points, rgb, query, detections, metadata, *, score_thr=.3,
                           query_score_thr=None, query_topk=None, max_points=200000,
                           radius=None, preview=True, elev=55, azim=-65):
    """VGGT_Det directory/style convention; raw sidecars NEVER score filtered."""
    from matplotlib import colormaps
    root = Path(root)
    scene = str(metadata['scene_id'])
    if Path(scene).name != scene:
        raise ValueError('scene_id must be a filename component')
    points, rgb = np.asarray(points,np.float32), np.asarray(rgb,np.uint8)
    if not len(points) or points.shape != rgb.shape or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite scene XYZ and matching RGB')
    qboxes = np.asarray(query['query_boxes'])
    boxes, scores, labels = (np.asarray(detections[key]) for key in ('boxes','scores','labels'))
    qscores = np.asarray(query['scores'])
    if qboxes.shape != (len(qscores),6) or boxes.shape != (len(scores),6) or labels.shape != scores.shape:
        raise ValueError('Invalid box/score/label shapes')
    query_threshold = score_thr if query_score_thr is None else query_score_thr
    qmask = (np.isfinite(qscores) & (qscores >= query_threshold) &
             (np.asarray(query['source_indices']) >= 0) & np.isfinite(qboxes).all(-1) &
             (qboxes[:,3:] > 0).all(-1))
    qids = np.flatnonzero(qmask)
    if query_topk is not None:
        qids = qids[np.argsort(-qscores[qids],kind='stable')[:query_topk]]
    qmask = np.zeros(len(qboxes),bool); qmask[qids] = True
    dmask = np.isfinite(scores) & (scores >= score_thr) & np.isfinite(boxes).all(-1) & (boxes[:,3:] > 0).all(-1)
    dids = np.flatnonzero(dmask)
    class_names = list(metadata.get('class_names', [
        'cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
        'counter','desk','curtain','refrigerator','showercurtrain','toilet','sink','bathtub','garbagebin']))
    def class_name(label):
        return class_names[int(label)] if 0 <= int(label) < len(class_names) else f'class_{int(label)}'
    palette = np.rint(colormaps['tab20'](np.arange(20))[:,:3]*255).astype(np.uint8)
    dcolors = palette[labels[dids].astype(int)%20]
    qcolors = np.tile([227,39,39],(len(qids),1)).astype(np.uint8)
    shown_points, shown_rgb = points, rgb
    if len(points) > max_points:
        ids = np.linspace(0,len(points)-1,max_points).astype(int)
        shown_points, shown_rgb = points[ids], rgb[ids]
    radius = .01 if radius is None else radius
    empty_xyz, empty_boxes = np.empty((0,3),np.float32),np.empty((0,6),np.float32)
    qdir, ddir = root/'queries',root/'predictions'
    qdir.mkdir(parents=True,exist_ok=True); ddir.mkdir(parents=True,exist_ok=True)
    qprefix, dprefix = f'{scene}_initial_queries',f'{scene}_predictions'
    # Query initialization has extents in IGDet: retain boxes as well as centers.
    write_overlay(qdir/f'{qprefix}.ply',qboxes[qids,:3],qcolors,qboxes[qids],qcolors,radius)
    write_overlay(qdir/f'{scene}_gt_initial_queries.ply',np.concatenate((shown_points,qboxes[qids,:3])),
                  np.concatenate((shown_rgb,qcolors)),qboxes[qids],qcolors,radius)
    write_overlay(ddir/f'{dprefix}_boxes.ply',empty_xyz,empty_xyz,boxes[dids],dcolors,radius)
    write_overlay(ddir/f'{scene}_gt_predictions.ply',shown_points,shown_rgb,boxes[dids],dcolors,radius)
    # Corner order matches DepthInstance3DBoxes.corners (zero yaw).
    def corners(values):
        return (values[:,None,:3] + SIGNS[None]*values[:,None,3:]/2)[:,[0,1,3,2,4,5,7,6]]
    np.savez_compressed(qdir/f'{qprefix}.npz', **{f'raw_{key}':value for key,value in query.items()},
                        query_world=qboxes[:,:3], all_query_world=qboxes[:,:3],
                        boxes_center_size_world=qboxes, box_corners_world=corners(qboxes),
                        scores=qscores, labels=query.get('labels',np.full(len(qboxes),-1)),
                        gt_world=points,gt_rgb=rgb,displayed_mask=qmask,displayed_indices=qids)
    np.savez_compressed(ddir/f'{dprefix}.npz',gt_world=points,gt_rgb=rgb,
                        boxes_center_size_world=boxes,box_corners_world=corners(boxes),
                        scores=scores,labels=labels,displayed_mask=dmask,displayed_indices=dids)
    common = dict(metadata,scene=scene,gt_points=len(points),frames=len(metadata.get('input_image_paths',[])),
                  input_images=metadata.get('input_image_paths',[]),class_names=class_names,
                  coordinate_system='axis-aligned ScanNet world',
                  scale_policy='unchanged current model alignment; no additional scale recovery',
                  box_format='gravity center xyz + axis-aligned size xyz',
                  visualization_threshold_applied_to_saved_predictions=False,
                  elevation=elev,azimuth=azim,ply_box_edge_radius_m=radius)
    qlabels = np.asarray(query.get('labels',np.full(len(qboxes),-1)))
    all_queries = [dict(query_id=i,source_index=int(query['source_indices'][i]),
                        label=int(qlabels[i]),name=class_name(qlabels[i]),
                        score=float(qscores[i]),center_size=qboxes[i].tolist(),
                        fallback=bool(query.get('fallback_mask',np.zeros(len(qboxes),bool))[i]))
                   for i in range(len(qboxes))]
    all_predictions = [dict(label=int(labels[i]),name=class_name(labels[i]),score=float(scores[i]),
                            center_size=boxes[i].tolist()) for i in range(len(boxes))]
    qm = dict(common,model_queries=len(qboxes),queries=all_queries,query_count=len(qboxes),
              score_threshold=query_threshold,ply_queries=len(qids),displayed_indices=qids,
              displayed_queries=[all_queries[i] for i in qids],
              ply_files=dict(queries=f'{qprefix}.ply',scene_and_queries=f'{scene}_gt_initial_queries.ply'))
    dm = dict(common,predictions_after_nms=len(boxes),saved_prediction_stage='post_nms',
              score_threshold=score_thr,visualization_score_threshold=score_thr,
              displayed_boxes=len(dids),ply_predictions=len(dids),predictions=all_predictions,
              displayed_predictions=[all_predictions[i] for i in dids],displayed_indices=dids,
              ply_files=dict(boxes=f'{dprefix}_boxes.ply',scene_and_boxes=f'{scene}_gt_predictions.ply'))
    for path, content in ((qdir/f'{qprefix}.json',qm),(ddir/f'{dprefix}.json',dm)):
        path.write_text(json.dumps(json_safe(content),indent=2,allow_nan=False))
    if preview:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        combined = np.concatenate((shown_points,box_segments(qboxes[qids]).reshape(-1,3),box_segments(boxes[dids]).reshape(-1,3)))
        low, high = combined.min(0),combined.max(0)
        center, half = (low+high)/2,max(float((high-low).max())*.53,.01)
        for path, selected_boxes, colors, ids, is_query in (
                (qdir/f'{qprefix}.png',qboxes[qids],qcolors,qids,True),
                (ddir/f'{dprefix}.png',boxes[dids],dcolors,dids,False)):
            fig = Figure(figsize=(8,8),dpi=300,facecolor='white'); FigureCanvasAgg(fig)
            ax = fig.add_subplot(111,projection='3d',computed_zorder=False)
            ax.scatter(*shown_points.T,s=.6,c='#999999',alpha=.28,linewidths=0,depthshade=False,zorder=1,rasterized=True)
            if len(selected_boxes):
                ax.add_collection3d(Line3DCollection(box_segments(selected_boxes),
                    colors=np.repeat(colors/255.,12,axis=0),linewidths=1.3,zorder=3))
                if is_query:
                    ax.scatter(*selected_boxes[:,:3].T,s=20,c='#e32727',edgecolors='white',
                               linewidths=.35,depthshade=False,zorder=4)
                else:
                    for box, i, color in zip(selected_boxes,ids,colors/255.):
                        top = box[:3].copy(); top[2] += box[5]/2
                        ax.text(*top,f'{class_name(labels[i])} {scores[i]:.2f}',fontsize=6,color=color,zorder=4,
                                bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=.5))
            ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
            ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho'); ax.view_init(elev=elev,azim=azim); ax.set_axis_off()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1); fig.savefig(path,facecolor='white',dpi=300); fig.clear()
    else:
        for path in (qdir/f'{qprefix}.png',ddir/f'{dprefix}.png'):
            path.unlink(missing_ok=True)
    return dict(scene_id=scene,query_count=len(qboxes),rendered_queries=len(qids),
                detection_count=len(boxes),rendered_detections=len(dids))


def render_saved_results(args):
    """Reformat saved validation predictions without checkpoint/GPU inference."""
    source = args.from_results.resolve()
    if source == args.output_dir.resolve():
        raise ValueError('Use a separate output directory to preserve source raw results')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{args.output_dir} is not empty; use --overwrite or a new directory')
    index = json.loads((source/'index.json').read_text())
    scenes = index['completed']
    if args.scenes:
        if set(args.scenes)-set(scenes):
            raise ValueError('Unknown scenes in saved results')
        scenes = [scene for scene in scenes if scene in args.scenes]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    output_index = dict(source_results=str(source),selected_scenes=scenes,completed=[],
                        prediction_verification=index.get('prediction_verification'),layout='queries/predictions')
    for scene in scenes:
        if (source/'queries').is_dir():
            with np.load(source/'queries'/f'{scene}_initial_queries.npz') as data:
                query = {key[4:]:data[key] for key in data.files if key.startswith('raw_')}
                points, rgb = data['gt_world'],data['gt_rgb']
            with np.load(source/'predictions'/f'{scene}_predictions.npz') as data:
                detections = dict(boxes=data['boxes_center_size_world'],scores=data['scores'],labels=data['labels'])
            metadata = json.loads((source/'predictions'/f'{scene}_predictions.json').read_text())
            metadata = {key:metadata[key] for key in ('scene_id','checkpoint','seed','input_image_paths','class_names') if key in metadata}
        else:
            from plyfile import PlyData
            with np.load(source/scene/'predictions.npz') as data:
                query = {key[6:]:data[key] for key in data.files if key.startswith('query_')}
                detections = {key:data['detection_'+key] for key in ('boxes','scores','labels')}
            vertex = PlyData.read(source/scene/'scene.ply')['vertex'].data
            points = np.column_stack([vertex[key] for key in ('x','y','z')])
            rgb = np.column_stack([vertex[key] for key in ('red','green','blue')])
            metadata = json.loads((source/scene/'manifest.json').read_text())
            # Old display counts/thresholds describe the source rendering only.
            metadata = {key:metadata[key] for key in ('scene_id','checkpoint','seed',
                        'input_image_paths','class_names','gt_point_file','preview_note') if key in metadata}
        export_reference_scene(args.output_dir,points,rgb,query,detections,metadata,
            score_thr=args.score_thr,query_score_thr=args.query_score_thr,query_topk=args.query_topk,
            max_points=args.max_points,radius=args.line_radius,preview=not args.no_preview,
            elev=args.preview_elev,azim=args.preview_azim)
        output_index['completed'].append(scene)
        (args.output_dir/'index.json').write_text(json.dumps(output_index,indent=2))
        print(f'Exported {scene}',flush=True)


def parse_args():
    from mmengine.config import DictAction
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--from-results',type=Path,help='Re-render saved raw results without model inference')
    parser.add_argument('--query-score-thr',type=float,help='Query display threshold; default same as --score-thr')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'vis/outputs/visualization_all')
    parser.add_argument('--scenes',nargs='+',help='Optional filter; default ALL validation scenes')
    parser.add_argument('--max-scenes',type=int,default=None)
    parser.add_argument('--score-thr',type=float,default=.3)
    parser.add_argument('--query-topk',type=int)
    parser.add_argument('--max-points',type=int,default=200000)
    parser.add_argument('--line-radius',type=float)
    parser.add_argument('--device',default='auto',
                        help='auto (NPU, CUDA, CPU), npu[:index], cuda[:index], or cpu')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--verify-predictions',action='store_true',
                        help='Repeat first scene without capture and require exactly equal predictions')
    parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--preview-elev',type=float,default=55)
    parser.add_argument('--preview-azim',type=float,default=-65)
    parser.add_argument('--cfg-options',nargs='+',action=DictAction)
    args = parser.parse_args()
    for name in ('max_scenes','query_topk','max_points','line_radius'):
        value = getattr(args,name)
        if value is not None and value <= 0:
            parser.error(f'--{name.replace("_","-")} must be positive')
    if not 0 <= args.score_thr <= 1:
        parser.error('--score-thr must be in [0,1]')
    if args.query_score_thr is not None and not 0 <= args.query_score_thr <= 1:
        parser.error('--query-score-thr must be in [0,1]')
    if not args.from_results and not (args.config and args.checkpoint):
        parser.error('--config and --checkpoint are required without --from-results')
    return args


def compare_predictions(baseline, observed):
    """Exact equality is expected for transparent instrumentation on one GPU."""
    if len(baseline) != len(observed):
        raise AssertionError('Prediction batch lengths differ')
    max_difference = 0.0
    for first, second in zip(baseline, observed):
        for key in ('bboxes_3d', 'scores_3d', 'labels_3d'):
            a, b = getattr(first.pred_instances_3d,key), getattr(second.pred_instances_3d,key)
            if key == 'bboxes_3d':
                a, b = a.tensor, b.tensor
            if a.shape != b.shape:
                raise AssertionError(f'{key}: shape changed {a.shape} -> {b.shape}')
            if a.numel():
                max_difference = max(max_difference,float((a.float()-b.float()).abs().max()))
            if not torch.equal(a,b):
                raise AssertionError(f'{key}: predictions changed; max absolute difference {max_difference}')
    return dict(exact_match=True, max_absolute_difference=max_difference)


def main():
    args = parse_args()
    if args.from_results:
        render_saved_results(args)
        return
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    args.device = resolve_execution_device(args.device)
    print(f'Visualization inference device: {args.device}',flush=True)
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint, set_random_seed
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.utils import register_all_modules
    from mmdet3d.registry import MODELS
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules(init_default_scope=False)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope','mmdet3d'))
    # Match normal evaluation: this model uses CUDA cumsum, which PyTorch
    # rejects under globally forced deterministic algorithms. Reproducibility
    # is checked directly on real outputs by --verify-predictions instead.
    set_random_seed(args.seed, deterministic=False)
    if args.device.startswith('npu:'):
        torch.npu.manual_seed_all(args.seed)
    # Existing optional model visualization draws GT overlays; use this exporter.
    cfg.model.prediction_visualization = False
    if cfg.model.get('train_2d_only',False):
        raise ValueError('3D visualization requires a 3D model config')
    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.update(batch_size=1, num_workers=0, persistent_workers=False, drop_last=False,
                      sampler=dict(type='DefaultSampler',shuffle=False))
    loader = Runner.build_dataloader(loader_cfg, seed=args.seed, diff_rank_seed=False)
    # Validate requested IDs before inference and fail on misspellings.
    available = [str(loader.dataset.get_data_info(i)['scene_id']) for i in range(len(loader.dataset))]
    if len(set(available)) != len(available):
        raise ValueError('Duplicate scene IDs in validation dataset')
    if args.scenes and set(args.scenes)-set(available):
        raise ValueError(f'Unknown validation scenes: {set(args.scenes)-set(available)}')
    selected = [scene for scene in available if not args.scenes or scene in args.scenes]
    if args.max_scenes:
        selected = selected[:args.max_scenes]
    if not selected:
        raise ValueError('No validation scenes selected')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{args.output_dir} is not empty; use --overwrite or a new directory')
    model = MODELS.build(cfg.model).to(args.device)
    load_checkpoint(model,args.checkpoint,map_location='cpu',strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg.dump(str(args.output_dir/'resolved_config.py'))
    index = dict(config=str(Path(args.config).resolve()),checkpoint=str(Path(args.checkpoint).resolve()),
                 seed=args.seed,device=args.device,validation_scene_count=len(available),selected_scenes=selected,completed=[])
    with torch.no_grad():
        for batch in loader:
            scene = str(batch['data_samples'][0].metainfo['scene_id'])
            if scene not in selected:
                continue
            baseline = None
            if args.verify_predictions and not index['completed']:
                # Preserve RNG state so comparison uses identical stochastic input.
                with preserve_random_state(args.device):
                    baseline = model.test_step(copy.deepcopy(batch))
            with QueryCapture(model) as capture:
                predictions = model.test_step(batch)
                if capture.payload is None:
                    raise RuntimeError('Current model did not expose query initialization')
                query_payload = capture.payload
            if baseline is not None:
                index['prediction_verification'] = compare_predictions(baseline,predictions)
                print('Prediction verification: exact match with capture disabled',flush=True)
                del baseline
            sample = predictions[0]
            path, points = model._load_axis_aligned_gt_points(sample.metainfo)
            raw = np.fromfile(path,dtype=np.float32).reshape(-1,int(sample.metainfo.get('num_pts_feats',6)))
            # ScanNet processed points store RGB as 0..255. XYZ is aligned once
            # by the model helper; the raw colors require no geometric transform.
            rgb = np.clip(raw[:,3:6],0,255).astype(np.uint8) if raw.shape[1]>=6 else np.full(points.shape,160,np.uint8)
            result = sample.pred_instances_3d
            boxes = np.concatenate((numpy(result.bboxes_3d.gravity_center),numpy(result.bboxes_3d.tensor[:,3:6])),axis=-1)
            query = {key: value[0] for key,value in query_payload.items()}
            export_reference_scene(args.output_dir,points,rgb,query,
                         dict(boxes=boxes,scores=numpy(result.scores_3d),labels=numpy(result.labels_3d)),
                         dict(scene_id=scene,checkpoint=index['checkpoint'],seed=args.seed,
                              gt_point_file=str(path.resolve()),class_names=cfg.get('class_names',[]),
                              input_image_paths=sample.metainfo.get('img_path',[])),
                         score_thr=args.score_thr,query_score_thr=args.query_score_thr,
                         query_topk=args.query_topk,max_points=args.max_points,
                         radius=args.line_radius,preview=not args.no_preview,elev=args.preview_elev,azim=args.preview_azim)
            index['completed'].append(scene)
            (args.output_dir/'index.json').write_text(json.dumps(index,indent=2))
            print(f'[{len(index["completed"])}/{len(selected)}] {scene}: {args.output_dir}',flush=True)
            if len(index['completed']) == len(selected):
                break
    if index['completed'] != selected:
        raise RuntimeError('Validation loader did not export every selected scene')


if __name__ == '__main__':
    main()
