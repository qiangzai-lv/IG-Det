#!/usr/bin/env python3
"""Score-filter saved axis-aligned metric detections and overlay GT on RGB scans.

No model/checkpoint inference, GT matching, box correction or new NMS. Filtered
arrays are the main NPZ fields; raw_* arrays preserve every input prediction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from vis.visualize_val_predictions import box_segments, write_overlay, json_safe

GT_COLOR = np.array([40,190,70],np.uint8)
PRED_COLOR = np.array([235,55,45],np.uint8)


def discover_predictions(path):
    path = Path(path)
    if path.is_file():
        paths = [path]
    else:
        folder = path/'predictions' if (path/'predictions').is_dir() else path
        paths = sorted(folder.glob('*_predictions.npz'))
        if not paths:
            paths = sorted(path.glob('scene*/predictions.npz'))
    results = {}
    for file in paths:
        scene = file.parent.name if file.name == 'predictions.npz' else file.stem.removesuffix('_predictions')
        if re.fullmatch(r'scene\d{4}_\d{2}',scene) is None:
            raise ValueError(f'Cannot determine scene ID from {file}')
        if scene in results:
            raise ValueError(f'Duplicate scene {scene}')
        results[scene] = file
    if not results:
        raise FileNotFoundError(f'No saved prediction NPZ found under {path}')
    return results


def read_predictions(path):
    with np.load(path,allow_pickle=False) as data:
        if 'boxes_center_size_world' in data:
            keys = ('boxes_center_size_world','scores','labels')
        else:
            keys = ('detection_boxes','detection_scores','detection_labels')
        boxes,scores,labels = (np.array(data[key],copy=True) for key in keys)
    if boxes.shape != (len(scores),6) or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError(f'Expected center-size [N,6], scores/labels [N]: {path}')
    if not np.issubdtype(labels.dtype,np.integer):
        raise ValueError(f'Expected integer category labels: {path}')
    return boxes,scores,labels


def filter_predictions(boxes,scores,labels,threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('score threshold must be finite and in [0,1]')
    keep = np.isfinite(scores) & (scores >= threshold)
    keep &= np.isfinite(boxes).all(1) & (boxes[:,3:] > 0).all(1)
    # Unknown labels remain raw but do not produce a displayed detection.
    keep &= labels >= 0
    return keep


def render(path,points,rgb,gt_boxes,boxes,scores,labels,names,elev,azim,labels_text):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig=Figure(figsize=(8,8),dpi=300,facecolor='white'); FigureCanvasAgg(fig)
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    ax.scatter(*points.T,c=rgb/255.,s=.65,alpha=.65,linewidths=0,depthshade=False,zorder=1,rasterized=True)
    for values,color,style in ((gt_boxes,GT_COLOR,'dashed'),(boxes,PRED_COLOR,'solid')):
        if len(values):
            ax.add_collection3d(Line3DCollection(box_segments(values),colors=[color/255.],
                                                linewidths=1.2,linestyles=style,zorder=3))
    if labels_text:
        for box,score,label in zip(boxes,scores,labels):
            top=box[:3].copy();top[2]+=box[5]/2
            name=names[int(label)] if 0 <= label < len(names) else str(label)
            ax.text(*top,f'{name} {score:.3f}',fontsize=6,color=PRED_COLOR/255.,zorder=4)
    all_xyz=np.concatenate((points,box_segments(gt_boxes).reshape(-1,3),box_segments(boxes).reshape(-1,3)))
    low,high=all_xyz.min(0),all_xyz.max(0);center=(low+high)/2;half=max(float((high-low).max())*.53,.01)
    ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
    ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=elev,azim=azim);ax.set_axis_off()
    ax.legend(handles=[Line2D([0],[0],color=GT_COLOR/255.,linestyle='--',label=f'GT ({len(gt_boxes)})'),
                       Line2D([0],[0],color=PRED_COLOR/255.,label=f'Prediction ({len(boxes)})')],loc='upper left')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1);fig.savefig(path,dpi=300,facecolor='white');fig.clear()


def save_filtered_scene(output,scene,source,points,rgb,gt_boxes,gt_labels,boxes,scores,labels,*,
                        threshold,class_names=(),axis_alignment=None,max_points=200000,
                        radius=.01,preview=True,elev=55,azim=-65,labels_text=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    points,rgb=np.asarray(points,np.float32),np.asarray(rgb,np.uint8)
    if not len(points) or points.shape!=rgb.shape or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite RGB point cloud')
    if gt_boxes.shape!=(len(gt_labels),6) or not np.isfinite(gt_boxes).all() or (gt_boxes[:,3:]<=0).any():
        raise ValueError('Invalid GT boxes')
    keep=filter_predictions(boxes,scores,labels,threshold);ids=np.flatnonzero(keep)
    prefix=output/f'{scene}_filtered_predictions'
    np.savez_compressed(prefix.with_suffix('.npz'),boxes_center_size_world=boxes[keep],scores=scores[keep],labels=labels[keep],
        source_indices=ids,selection_mask=keep,score_threshold=np.float64(threshold),
        raw_boxes_center_size_world=boxes,raw_scores=scores,raw_labels=labels,
        gt_boxes_center_size_world=gt_boxes,gt_labels=gt_labels,gt_world=points,gt_rgb=rgb,
        axis_alignment=np.eye(4) if axis_alignment is None else axis_alignment,
        coordinate_system=np.array('axis_aligned_metric'),source_prediction_file=np.array(str(Path(source).resolve())))
    def records(values,conf,categories):
        return [dict(label=int(label),name=class_names[int(label)] if 0 <= label < len(class_names) else str(label),
                     center_size=box.tolist(),**({} if conf is None else dict(score=float(conf[i]))))
                for i,(box,label) in enumerate(zip(values,categories))]
    metadata=dict(scene_id=scene,source_prediction_file=str(Path(source).resolve()),score_threshold=threshold,
        raw_prediction_count=len(boxes),filtered_prediction_count=len(ids),gt_count=len(gt_boxes),source_indices=ids,
        coordinate_system='axis_aligned_metric',box_format='gravity center xyz + axis-aligned size xyz',
        processing='score >= threshold; remove invalid boxes/scores and negative labels; no GT-based selection',
        predictions=records(boxes,scores,labels),filtered_predictions=records(boxes[keep],scores[keep],labels[keep]),
        gt_boxes=records(gt_boxes,None,gt_labels),colors=dict(gt='green',prediction='red'),class_names=list(class_names))
    prefix.with_suffix('.json').write_text(json.dumps(json_safe(metadata),indent=2,allow_nan=False))
    selected_points,selected_rgb=points,rgb
    if len(points)>max_points:
        indices=np.linspace(0,len(points)-1,max_points).astype(int)
        selected_points,selected_rgb=points[indices],rgb[indices]
    overlay_boxes=np.concatenate((gt_boxes,boxes[keep]))
    colors=np.concatenate((np.tile(GT_COLOR,(len(gt_boxes),1)),np.tile(PRED_COLOR,(len(ids),1))))
    write_overlay(output/f'{scene}_gt_predictions.ply',selected_points,selected_rgb,overlay_boxes,colors,radius)
    png=output/f'{scene}_gt_predictions.png'
    if preview:
        render(png,selected_points,selected_rgb,gt_boxes,boxes[keep],scores[keep],labels[keep],
               class_names,elev,azim,labels_text)
    else:
        png.unlink(missing_ok=True)
    return {key:metadata[key] for key in ('scene_id','raw_prediction_count','filtered_prediction_count','gt_count')}


def build_gt_dataset(config):
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmengine.registry import init_default_scope
    from mmdet3d.utils import register_all_modules
    from mmdet3d.registry import DATASETS
    cfg=Config.fromfile(config)
    register_all_modules(init_default_scope=False)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope','mmdet3d'))
    dataset_cfg=cfg.val_dataloader.dataset.copy()
    dataset_cfg.update(pipeline=[],test_mode=True,load_eval_anns=True)
    return DATASETS.build(dataset_cfg)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'vis/outputs/visualization_all')
    parser.add_argument('--config',default=str(ROOT/'configs/igdet/igdet_scannet.py'),help='Validation dataset config; no model is built')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'vis/outputs/filtered_predictions')
    parser.add_argument('--score-thr',type=float,required=True)
    parser.add_argument('--scenes',nargs='+',help='Default: all saved scenes; every scene must have validation GT')
    parser.add_argument('--max-points',type=int,default=200000,help='PLY/PNG background only; NPZ keeps all points')
    parser.add_argument('--line-radius',type=float,default=.01)
    parser.add_argument('--preview-elev',type=float,default=55);parser.add_argument('--preview-azim',type=float,default=-65)
    parser.add_argument('--label-text',action='store_true');parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.score_thr) or not 0<=args.score_thr<=1 or args.max_points<=0 or not np.isfinite(args.line_radius) or args.line_radius<=0:
        parser.error('Require score threshold in [0,1], positive max-points and finite positive line-radius')
    if args.output_dir.resolve().is_relative_to(args.input_dir.resolve()):
        parser.error('Output must be separate from the input directory to preserve source results')
    files=discover_predictions(args.input_dir)
    if args.scenes:
        if set(args.scenes)-set(files):
            parser.error(f'Missing predictions for {set(args.scenes)-set(files)}')
        files={scene:path for scene,path in files.items() if scene in args.scenes}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError('Output is not empty; use another directory or --overwrite')
    dataset=build_gt_dataset(args.config)
    lookup={dataset.get_data_info(i)['scene_id']:i for i in range(len(dataset))}
    if set(files)-set(lookup):
        raise ValueError(f'Saved scenes missing from validation GT: {set(files)-set(lookup)}; use the matching --config')
    summaries=[]
    for scene,source in files.items():
        info=dataset.get_data_info(lookup[scene]);ann=dataset.get_ann_info(lookup[scene])
        gt=ann['gt_bboxes_3d']
        # Annotation boxes already use aligned scene axes; do not rotate twice.
        gt_boxes=np.concatenate((gt.gravity_center.numpy(),gt.tensor[:,3:6].numpy()),axis=1)
        raw=np.fromfile(info['lidar_path'],np.float32).reshape(-1,int(info.get('num_pts_feats',6)))
        if raw.shape[1]<6:
            raise ValueError(f'Need XYZRGB scan: {info["lidar_path"]}')
        axis=np.asarray(info['axis_align_matrix'],np.float32)
        points=raw[:,:3]@axis[:3,:3].T+axis[:3,3]
        rgb=np.clip(np.rint(raw[:,3:6]),0,255).astype(np.uint8)
        boxes,scores,labels=read_predictions(source)
        summary=save_filtered_scene(args.output_dir,scene,source,points,rgb,gt_boxes,ann['gt_labels_3d'],boxes,scores,labels,
             threshold=args.score_thr,class_names=dataset.metainfo['classes'],axis_alignment=axis,
             max_points=args.max_points,radius=args.line_radius,preview=not args.no_preview,
             elev=args.preview_elev,azim=args.preview_azim,labels_text=args.label_text)
        summaries.append(summary)
        (args.output_dir/'index.json').write_text(json.dumps(dict(input_dir=str(args.input_dir.resolve()),
             config=str(Path(args.config).resolve()),score_threshold=args.score_thr,completed=summaries),indent=2))
        print(f'{scene}: {summary["raw_prediction_count"]} -> {summary["filtered_prediction_count"]} predictions; GT={summary["gt_count"]}',flush=True)


if __name__=='__main__':
    main()
