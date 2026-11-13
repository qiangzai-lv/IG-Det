#!/usr/bin/env python3
"""Query initialization boxes: score-ranked 3D NMS, then Top K. GT is display only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vis.filter_query_boxes_nms_gt import nms_query_boxes,read_queries
from vis.filter_predictions_with_gt import build_gt_dataset,render,GT_COLOR,PRED_COLOR
from vis.visualize_val_predictions import write_overlay,json_safe


def nms_then_topk(boxes,scores,labels,*,top_k=64,nms_thr=.5,class_agnostic=False,valid_mask=None):
    """No GT input. Keep NMS score order, then truncate globally to at most K."""
    if isinstance(top_k,bool) or not isinstance(top_k,(int,np.integer)) or top_k<=0:
        raise ValueError('top_k must be a positive integer')
    kept=nms_query_boxes(boxes,scores,labels,nms_thr,class_agnostic=class_agnostic,valid_mask=valid_mask)
    return kept,kept[:top_k].copy()


def save_topk_scene(output,scene,source,points,rgb,gt_boxes,gt_labels,boxes,scores,labels,nms_ids,selected_ids,*,
                    top_k,nms_thr,class_agnostic,axis_alignment,observation_sources,extras=None,
                    class_names=(),max_points=200000,radius=.01,preview=True,elev=55,azim=-65,labels_text=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prefix=output/f'{scene}_query_nms_topk'
    np.savez_compressed(prefix.with_suffix('.npz'),boxes_center_size_world=boxes[selected_ids],scores=scores[selected_ids],
        labels=labels[selected_ids],source_indices=selected_ids,raw_boxes_center_size_world=boxes,raw_scores=scores,raw_labels=labels,
        nms_source_indices=nms_ids,nms_boxes_center_size_world=boxes[nms_ids],nms_scores=scores[nms_ids],nms_labels=labels[nms_ids],
        top_k=np.int64(top_k),nms_iou_threshold=np.float64(nms_thr),class_agnostic_nms=np.bool_(class_agnostic),
        gt_guided_selection=np.bool_(False),gt_boxes_center_size_world=gt_boxes,gt_labels=gt_labels,gt_world=points,gt_rgb=rgb,
        axis_alignment=axis_alignment,query_observation_source_indices=observation_sources,
        source_query_file=np.array(str(Path(source).resolve())),coordinate_system=np.array('axis_aligned_metric'),**(extras or {}))
    def record(i):
        label=int(labels[i]);name=class_names[label] if 0<=label<len(class_names) else str(label)
        return dict(source_index=int(i),label=label,name=name,score=float(scores[i]),center_size=boxes[i].tolist())
    metadata=dict(scene_id=scene,source_query_file=str(Path(source).resolve()),gt_guided_selection=False,
        pipeline='valid observed queries -> score-ranked 3D NMS -> global score Top K; GT is display only',
        nms_iou_threshold=nms_thr,nms_class_aware=not class_agnostic,top_k=top_k,
        raw_query_count=len(boxes),nms_query_count=len(nms_ids),selected_query_count=len(selected_ids),gt_count=len(gt_boxes),
        nms_source_indices=nms_ids,source_indices=selected_ids,coordinate_system='axis_aligned_metric',
        queries=[record(i) for i in range(len(boxes))],selected_queries=[record(i) for i in selected_ids])
    prefix.with_suffix('.json').write_text(json.dumps(json_safe(metadata),indent=2,allow_nan=False))
    xyz,colors=points,rgb
    if len(xyz)>max_points:
        point_ids=np.linspace(0,len(xyz)-1,max_points).astype(int);xyz,colors=xyz[point_ids],colors[point_ids]
    overlay=np.concatenate((gt_boxes,boxes[selected_ids]))
    box_colors=np.concatenate((np.tile(GT_COLOR,(len(gt_boxes),1)),np.tile(PRED_COLOR,(len(selected_ids),1))))
    write_overlay(output/f'{scene}_query_nms_topk_overlay.ply',xyz,colors,overlay,box_colors,radius)
    png=output/f'{scene}_query_nms_topk_overlay.png'
    if preview:
        render(png,xyz,colors,gt_boxes,boxes[selected_ids],scores[selected_ids],labels[selected_ids],class_names,elev,azim,labels_text)
    else:png.unlink(missing_ok=True)
    return {key:metadata[key] for key in ('scene_id','raw_query_count','nms_query_count','selected_query_count','gt_count')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'vis/outputs/visualization_all/queries')
    parser.add_argument('--config',default=str(ROOT/'configs/igdet/igdet_scannet.py'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'vis/outputs/query_nms_topk')
    parser.add_argument('--nms-iou-thr',type=float,default=.5)
    parser.add_argument('--top-k',type=int,default=64,help='Maximum queries AFTER NMS; default 64')
    parser.add_argument('--class-agnostic-nms',action='store_true',help='Suppress overlaps across classes; default same-class NMS')
    parser.add_argument('--scenes',nargs='+');parser.add_argument('--max-points',type=int,default=200000)
    parser.add_argument('--line-radius',type=float,default=.01)
    parser.add_argument('--preview-elev',type=float,default=55);parser.add_argument('--preview-azim',type=float,default=-65)
    parser.add_argument('--label-text',action='store_true');parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    for name in ('nms_iou_thr',):
        value=getattr(args,name)
        if not np.isfinite(value) or not 0<=value<=1:parser.error(f'{name} must be in [0,1]')
    if args.top_k<=0:parser.error('--top-k must be positive')
    if args.max_points<=0 or not np.isfinite(args.line_radius) or args.line_radius<=0:parser.error('max-points/line-radius must be positive')
    if args.output_dir.resolve().is_relative_to(args.input_dir.resolve()):parser.error('Output must be separate from input')
    folder=args.input_dir/'queries' if (args.input_dir/'queries').is_dir() else args.input_dir
    files={p.name.removesuffix('_initial_queries.npz'):p for p in sorted(folder.glob('*_initial_queries.npz'))}
    if not files:raise FileNotFoundError(f'No initial query NPZ found in {folder}')
    if any(re.fullmatch(r'scene\d{4}_\d{2}',s) is None for s in files):raise ValueError('Invalid scene filename')
    if args.scenes:
        if set(args.scenes)-set(files):parser.error('Requested scene missing query NPZ')
        files={s:p for s,p in files.items() if s in args.scenes}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:raise FileExistsError('Output exists; use --overwrite or a new directory')
    dataset=build_gt_dataset(args.config)
    lookup={dataset.get_data_info(i)['scene_id']:i for i in range(len(dataset))}
    if set(files)-set(lookup):raise ValueError(f'GT missing for {set(files)-set(lookup)}')
    summaries=[]
    for scene,path in files.items():
        info=dataset.get_data_info(lookup[scene]);ann=dataset.get_ann_info(lookup[scene]);gt=ann['gt_bboxes_3d']
        gt_boxes=np.concatenate((gt.gravity_center.numpy(),gt.tensor[:,3:6].numpy()),axis=1)
        raw=np.fromfile(info['lidar_path'],np.float32).reshape(-1,int(info.get('num_pts_feats',6)))
        if raw.shape[1]<6:raise ValueError('Expected XYZRGB scene points')
        axis=np.asarray(info['axis_align_matrix'],np.float32)
        points=raw[:,:3]@axis[:3,:3].T+axis[:3,3];rgb=np.clip(np.rint(raw[:,3:6]),0,255).astype(np.uint8)
        boxes,scores,labels,observation_sources,extras=read_queries(path)
        nms_ids,selected_ids=nms_then_topk(boxes,scores,labels,top_k=args.top_k,
            nms_thr=args.nms_iou_thr,class_agnostic=args.class_agnostic_nms,valid_mask=observation_sources>=0)
        summary=save_topk_scene(args.output_dir,scene,path,points,rgb,gt_boxes,ann['gt_labels_3d'],
            boxes,scores,labels,nms_ids,selected_ids,top_k=args.top_k,nms_thr=args.nms_iou_thr,
            class_agnostic=args.class_agnostic_nms,axis_alignment=axis,observation_sources=observation_sources,extras=extras,
            class_names=dataset.metainfo['classes'],max_points=args.max_points,radius=args.line_radius,
            preview=not args.no_preview,elev=args.preview_elev,azim=args.preview_azim,labels_text=args.label_text)
        summaries.append(summary)
        (args.output_dir/'index.json').write_text(json.dumps(dict(gt_guided_selection=False,input_kind='query_initialization_boxes',
            input_dir=str(folder.resolve()),nms_iou_threshold=args.nms_iou_thr,nms_class_aware=not args.class_agnostic_nms,
            top_k=args.top_k,completed=summaries),indent=2))
        print(f'{scene}: {len(boxes)} raw -> {len(nms_ids)} NMS -> {len(selected_ids)} Top-K queries',flush=True)



if __name__=='__main__':main()
