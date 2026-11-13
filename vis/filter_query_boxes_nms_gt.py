#!/usr/bin/env python3
"""Query boxes: class-aware greedy 3D NMS (default 0.5), then GT max-IoU selection.

Reads all saved raw query rows, ignoring previous visualization score masks.
Exports a GT-guided visualization subset with complete raw/NMS provenance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vis.filter_predictions_by_gt_iou import box_iou_3d,select_same_class_max_iou,save_scene
from vis.filter_predictions_with_gt import build_gt_dataset


def nms_query_boxes(boxes,scores,labels,threshold=.5,*,class_agnostic=False,valid_mask=None):
    """IoU > threshold suppresses lower score; ties keep smaller source row."""
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('NMS threshold must be finite and in [0,1]')
    boxes,scores,labels=np.asarray(boxes),np.asarray(scores),np.asarray(labels)
    if boxes.shape!=(len(scores),6) or scores.ndim!=1 or labels.shape!=scores.shape:
        raise ValueError('Invalid query box/score/label shapes')
    valid=np.isfinite(boxes).all(1)&(boxes[:,3:]>0).all(1)&np.isfinite(scores)&(labels>=0)
    if valid_mask is not None:valid&=np.asarray(valid_mask,bool)
    ids=np.flatnonzero(valid)
    order=ids[np.lexsort((ids,-scores[ids]))]
    kept=[]
    while len(order):
        current=int(order[0]);kept.append(current);rest=order[1:]
        if not len(rest):break
        overlaps=box_iou_3d(boxes[current:current+1],boxes[rest])[0]
        same=np.ones(len(rest),bool) if class_agnostic else labels[rest]==labels[current]
        order=rest[~(same&(overlaps>threshold))]
    return np.asarray(kept,np.int64)


def nms_then_match(gt_boxes,gt_labels,boxes,scores,labels,*,nms_thr=.5,min_iou=0.,class_agnostic=False,valid_mask=None):
    nms_ids=nms_query_boxes(boxes,scores,labels,nms_thr,class_agnostic=class_agnostic,valid_mask=valid_mask)
    selected=select_same_class_max_iou(gt_boxes,gt_labels,boxes[nms_ids],scores[nms_ids],labels[nms_ids],min_iou)
    # Remap every candidate/source index to the ORIGINAL query row. Pair indices
    # into the final exported list do not change, preserving many-to-one matches.
    for key in ('source_indices','matched_prediction_source_indices'):
        selected[key]=nms_ids[selected[key]]
    best=selected['gt_best_prediction_source_indices'].copy();valid=best>=0
    best[valid]=nms_ids[best[valid]];selected['gt_best_prediction_source_indices']=best
    return nms_ids,selected


def read_queries(path):
    with np.load(path,allow_pickle=False) as data:
        boxes=np.array(data['boxes_center_size_world'],copy=True)
        scores=np.array(data['scores'],copy=True);labels=np.array(data['labels'],copy=True)
        # In the exporter, -1 means synthetic origin padding. Actual FPS
        # observations (including fallback=True) remain eligible.
        source=np.array(data['raw_source_indices'],copy=True) if 'raw_source_indices' in data else np.arange(len(scores))
        extras={f'input_{key}':np.array(data[key],copy=True) for key in data.files if key.startswith('raw_')}
    return boxes,scores,labels,source,extras


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'vis/outputs/visualization_all/queries')
    parser.add_argument('--config',default=str(ROOT/'configs/igdet/igdet_scannet.py'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'vis/outputs/query_nms_gt_iou')
    parser.add_argument('--nms-iou-thr',type=float,default=.5)
    parser.add_argument('--min-iou',type=float,default=0.)
    parser.add_argument('--class-agnostic-nms',action='store_true',help='Suppress overlaps across classes; default same-class NMS')
    parser.add_argument('--scenes',nargs='+');parser.add_argument('--max-points',type=int,default=200000)
    parser.add_argument('--line-radius',type=float,default=.01)
    parser.add_argument('--preview-elev',type=float,default=55);parser.add_argument('--preview-azim',type=float,default=-65)
    parser.add_argument('--label-text',action='store_true');parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    for name in ('nms_iou_thr','min_iou'):
        value=getattr(args,name)
        if not np.isfinite(value) or not 0<=value<=1:parser.error(f'{name} must be in [0,1]')
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
        nms_ids,selection=nms_then_match(gt_boxes,ann['gt_labels_3d'],boxes,scores,labels,
            nms_thr=args.nms_iou_thr,min_iou=args.min_iou,class_agnostic=args.class_agnostic_nms,valid_mask=observation_sources>=0)
        stage=dict(nms_source_indices=nms_ids,nms_boxes_center_size_world=boxes[nms_ids],
            nms_scores=scores[nms_ids],nms_labels=labels[nms_ids],nms_iou_threshold=np.float64(args.nms_iou_thr),
            query_observation_source_indices=observation_sources,
            class_agnostic_nms=np.bool_(args.class_agnostic_nms),**extras)
        meta=dict(input_kind='query_initialization_boxes',nms_iou_threshold=args.nms_iou_thr,
            nms_class_aware=not args.class_agnostic_nms,nms_query_count=len(nms_ids),nms_source_indices=nms_ids.tolist(),
            pipeline='all raw query rows -> valid observed queries -> score-ranked 3D NMS -> per-GT same-class maximum IoU',
            index_convention='source_indices/matched_prediction_source_indices refer to original query row, not observation index')
        summary=save_scene(args.output_dir,scene,path,points,rgb,gt_boxes,ann['gt_labels_3d'],boxes,scores,labels,selection,
            min_iou=args.min_iou,axis_alignment=axis,class_names=dataset.metainfo['classes'],max_points=args.max_points,
            radius=args.line_radius,preview=not args.no_preview,elev=args.preview_elev,azim=args.preview_azim,labels_text=args.label_text,
            name_suffix='query_nms_gt_iou',overlay_suffix='query_nms_gt_overlay',extra_data=stage,extra_metadata=meta)
        summary['nms_query_count']=len(nms_ids);summaries.append(summary)
        (args.output_dir/'index.json').write_text(json.dumps(dict(gt_guided_selection=True,input_kind='query_initialization_boxes',
            input_dir=str(folder.resolve()),nms_iou_threshold=args.nms_iou_thr,nms_class_aware=not args.class_agnostic_nms,
            min_iou=args.min_iou,completed=summaries),indent=2))
        print(f'{scene}: {len(boxes)} raw -> {len(nms_ids)} NMS -> {len(selection["source_indices"])} GT-selected queries',flush=True)


if __name__=='__main__':main()
