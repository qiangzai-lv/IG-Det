#!/usr/bin/env python3
"""GT-guided visualization subset: for each GT choose the same-class max-IoU prediction.

This is an oracle subset for inspection, not unfiltered model output or an
independent evaluation. Never changes predicted coordinates/sizes/scores.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from vis.filter_predictions_with_gt import (
    discover_predictions,read_predictions,build_gt_dataset,render,GT_COLOR,PRED_COLOR)
from vis.visualize_val_predictions import write_overlay,json_safe


def box_iou_3d(gt_boxes,pred_boxes):
    """Pairwise axis-aligned volumetric IoU for gravity-center/size boxes."""
    gt=np.asarray(gt_boxes,np.float64).reshape(-1,6)
    pred=np.asarray(pred_boxes,np.float64).reshape(-1,6)
    low=np.maximum(gt[:,None,:3]-gt[:,None,3:]/2,pred[None,:,:3]-pred[None,:,3:]/2)
    high=np.minimum(gt[:,None,:3]+gt[:,None,3:]/2,pred[None,:,:3]+pred[None,:,3:]/2)
    intersection=np.maximum(high-low,0).prod(-1)
    union=gt[:,3:].prod(-1)[:,None]+pred[:,3:].prod(-1)[None,:]-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)


def select_same_class_max_iou(gt_boxes,gt_labels,boxes,scores,labels,min_iou=0.):
    """Independent per-GT maximum; deduplicate only the exported box list.

    Ties: larger score, then smaller source index. The same prediction may be
    best for multiple GTs; all associations are preserved. Zero overlap is an
    unmatched GT, even when min_iou=0.
    """
    if not np.isfinite(min_iou) or not 0<=min_iou<=1:
        raise ValueError('min_iou must be finite and in [0,1]')
    gt_boxes,boxes=np.asarray(gt_boxes),np.asarray(boxes)
    gt_labels,scores,labels=np.asarray(gt_labels),np.asarray(scores),np.asarray(labels)
    if gt_boxes.shape!=(len(gt_labels),6) or boxes.shape!=(len(scores),6) or labels.shape!=scores.shape:
        raise ValueError('Invalid box/label/score shapes')
    if not np.isfinite(gt_boxes).all() or (gt_boxes[:,3:]<=0).any():
        raise ValueError('Invalid GT geometry')
    valid=np.isfinite(boxes).all(1)&(boxes[:,3:]>0).all(1)&np.isfinite(scores)&(labels>=0)
    candidates=np.flatnonzero(valid)
    ious=box_iou_3d(gt_boxes,boxes[candidates])
    best=np.full(len(gt_boxes),-1,np.int64)
    best_iou=np.zeros(len(gt_boxes),np.float64)
    for g,label in enumerate(gt_labels):
        same=np.flatnonzero(labels[candidates]==label) if label>=0 else np.empty(0,int)
        if not len(same):
            continue
        # lexsort's last key is primary; no approximate tie grouping.
        order=np.lexsort((candidates[same],-scores[candidates[same]],-ious[g,same]))
        local=same[order[0]]
        best[g]=candidates[local];best_iou[g]=ious[g,local]
    matched=(best>=0)&(best_iou>0)&(best_iou>=min_iou)
    gt_ids=np.flatnonzero(matched)
    source_ids=np.unique(best[matched])
    return dict(source_indices=source_ids,matched_gt_indices=gt_ids,
                matched_prediction_source_indices=best[matched],matched_ious=best_iou[matched],
                matched_prediction_indices=np.searchsorted(source_ids,best[matched]),
                gt_best_prediction_source_indices=best,gt_best_ious=best_iou,
                unmatched_gt_indices=np.flatnonzero(~matched))


def save_scene(output,scene,source,points,rgb,gt_boxes,gt_labels,boxes,scores,labels,selection,*,
               min_iou,axis_alignment,class_names=(),max_points=200000,radius=.01,
               preview=True,elev=55,azim=-65,labels_text=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ids=selection['source_indices']
    path=output/f'{scene}_gt_iou_predictions'
    np.savez_compressed(path.with_suffix('.npz'),boxes_center_size_world=boxes[ids],scores=scores[ids],labels=labels[ids],
        raw_boxes_center_size_world=boxes,raw_scores=scores,raw_labels=labels,
        gt_boxes_center_size_world=gt_boxes,gt_labels=gt_labels,gt_world=points,gt_rgb=rgb,
        axis_alignment=axis_alignment,min_iou=np.float64(min_iou),gt_guided_selection=np.bool_(True),
        source_prediction_file=np.array(str(Path(source).resolve())),
        coordinate_system=np.array('axis_aligned_metric'),**selection)
    metadata=dict(scene_id=scene,gt_guided_selection=True,
        selection_rule='per-GT same-class maximum volumetric 3D IoU; IoU > 0 and >= min_iou',
        tie_break='higher confidence then smaller source index',
        duplicate_policy='a prediction may match multiple GTs; exported prediction boxes are unique',
        source_prediction_file=str(Path(source).resolve()),min_iou=min_iou,
        raw_prediction_count=len(boxes),selected_prediction_count=len(ids),gt_count=len(gt_boxes),
        matched_gt_count=len(selection['matched_gt_indices']),unmatched_gt_indices=selection['unmatched_gt_indices'],
        source_indices=ids,class_names=list(class_names),coordinate_system='axis_aligned_metric',
        matches=[dict(gt_index=int(g),prediction_source_index=int(p),selected_prediction_index=int(s),iou=float(i))
                 for g,p,s,i in zip(selection['matched_gt_indices'],selection['matched_prediction_source_indices'],
                                   selection['matched_prediction_indices'],selection['matched_ious'])])
    metadata['selected_predictions']=[dict(source_index=int(i),label=int(labels[i]),score=float(scores[i]),
                                           center_size=boxes[i].tolist()) for i in ids]
    path.with_suffix('.json').write_text(json.dumps(json_safe(metadata),indent=2,allow_nan=False))
    xyz,colors=points,rgb
    if len(xyz)>max_points:
        point_ids=np.linspace(0,len(xyz)-1,max_points).astype(int);xyz,colors=xyz[point_ids],colors[point_ids]
    overlay=np.concatenate((gt_boxes,boxes[ids]))
    box_colors=np.concatenate((np.tile(GT_COLOR,(len(gt_boxes),1)),np.tile(PRED_COLOR,(len(ids),1))))
    write_overlay(output/f'{scene}_gt_iou_overlay.ply',xyz,colors,overlay,box_colors,radius)
    image=output/f'{scene}_gt_iou_overlay.png'
    if preview:
        render(image,xyz,colors,gt_boxes,boxes[ids],scores[ids],labels[ids],class_names,elev,azim,labels_text)
    else:
        image.unlink(missing_ok=True)
    return {key:metadata[key] for key in ('scene_id','gt_count','raw_prediction_count','selected_prediction_count','matched_gt_count')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'vis/outputs/visualization_all')
    parser.add_argument('--config',default=str(ROOT/'configs/igdet/igdet_scannet.py'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'vis/outputs/gt_iou_predictions')
    parser.add_argument('--min-iou',type=float,default=0.,help='Accept best same-class IoU >= this value; zero overlap always omitted')
    parser.add_argument('--scenes',nargs='+')
    parser.add_argument('--max-points',type=int,default=200000)
    parser.add_argument('--line-radius',type=float,default=.01)
    parser.add_argument('--preview-elev',type=float,default=55);parser.add_argument('--preview-azim',type=float,default=-65)
    parser.add_argument('--label-text',action='store_true');parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.min_iou) or not 0<=args.min_iou<=1 or args.max_points<=0 or not np.isfinite(args.line_radius) or args.line_radius<=0:
        parser.error('Require min-iou in [0,1], positive max-points and line-radius')
    if args.output_dir.resolve().is_relative_to(args.input_dir.resolve()):
        parser.error('Output must be separate from input')
    files=discover_predictions(args.input_dir)
    if args.scenes:
        if set(args.scenes)-set(files):parser.error('Some requested scenes have no saved predictions')
        files={s:p for s,p in files.items() if s in args.scenes}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError('Output is not empty; use a new directory or --overwrite')
    dataset=build_gt_dataset(args.config)
    lookup={dataset.get_data_info(i)['scene_id']:i for i in range(len(dataset))}
    if set(files)-set(lookup):raise ValueError(f'Scenes missing from validation GT: {set(files)-set(lookup)}')
    summaries=[]
    for scene,source in files.items():
        info=dataset.get_data_info(lookup[scene]);ann=dataset.get_ann_info(lookup[scene]);gt=ann['gt_bboxes_3d']
        gt_boxes=np.concatenate((gt.gravity_center.numpy(),gt.tensor[:,3:6].numpy()),axis=1)
        raw=np.fromfile(info['lidar_path'],np.float32).reshape(-1,int(info.get('num_pts_feats',6)))
        if raw.shape[1]<6:raise ValueError('Expected XYZRGB point file')
        axis=np.asarray(info['axis_align_matrix'],np.float32)
        points=raw[:,:3]@axis[:3,:3].T+axis[:3,3];rgb=np.clip(np.rint(raw[:,3:6]),0,255).astype(np.uint8)
        boxes,scores,labels=read_predictions(source)
        selection=select_same_class_max_iou(gt_boxes,ann['gt_labels_3d'],boxes,scores,labels,args.min_iou)
        summary=save_scene(args.output_dir,scene,source,points,rgb,gt_boxes,ann['gt_labels_3d'],boxes,scores,labels,selection,
            min_iou=args.min_iou,axis_alignment=axis,class_names=dataset.metainfo['classes'],max_points=args.max_points,
            radius=args.line_radius,preview=not args.no_preview,elev=args.preview_elev,azim=args.preview_azim,labels_text=args.label_text)
        summaries.append(summary)
        (args.output_dir/'index.json').write_text(json.dumps(dict(gt_guided_selection=True,min_iou=args.min_iou,
             input_dir=str(args.input_dir.resolve()),config=str(Path(args.config).resolve()),completed=summaries),indent=2))
        print(f'{scene}: {summary["matched_gt_count"]}/{summary["gt_count"]} GT matched; '
              f'{summary["selected_prediction_count"]}/{summary["raw_prediction_count"]} unique predictions saved',flush=True)


if __name__=='__main__':
    main()
