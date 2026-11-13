import importlib.util
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData
import pytest

spec=importlib.util.spec_from_file_location('filtered_vis',Path(__file__).with_name('filter_predictions_with_gt.py'))
vis=importlib.util.module_from_spec(spec);spec.loader.exec_module(vis)


def test_filter_threshold_is_inclusive_and_independent_of_gt():
    boxes=np.ones((5,6),np.float32);boxes[2,3]=0;boxes[3,0]=np.nan
    scores=np.array([.5,.49,.9,.9,np.nan]);labels=np.array([0,1,2,3,4])
    np.testing.assert_array_equal(vis.filter_predictions(boxes,scores,labels,.5),[True,False,False,False,False])
    for bad in (-1,2,np.nan):
        with pytest.raises(ValueError):vis.filter_predictions(boxes,scores,labels,bad)


def test_save_filtered_npz_keeps_raw_and_ply_has_both_box_sets(tmp_path):
    xyz=np.array([[1,2,3],[3,4,5]],np.float32);rgb=np.array([[255,0,0],[0,0,255]],np.uint8)
    gt=np.array([[0,0,1,2,2,2]],np.float32);boxes=np.array([[0,0,1,1,1,1],[4,5,6,2,2,2]],np.float32)
    scores=np.array([.8,.1]);labels=np.array([0,1]);saved=boxes.copy()
    result=vis.save_filtered_scene(tmp_path,'scene0000_00',tmp_path/'source.npz',xyz,rgb,gt,np.array([0]),
             boxes,scores,labels,threshold=.5,preview=True)
    np.testing.assert_array_equal(boxes,saved)
    data=np.load(tmp_path/'scene0000_00_filtered_predictions.npz')
    np.testing.assert_array_equal(data['boxes_center_size_world'],boxes[:1])
    np.testing.assert_array_equal(data['source_indices'],[0])
    np.testing.assert_array_equal(data['raw_boxes_center_size_world'],boxes)
    np.testing.assert_array_equal(data['gt_boxes_center_size_world'],gt)
    np.testing.assert_array_equal(data['gt_rgb'],rgb)
    ply=PlyData.read(tmp_path/'scene0000_00_gt_predictions.ply');v=ply['vertex'].data
    np.testing.assert_array_equal(np.column_stack([v[k] for k in ('red','green','blue')])[:2],rgb)
    assert len(v)==2+2*192 and len(ply['face'])==2*192
    assert result['filtered_prediction_count']==1
    metadata=json.loads((tmp_path/'scene0000_00_filtered_predictions.json').read_text())
    assert len(metadata['predictions'])==2 and len(metadata['filtered_predictions'])==1
    assert (tmp_path/'scene0000_00_gt_predictions.png').stat().st_size>1000


def test_zero_filtered_predictions_still_draws_gt(tmp_path):
    boxes=np.ones((1,6),np.float32)
    vis.save_filtered_scene(tmp_path,'scene0000_00','source',np.zeros((1,3)),np.zeros((1,3)),
                           boxes,np.array([0]),boxes,np.array([.1]),np.array([0]),threshold=.9,preview=True)
    data=np.load(tmp_path/'scene0000_00_filtered_predictions.npz')
    assert data['boxes_center_size_world'].shape==(0,6)
    assert len(PlyData.read(tmp_path/'scene0000_00_gt_predictions.ply')['vertex'])==1+192


def test_read_both_previous_formats_and_discover(tmp_path):
    boxes=np.ones((2,6),np.float32);scores=np.array([.2,.8]);labels=np.array([0,1])
    new=tmp_path/'predictions';new.mkdir();file=new/'scene0015_00_predictions.npz'
    np.savez(file,boxes_center_size_world=boxes,scores=scores,labels=labels)
    assert vis.discover_predictions(tmp_path)=={'scene0015_00':file}
    for actual,expected in zip(vis.read_predictions(file),(boxes,scores,labels)):
        np.testing.assert_array_equal(actual,expected)
    legacy=tmp_path/'legacy/scene0015_00';legacy.mkdir(parents=True)
    np.savez(legacy/'predictions.npz',detection_boxes=boxes,detection_scores=scores,detection_labels=labels)
    assert list(vis.discover_predictions(tmp_path/'legacy'))==['scene0015_00']
    np.testing.assert_array_equal(vis.read_predictions(legacy/'predictions.npz')[0],boxes)
