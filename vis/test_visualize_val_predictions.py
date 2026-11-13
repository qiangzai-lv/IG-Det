import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData

spec = importlib.util.spec_from_file_location('val_vis', Path(__file__).with_name('visualize_val_predictions.py'))
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


class FakeModel(torch.nn.Module):
    def _select_reconstruction_boxes(self, *args):
        xyz = torch.tensor([[[1.,2.,3.],[2.,3.,4.]]])
        return dict(query_xyz_vggt=xyz,query_size_vggt=torch.ones_like(xyz),
                    scores=torch.tensor([[.9,.8]]),labels=torch.tensor([[1,2]]),
                    source_indices=torch.tensor([[3,4]]),fallback_mask=torch.tensor([[False,True]]))

    def get_box_features(self, maps, inputs, images, xyz, query):
        return xyz + 1

    def forward(self, inputs):
        s = self._select_reconstruction_boxes()
        result = self.get_box_features(None, inputs, None,s['query_xyz_vggt'],None)
        s['query_xyz_vggt'].add_(999)  # Captured values must survive mutation.
        return result


def test_capture_preserves_forward_and_restores_methods():
    model = FakeModel().eval()
    # 90 degree rotation + translation; sizes are scene-axis sizes and must
    # only be scaled, NOT rotated or re-AABBed (same contract as detection head).
    pose = torch.tensor([[[0.,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]]])
    inputs = dict(pose_matrix=pose,axis_align_matrix=torch.eye(4)[None],vggt_gt_scale=torch.tensor([2.]))
    baseline = model(inputs)
    with vis.QueryCapture(model) as capture:
        actual = model(inputs)
        np.testing.assert_array_equal(actual,baseline)
        np.testing.assert_allclose(capture.payload['query_boxes'][0,0], [6,22,36,2,2,2])
        assert capture.payload['query_xyz_vggt'][0,0,0] == 1
    assert 'get_box_features' not in model.__dict__
    torch.testing.assert_close(model(inputs),baseline)
    with pytest.raises(RuntimeError):
        with vis.QueryCapture(model):
            raise RuntimeError('failure')
    assert '_select_reconstruction_boxes' not in model.__dict__


def test_ply_cylinders_preserve_cloud_and_geometry(tmp_path):
    xyz = np.array([[1,2,3],[4,5,6]],np.float32)
    rgb = np.array([[1,2,3],[255,0,0]],np.uint8)
    boxes = np.array([[0,0,0,2,4,6]],np.float32)
    path = tmp_path/'overlay.ply'
    vis.write_overlay(path,xyz,rgb,boxes,np.array([[255,140,25]]),.01)
    ply = PlyData.read(path)
    vertices = ply['vertex'].data
    actual = np.column_stack([vertices[k] for k in ('x','y','z')])
    np.testing.assert_array_equal(actual[:2],xyz)
    assert len(vertices) == 2+12*16
    assert len(ply['face']) == 12*16
    faces = np.stack(ply['face'].data['vertex_indices'])
    assert faces.min() >= 2 and faces.max() < len(vertices)
    # Each edge cylinder is centered on its exact box segment.
    for edge, segment in enumerate(vis.box_segments(boxes)):
        rings = actual[2+16*edge:2+16*(edge+1)].reshape(2,8,3)
        np.testing.assert_allclose(rings.mean(1),segment,atol=1e-6)


def test_export_filters_rendering_but_keeps_raw_predictions(tmp_path):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(50,3)).astype(np.float32)
    boxes = np.array([[0,0,0,1,1,1],[1,1,1,2,2,2],[0,0,0,1,1,1]],np.float32)
    query = dict(query_boxes=boxes, source_indices=np.array([1,2,-1]),
                 scores=np.array([.7,.9,0]),fallback_mask=np.array([False,True,True]))
    detections = dict(boxes=boxes, scores=np.array([.9,.1,np.nan]), labels=np.array([0,1,2]))
    manifest = vis.export_scene(tmp_path,points,np.full((50,3),160,np.uint8),query,detections,
                                dict(scene_id='test'),query_topk=1,preview=True)
    assert manifest['rendered_queries'] == 1
    assert manifest['rendered_detections'] == 1
    data = np.load(tmp_path/'predictions.npz')
    np.testing.assert_array_equal(data['rendered_query_ids'],[1])  # Real FPS fallback retained.
    np.testing.assert_array_equal(data['rendered_detection_ids'],[0])
    assert len(data['detection_boxes']) == 3
    assert (tmp_path/'comparison.png').stat().st_size > 1000
    assert (tmp_path/'comparison.pdf').stat().st_size > 1000


def test_empty_detections_and_queries(tmp_path):
    empty = np.empty((0,6),np.float32)
    manifest = vis.export_scene(tmp_path,np.zeros((2,3)),np.ones((2,3),np.uint8),
                     dict(query_boxes=empty,source_indices=np.empty(0,int),scores=np.empty(0)),
                     dict(boxes=empty,scores=np.empty(0),labels=np.empty(0,int)),{},preview=True)
    assert manifest['rendered_queries'] == manifest['rendered_detections'] == 0
    assert len(PlyData.read(tmp_path/'detection_boxes.ply')['vertex']) == 2


def test_gravity_center_of_depth_boxes():
    from mmdet3d.structures import DepthInstance3DBoxes
    box = DepthInstance3DBoxes(torch.tensor([[0.,0,0,2,2,4]]),box_dim=6,with_yaw=False)
    # Its tensor center is bottom based; gravity center is required for drawing.
    np.testing.assert_allclose(box.gravity_center,[[0,0,2]])


@pytest.mark.skipif(__import__('os').environ.get('IGDET_REAL_DATA_TEST') != '1',
                    reason='Set IGDET_REAL_DATA_TEST=1 to read all local validation scenes')
def test_all_real_validation_inputs_and_scan_exports(tmp_path):
    """Real input/output smoke only: empty detections, no trained-model claim."""
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.utils import register_all_modules
    from igdet.camera_alignment import load_axis_aligned_points
    cfg = Config.fromfile(str(vis.ROOT/'configs/igdet/igdet_scannet.py'))
    register_all_modules()
    import_modules_from_strings(**cfg.custom_imports)
    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.update(num_workers=0,persistent_workers=False)
    loader = Runner.build_dataloader(loader_cfg,seed=0)
    scenes = []
    for batch in loader:
        meta = batch['data_samples'][0].metainfo
        scene = meta['scene_id']
        scenes.append(scene)
        assert len(meta['img_path']) == 128
        assert len(meta['view_indices']) == 128
        raw = np.fromfile(meta['lidar_path'],np.float32).reshape(-1,meta['num_pts_feats'])
        aligned = load_axis_aligned_points(meta['lidar_path'],meta['num_pts_feats'],meta['axis_align_matrix'])
        np.testing.assert_allclose(aligned,raw[:,:3]@np.asarray(meta['axis_align_matrix'])[:3,:3].T+
                                   np.asarray(meta['axis_align_matrix'])[:3,3],atol=1e-5)
        empty = np.empty((0,6),np.float32)
        directory = tmp_path/scene
        manifest = vis.export_scene(directory,aligned,raw[:,3:6],
                     dict(query_boxes=empty,source_indices=np.empty(0,int),scores=np.empty(0)),
                     dict(boxes=empty,scores=np.empty(0),labels=np.empty(0,int)),
                     dict(scene_id=scene,test_only=True),preview=(len(scenes)==1),max_points=2000)
        assert manifest['cloud_points'] > 0
        ply = PlyData.read(directory/'scene.ply')['vertex'].data
        assert len(ply) == manifest['cloud_points']
    assert len(set(scenes)) == len(scenes) == len(loader.dataset)
    print(f'\nValidated all {len(scenes)} real validation scenes (128 RGB views each); scan-only export smoke.')


def test_exact_prediction_comparison_detects_changes():
    from types import SimpleNamespace as NS
    import copy
    prediction = NS(pred_instances_3d=NS(
        bboxes_3d=NS(tensor=torch.tensor([[1.,2.,3.,4.,5.,6.]])),
        scores_3d=torch.tensor([.8]),labels_3d=torch.tensor([2])))
    assert vis.compare_predictions([prediction],[copy.deepcopy(prediction)])['exact_match']
    changed = copy.deepcopy(prediction)
    changed.pred_instances_3d.scores_3d += .01
    with pytest.raises(AssertionError,match='scores_3d'):
        vis.compare_predictions([prediction],[changed])
    assert vis.compare_predictions([],[])['exact_match']


def test_reference_layout_threshold_only_affects_visuals(tmp_path):
    import json
    boxes = np.array([[0,0,0,1,2,3],[1,2,3,2,3,4],[0,0,0,1,1,1]],np.float32)
    scores = np.array([.9,.1,np.nan])
    query = dict(query_boxes=boxes,scores=scores,labels=np.array([0,1,2]),
                 source_indices=np.array([5,6,-1]),fallback_mask=np.array([False,True,True]))
    det = dict(boxes=boxes,scores=scores,labels=np.array([0,1,2]))
    points = np.array([[0,0,0],[1,2,3]],np.float32)
    summary = vis.export_reference_scene(tmp_path,points,np.full((2,3),100,np.uint8),query,det,
                                        dict(scene_id='scene_test'),score_thr=.5,preview=True)
    assert summary['rendered_queries'] == summary['rendered_detections'] == 1
    for sub,stem,count_key in [('queries','initial_queries','queries'),('predictions','predictions','predictions')]:
        prefix = tmp_path/sub/f'scene_test_{stem}'
        data = np.load(str(prefix)+'.npz')
        np.testing.assert_array_equal(data['boxes_center_size_world'],boxes)
        np.testing.assert_array_equal(data['scores'],scores)
        np.testing.assert_array_equal(data['displayed_mask'],[True,False,False])
        metadata = json.loads(Path(str(prefix)+'.json').read_text())
        assert len(metadata[count_key]) == 3
        assert metadata[count_key][2]['score'] is None  # Invalid score retained as JSON null.
        assert metadata['visualization_threshold_applied_to_saved_predictions'] is False
        assert Path(str(prefix)+'.png').stat().st_size > 1000
    ply = PlyData.read(tmp_path/'predictions/scene_test_predictions_boxes.ply')
    assert len(ply['vertex']) == 12*16
    combined = PlyData.read(tmp_path/'predictions/scene_test_gt_predictions.ply')
    assert len(combined['vertex']) == 2+12*16
    from mmdet3d.structures import DepthInstance3DBoxes
    expected = DepthInstance3DBoxes(torch.from_numpy(boxes),box_dim=6,with_yaw=False,origin=(.5,.5,.5)).corners.numpy()
    np.testing.assert_allclose(data['box_corners_world'],expected,atol=1e-6)


def test_reference_layout_zero_shown_keeps_all_raw_rows(tmp_path):
    import json
    boxes = np.ones((2,6),np.float32)
    query = dict(query_boxes=boxes,scores=np.array([.1,.2]),labels=np.array([0,1]),source_indices=np.array([0,1]))
    det = dict(boxes=boxes,scores=np.array([.1,.2]),labels=np.array([0,1]))
    vis.export_reference_scene(tmp_path,np.zeros((3,3)),np.zeros((3,3)),query,det,
                               dict(scene_id='zero'),score_thr=.9,preview=True)
    assert len(PlyData.read(tmp_path/'predictions/zero_predictions_boxes.ply')['vertex']) == 0
    for sub,stem,key in [('queries','initial_queries','queries'),('predictions','predictions','predictions')]:
        metadata = json.loads((tmp_path/sub/f'zero_{stem}.json').read_text())
        assert len(metadata[key]) == 2
        data = np.load(tmp_path/sub/f'zero_{stem}.npz')
        assert len(data['scores']) == 2 and not data['displayed_mask'].any()


def test_saved_reference_results_can_be_refiltered_without_inference(tmp_path):
    import json
    from types import SimpleNamespace
    source, target = tmp_path/'source',tmp_path/'target'
    boxes = np.ones((2,6),np.float32)
    query = dict(query_boxes=boxes,scores=np.array([.1,.8]),source_indices=np.array([0,1]))
    det = dict(boxes=boxes,scores=np.array([.1,.8]),labels=np.array([0,1]))
    vis.export_reference_scene(source,np.zeros((2,3)),np.zeros((2,3)),query,det,
                               dict(scene_id='test'),score_thr=.05,preview=False)
    (source/'index.json').write_text(json.dumps(dict(completed=['test'])))
    vis.render_saved_results(SimpleNamespace(from_results=source,output_dir=target,overwrite=False,
         scenes=None,max_scenes=None,score_thr=.9,query_score_thr=None,query_topk=None,
         max_points=200000,line_radius=None,no_preview=True,preview_elev=55,preview_azim=-65))
    original = np.load(source/'queries/test_initial_queries.npz')
    after = np.load(target/'queries/test_initial_queries.npz')
    np.testing.assert_array_equal(after['scores'],original['scores'])
    assert not after['displayed_mask'].any()
    assert len(json.loads((target/'queries/test_initial_queries.json').read_text())['queries']) == 2


def test_legacy_results_rerender_does_not_reuse_old_display_counts(tmp_path):
    import json
    from types import SimpleNamespace
    source,target = tmp_path/'legacy',tmp_path/'new'
    boxes = np.ones((2,6),np.float32)
    query = dict(query_boxes=boxes,scores=np.array([.1,.8]),source_indices=np.array([0,1]))
    det = dict(boxes=boxes,scores=np.array([.1,.8]),labels=np.array([0,1]))
    vis.export_scene(source/'test',np.zeros((2,3)),np.zeros((2,3)),query,det,
                     dict(scene_id='test'),score_thr=.05,preview=False)
    (source/'index.json').write_text(json.dumps(dict(completed=['test'])))
    vis.render_saved_results(SimpleNamespace(from_results=source,output_dir=target,overwrite=False,
         scenes=None,max_scenes=None,score_thr=.9,query_score_thr=None,query_topk=None,
         max_points=200000,line_radius=None,no_preview=True,preview_elev=55,preview_azim=-65))
    metadata = json.loads((target/'predictions/test_predictions.json').read_text())
    assert metadata['displayed_boxes'] == 0
    assert 'rendered_detections' not in metadata
    assert len(metadata['predictions']) == 2


def test_device_auto_prefers_npu_and_selects_index_without_cuda(monkeypatch):
    from types import SimpleNamespace
    chosen = []
    monkeypatch.setattr(torch,'npu',SimpleNamespace(is_available=lambda:True,
        current_device=lambda:1,device_count=lambda:2,set_device=chosen.append),raising=False)
    def no_cuda(*args,**kwargs):
        raise AssertionError('NPU selection must not call CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',no_cuda)
    assert vis.resolve_execution_device('auto') == 'npu:1'
    assert vis.resolve_execution_device('npu:0') == 'npu:0'
    assert chosen == [1,0]
    with pytest.raises(ValueError,match='available device count'):
        vis.resolve_execution_device('npu:2')


def test_device_cuda_and_cpu_fallback_and_invalid_options(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch,'npu',SimpleNamespace(is_available=lambda:False),raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    chosen=[]
    monkeypatch.setattr(torch.cuda,'set_device',chosen.append)
    assert vis.resolve_execution_device('auto') == 'cuda:0'
    assert vis.resolve_execution_device('cuda:1') == 'cuda:1'
    assert chosen == [0,1]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert vis.resolve_execution_device('auto') == 'cpu'
    assert vis.resolve_execution_device('cpu') == 'cpu'
    with pytest.raises(RuntimeError,match='unavailable'):
        vis.resolve_execution_device('cuda:0')
    for value in ('cuda:-1','npu:x','auto:0','cpu:0','bad'):
        with pytest.raises(ValueError):
            vis.resolve_execution_device(value)


def test_missing_npu_extension_has_clear_error(monkeypatch):
    monkeypatch.delattr(torch,'npu',raising=False)
    def missing(name):
        assert name == 'torch_npu'
        raise ModuleNotFoundError('No module named torch_npu')
    monkeypatch.setattr(vis.importlib,'import_module',missing)
    with pytest.raises(RuntimeError,match='torch_npu'):
        vis.resolve_execution_device('npu:0')


def test_npu_rng_restore_on_exception_never_calls_cuda(monkeypatch):
    import random
    from types import SimpleNamespace
    state = dict(value=torch.tensor([1,2],dtype=torch.uint8),indices=[])
    def get(index):
        state['indices'].append(('get',index))
        return state['value']
    def set_state(value,index):
        state['indices'].append(('set',index))
        state['value']=value.clone()
    monkeypatch.setattr(torch,'npu',SimpleNamespace(get_rng_state=get,set_rng_state=set_state),raising=False)
    def forbidden(*args,**kwargs):
        raise AssertionError('CUDA RNG must not be used on NPU')
    monkeypatch.setattr(torch.cuda,'get_rng_state',forbidden)
    monkeypatch.setattr(torch.cuda,'set_rng_state',forbidden)
    cpu, py, npstate = torch.get_rng_state(),random.getstate(),np.random.get_state()
    with pytest.raises(RuntimeError,match='inside'):
        with vis.preserve_random_state('npu:1'):
            torch.rand(3); random.random(); np.random.rand(3)
            state['value'].fill_(9)
            raise RuntimeError('inside')
    assert torch.equal(torch.get_rng_state(),cpu)
    assert random.getstate() == py
    np.testing.assert_array_equal(np.random.get_state()[1],npstate[1])
    assert torch.equal(state['value'],torch.tensor([1,2],dtype=torch.uint8))
    assert state['indices']==[('get',1),('set',1)]


def test_cpu_rng_restore_does_not_touch_accelerators(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('CPU RNG preservation must not access CUDA')
    monkeypatch.setattr(torch.cuda,'get_rng_state',forbidden)
    state=torch.get_rng_state()
    with vis.preserve_random_state('cpu'):
        torch.rand(5)
    assert torch.equal(torch.get_rng_state(),state)


def test_npu_verification_reports_tolerance_without_claiming_exactness():
    import copy
    from types import SimpleNamespace as NS
    first=NS(pred_instances_3d=NS(bboxes_3d=NS(tensor=torch.ones(1,6)),
             scores_3d=torch.tensor([.012]),labels_3d=torch.tensor([2])))
    second=copy.deepcopy(first)
    second.pred_instances_3d.bboxes_3d.tensor[0,0] += .0006844997
    second.pred_instances_3d.scores_3d += .000001
    with pytest.raises(AssertionError,match='bboxes_3d'):
        vis.compare_predictions([first],[second])
    report=vis.compare_predictions([first],[second],**vis.verification_tolerances('npu:0'))
    assert report['within_tolerance'] and not report['exact_match']
    assert report['labels_exact']
    assert .00068 < report['max_box_absolute_difference'] < .00069
    assert report['max_score_absolute_difference'] < 1e-5
    assert vis.verification_tolerances('cuda:0')['box_atol'] == 0
    assert vis.verification_tolerances('npu:0',0,0,0)['box_atol'] == 0


def test_verification_rejects_large_errors_labels_counts_and_nonfinite():
    import copy
    from types import SimpleNamespace as NS
    first=NS(pred_instances_3d=NS(bboxes_3d=NS(tensor=torch.ones(1,6)),
             scores_3d=torch.tensor([.012]),labels_3d=torch.tensor([2])))
    tol=vis.verification_tolerances('npu:0')
    for key,value in [('box',.01),('score',.0001),('label',3),('nan',float('nan'))]:
        second=copy.deepcopy(first)
        if key=='box': second.pred_instances_3d.bboxes_3d.tensor[0,0] += value
        elif key=='score': second.pred_instances_3d.scores_3d += value
        elif key=='label': second.pred_instances_3d.labels_3d[0] = value
        else: second.pred_instances_3d.bboxes_3d.tensor[0,0] = value
        with pytest.raises(AssertionError):
            vis.compare_predictions([first],[second],**tol)
    second=copy.deepcopy(first)
    second.pred_instances_3d.bboxes_3d.tensor=torch.ones(2,6)
    with pytest.raises(AssertionError,match='shape changed'):
        vis.compare_predictions([first],[second],**tol)
    for invalid in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            vis.compare_predictions([],[],box_atol=invalid)
