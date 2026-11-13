"""Fast contracts for the visualization script; no large model checkpoint."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from vis.reconstruct_vggt_omega import (
    discover_scenes, filter_cloud, image_files, parse_args, preprocess,
    reconstruct_scene, save_ply, uniform_indices)


def write_image(path, size=(32, 24), color=(120, 180, 220)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)


def test_discovery_natural_order_and_no_depth_input(tmp_path):
    for name in ['10.jpg', '2.jpg', '0.jpg']:
        write_image(tmp_path / 'scene_a' / name)
    write_image(tmp_path / 'scene_a/depth/0.png')
    (tmp_path / 'scene_a/0.txt').write_text('not read')
    write_image(tmp_path / 'scene_b/color/0.jpg')
    scenes = discover_scenes(tmp_path)
    assert [name for name, _ in scenes] == ['scene_a', 'scene_b']
    assert [p.name for p in image_files(scenes[0][1])] == ['0.jpg', '2.jpg', '10.jpg']
    assert discover_scenes(tmp_path, ['scene_b'])[0][1] == tmp_path / 'scene_b/color'
    assert discover_scenes(tmp_path / 'scene_a')[0][0] == 'scene_a'


@pytest.mark.parametrize('total,count,expected', [(11, 4, [0, 3, 7, 10]),
    (3, 10, [0, 1, 2]), (3, -1, [0, 1, 2]), (3, 1, [0])])
def test_uniform_sampling(total, count, expected):
    assert uniform_indices(total, count).tolist() == expected


def test_padding_and_color_contract(tmp_path):
    paths = [tmp_path / 'a.png', tmp_path / 'b.png']
    write_image(paths[0], (64, 32), (255, 0, 0))
    write_image(paths[1], (32, 64), (0, 255, 0))
    images, valid, meta = preprocess(paths, 'max_size', 64)
    assert images.shape == (2, 3, 64, 64)
    assert valid.sum(axis=(1, 2)).tolist() == [2048, 2048]
    assert meta[0]['padding_top'] == 16 and meta[1]['padding_left'] == 16
    assert images.dtype == torch.float32 and 0 <= images.min() <= images.max() <= 1
    assert images[0, :, 32, 32].tolist() == [1, 0, 0]


def test_cloud_filter_and_ply(tmp_path):
    points = np.arange(24, dtype=np.float32).reshape(1, 2, 4, 3)
    colors = np.full((1, 2, 4, 3), 123, dtype=np.uint8)
    depth = np.ones((1, 2, 4), dtype=np.float32)
    conf = np.arange(8).reshape(1, 2, 4)
    valid = np.ones_like(depth, dtype=bool)
    valid[0, 0, 0] = False
    depth[0, 0, 1] = 0
    points[0, 0, 2] = np.nan
    xyz, rgb, stats = filter_cloud(points, colors, depth, conf, valid,
                                   stride=1, confidence_percentile=0, max_points=3)
    assert len(xyz) == 3 and stats['point_count_before_cap'] == 5
    path = tmp_path / 'cloud.ply'
    save_ply(path, xyz, rgb)
    header, data = path.read_bytes().split(b'end_header\n')
    assert b'element vertex 3' in header and len(data) == 3 * 15


class FakeModel:
    def __call__(self, images):
        views, _, height, width = images.shape
        pose = torch.zeros(1, views, 9)
        pose[..., 6] = 1  # identity quaternion, xyzw
        pose[..., 7:] = 1
        return dict(pose_enc=pose, depth=torch.ones(1, views, height, width, 1),
                    depth_conf=torch.ones(1, views, height, width, 1))


def test_reconstruction_exports_exact_view_count_and_overwrite(tmp_path):
    scene = tmp_path / 'input'
    for i in range(7):
        write_image(scene / f'{i}.jpg')
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fake for manifest metadata only')
    args = parse_args(['--checkpoint', str(checkpoint), '--num-images', '4',
                       '--image-resolution', '32', '--pixel-stride', '1', '--no-preview'])
    out = tmp_path / 'output'
    manifest = reconstruct_scene(FakeModel(), 'scene', scene, out, torch.device('cpu'), args)
    assert [item['source_index'] for item in manifest['selected']] == [0, 2, 4, 6]
    assert len(list((out / 'scene/images').glob('*.png'))) == 4
    cameras = np.load(out / 'scene/cameras.npz')
    assert cameras['extrinsics_w2c'].shape == (4, 3, 4)
    np.testing.assert_allclose(cameras['extrinsics_w2c'][:, :3, :3],
                               np.tile(np.eye(3)[None], (4, 1, 1)), atol=1e-6)
    assert (out / 'scene/reconstruction.ply').is_file()
    assert not (out / 'scene/depth_confidence.npz').exists()
    with pytest.raises(FileExistsError):
        reconstruct_scene(FakeModel(), 'scene', scene, out, torch.device('cpu'), args)
    args.overwrite, args.num_images = True, 2
    reconstruct_scene(FakeModel(), 'scene', scene, out, torch.device('cpu'), args)
    assert len(list((out / 'scene/images').glob('*.png'))) == 2
    stored = json.loads((out / 'scene/manifest.json').read_text())
    assert stored['selected_images'] == 2


@pytest.mark.parametrize('argument', [['--num-images', '0'], ['--num-images', '-2'],
    ['--image-resolution', '33'], ['--pixel-stride', '0'], ['--confidence-percentile', '101']])
def test_invalid_arguments(argument):
    with pytest.raises(SystemExit):
        parse_args(argument)
