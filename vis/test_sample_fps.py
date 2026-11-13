"""FPS correctness and PLY overlay regressions."""
import json

import numpy as np
from plyfile import PlyData
import pytest
import torch

from vis.reconstruct_vggt_omega import save_ply
from vis.sample_fps import (farthest_point_sampling, load_cloud, make_overlay,
                            parse_args, sample_scene)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def reference_fps(points, count, start):
    points = points.astype(np.float64) - points[start]
    if np.abs(points).max() > 0:
        points /= np.abs(points).max()
    selected = []
    nearest = np.full(len(points), np.inf)
    current = start
    for _ in range(min(count, len(points))):
        selected.append(current)
        nearest = np.minimum(nearest, ((points-points[current])**2).sum(1))
        candidate = nearest.copy()
        candidate[selected] = -1
        current = candidate.argmax()
    return np.array(selected)


@pytest.mark.parametrize('device', DEVICES)
def test_greedy_fps_matches_independent_reference(device):
    points = np.random.default_rng(10).normal(size=(200, 3)).astype(np.float32)
    actual = farthest_point_sampling(points, 32, start_index=7, device=device)
    expected = reference_fps(points, 32, 7)
    assert np.array_equal(actual, expected)
    assert len(np.unique(actual)) == 32
    assert np.array_equal(farthest_point_sampling(points, 32, 7, device), actual)
    # Translation and positive uniform scaling leave geometric choices unchanged.
    assert np.array_equal(farthest_point_sampling(points * 5 + 2, 32, 7, device), actual)


@pytest.mark.parametrize('device', DEVICES)
def test_duplicates_and_count_larger_than_cloud(device):
    points = np.ones((5, 3), dtype=np.float32)
    actual = farthest_point_sampling(points, 10, 2, device)
    assert len(actual) == 5 and len(np.unique(actual)) == 5 and actual[0] == 2


def test_overlay_preserves_context_and_exact_sample_coordinates():
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.full((10, 3), 110, dtype=np.uint8)
    indices = np.array([2, 7])
    color = np.array([255, 0, 0], dtype=np.uint8)
    xyz, rgb, sources, types, sample_ids = make_overlay(points, colors, indices, color, .1, 64)
    np.testing.assert_array_equal(xyz[:10], points)
    np.testing.assert_array_equal(rgb[:10], colors)
    np.testing.assert_array_equal(xyz[types == 1], points[indices])
    np.testing.assert_array_equal(sources[types == 1], indices)
    np.testing.assert_array_equal(sample_ids[types == 1], [0, 1])
    assert len(xyz) == 10 + 2 + 2 * 64
    distances = np.linalg.norm(xyz[types == 2].reshape(2, 64, 3) - points[indices, None], axis=-1)
    np.testing.assert_allclose(distances, .1, atol=2e-6)


def test_scene_exports_new_ply_and_original_is_unchanged(tmp_path):
    points = np.random.default_rng(4).normal(size=(100, 3)).astype(np.float32)
    colors = np.random.default_rng(5).integers(0, 256, size=(100, 3), dtype=np.uint8)
    source = tmp_path / 'source/reconstruction.ply'
    source.parent.mkdir()
    save_ply(source, points, colors)
    source_bytes = source.read_bytes()
    args = parse_args(['--num-points', '8', '--marker-points', '32', '--no-preview'])
    destination = tmp_path / 'output'
    record = sample_scene(source, destination, args, 'cpu')
    assert record['sampled_points'] == 8
    assert source.read_bytes() == source_bytes
    selected = np.load(destination / 'fps_indices.npy')
    exported = PlyData.read(destination / 'scene_with_fps.ply')['vertex'].data
    np.testing.assert_array_equal(exported['source_index'][:100], np.arange(100))
    samples = exported[exported['point_type'] == 1]
    np.testing.assert_array_equal(np.column_stack([samples[k] for k in ['x', 'y', 'z']]), points[selected])
    np.testing.assert_array_equal(samples['source_index'], selected)
    assert len(exported) == 100 + 8 + 8 * 32
    assert len(load_cloud(destination / 'fps_points.ply')[0]) == 8
    assert json.loads((destination / 'manifest.json').read_text())['scene_points'] == 100
    with pytest.raises(FileExistsError):
        sample_scene(source, destination, args, 'cpu')


def test_invalid_cloud_and_options():
    with pytest.raises(ValueError, match='finite'):
        farthest_point_sampling(np.full((3, 3), np.nan), 2)
    with pytest.raises(ValueError, match='start index'):
        farthest_point_sampling(np.ones((3, 3)), 2, 10)
    for options in [['--num-points', '0'], ['--marker-color', 'red'], ['--marker-radius', '-1']]:
        with pytest.raises(SystemExit):
            parse_args(options)
