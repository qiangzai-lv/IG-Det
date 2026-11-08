"""Add scene and stable per-scene instance ids to ARKitScenes 3D annotations."""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import pickle
import numpy as np
import tempfile
from pathlib import Path


CLASSES = (
    'cabinet', 'refrigerator', 'shelf', 'stove', 'bed', 'sink', 'washer',
    'toilet', 'bathtub', 'oven', 'dishwasher', 'fireplace', 'stool',
    'chair', 'table', 'tv_monitor', 'sofa')


def validate_categories(payload: dict) -> None:
    expected = {name: index for index, name in enumerate(CLASSES)}
    if payload.get('metainfo', {}).get('categories') != expected:
        raise ValueError('Expected the ARKitScenes 17-class category mapping')


def scene_id_from_info(info: dict) -> str:
    paths = info.get('img_paths', [])
    if not paths:
        raise ValueError('annotation record has no img_paths')
    scene = Path(paths[0]).parent.name
    if not scene.isdigit():
        raise ValueError(f'invalid ARKit scene ID: {scene!r}')
    if 'scene_id' in info and str(info['scene_id']) != scene:
        raise ValueError('scene_id disagrees with image paths')
    return scene


def validate_scene(info: dict) -> None:
    """Validate this processed dataset contract rather than guessing transforms."""
    scene = scene_id_from_info(info)
    count = len(info['img_paths'])
    for key in ('depth_paths', 'cam2img', 'lidar2cam'):
        if len(info[key]) != count:
            raise ValueError(f'{scene}: inconsistent per-view array lengths: {key}')
    axis = np.asarray(info['axis_align_matrix'])
    if axis.shape != (4, 4) or not np.allclose(axis, np.eye(4), atol=1e-6, rtol=0):
        raise ValueError(f'{scene}: non-identity axis alignment requires checking data provenance')
    points = info['lidar_points']
    if points['lidar_path'] != f'{scene}_points.bin' or int(points['num_pts_feats']) != 6:
        raise ValueError(f'{scene}: expected scene_points.bin with six features')
    seen = set()
    path_parts = Path(info['img_paths'][0]).parts
    split = path_parts[2] if len(path_parts) > 2 else ''
    for rgb, depth in zip(info['img_paths'], info['depth_paths']):
        rgb, depth = Path(rgb), Path(depth)
        if (rgb.is_absolute() or depth.is_absolute() or '..' in rgb.parts or '..' in depth.parts
                or len(rgb.parts) != 5 or rgb.parts[:2] != ('processed', '3dod')
                or rgb.parts[2] not in ('train', 'val') or rgb.parts[2] != split or rgb.parent.name != scene
                or not rgb.name.startswith(scene + '_') or not rgb.name.endswith('_color.png')
                or str(depth) != str(rgb).replace('_color.png', '_depth.png')):
            raise ValueError(f'{scene}: inconsistent RGB/depth paths: {rgb}, {depth}')
        if str(rgb) in seen:
            raise ValueError(f'{scene}: duplicate image path: {rgb}')
        seen.add(str(rgb))
    # Batch camera checks avoid thousands of small NumPy calls per scene.
    k = np.asarray(info['cam2img'])
    t = np.asarray(info['lidar2cam'])
    if (k.shape != (count, 3, 3) or not np.isfinite(k).all()
            or (k[:, 0, 0] <= 0).any() or (k[:, 1, 1] <= 0).any()
            or not np.allclose(k[:, 2], [0, 0, 1], atol=1e-6)):
        raise ValueError(f'{scene}: invalid camera intrinsic')
    if (t.shape != (count, 4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[:, 3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:, :3, :3].transpose(0, 2, 1) @ t[:, :3, :3],
                               np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(t[:, :3, :3]), 1, atol=1e-4).all()):
        raise ValueError(f'{scene}: invalid rigid C2W pose')
    ids = []
    for index, instance in enumerate(info['instances']):
        box = np.asarray(instance['bbox_3d'])
        if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any():
            raise ValueError(f'{scene}: expected finite positive-dimension 7D OBB')
        label = instance['bbox_label_3d']
        if not isinstance(label, (int, np.integer)) or not 0 <= label < len(CLASSES):
            raise ValueError(f'{scene}: invalid class label: {label}')
        identifier = instance.get('instance_id', index)
        if not isinstance(identifier, (int, np.integer)) or identifier < 0:
            raise ValueError(f'{scene}: invalid instance ID')
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise ValueError(f'{scene}: duplicate instance IDs')


def add_metadata(payload: dict, progress=None) -> dict:
    """Return a metadata-enriched copy of a ARKitScenes info payload."""
    if not isinstance(payload, dict) or not isinstance(
            payload.get('data_list'), list):
        raise TypeError('expected a dict payload containing data_list')

    validate_categories(payload)
    enriched = dict(payload)
    enriched['metainfo'] = dict(payload['metainfo'], dataset='arkitscenes')
    enriched_records = []
    total = len(payload['data_list'])
    for index, info in enumerate(payload['data_list'], 1):
        if not isinstance(info, dict):
            raise TypeError('every data_list item must be a dict')
        if progress is not None and (index == 1 or index % 10 == 0 or index == total):
            progress(f'Validating scene {index}/{total}: {scene_id_from_info(info)}')
        validate_scene(info)
        updated_info = dict(info)
        updated_info['scene_id'] = scene_id_from_info(info)
        instances = info.get('instances', [])
        if not isinstance(instances, list):
            raise TypeError('instances must be a list')
        updated_instances = []
        for instance_id, instance in enumerate(instances):
            if not isinstance(instance, dict):
                raise TypeError('every instance must be a dict')
            updated_instance = dict(instance)
            updated_instance['instance_id'] = int(instance.get('instance_id', instance_id))
            updated_instances.append(updated_instance)
        updated_info['instances'] = updated_instances
        enriched_records.append(updated_info)
    if len({Path(info['img_paths'][0]).parts[2] for info in enriched_records}) > 1:
        raise ValueError('train and val scenes must not share one annotation file')
    if len({info['scene_id'] for info in enriched_records}) != len(enriched_records):
        raise ValueError('duplicate scene IDs')
    enriched['data_list'] = enriched_records
    return enriched


def update_file(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Use a separate output path; preserve the original annotations')
    print(f'Loading annotations: {input_path}', flush=True)
    with input_path.open('rb') as handle:
        payload = pickle.load(handle)
    enriched = add_metadata(payload, progress=lambda message: print(message, flush=True))
    print(f'Writing metadata: {output_path}', flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode='wb', dir=output_path.parent,
            prefix=output_path.name + '.', suffix='.tmp', delete=False) as tmp:
        temporary_path = Path(tmp.name)
        pickle.dump(enriched, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.flush()
    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    update_file(args.input, args.output)
    print(f'Wrote metadata-enriched annotations to {args.output}')


if __name__ == '__main__':
    main()
