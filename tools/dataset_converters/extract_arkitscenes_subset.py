#!/usr/bin/env python3
"""Extract a portable development subset from processed ARKitScenes.

Requires Python and NumPy only. Run on the machine holding the full dataset.
Original metadata, 7D boxes, labels, point clouds and camera conventions are
preserved. This is a development fixture, not a representative evaluation set.
"""
import argparse
import copy
import json
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np


VIEW_FIELDS = ('img_paths', 'depth_paths', 'cam2img', 'lidar2cam')


def relative_file(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a relative dataset path: {value}')
    source = root / path
    if not source.is_file():
        raise FileNotFoundError(source)
    if not source.stat().st_size:
        raise ValueError(f'Empty input file: {source}')
    return path


def scene_id(record, split):
    paths = record['img_paths']
    if not len(paths):
        raise ValueError('Scene has no image paths')
    first = Path(paths[0])
    expected = ('processed', '3dod', split)
    if len(first.parts) != 5 or first.parts[:3] != expected:
        raise ValueError(f'Unexpected {split} image path: {first}')
    identifier = first.parent.name
    if any(Path(p).parent != first.parent for p in paths):
        raise ValueError(f'Scene {identifier} contains different image directories')
    return identifier


def plan_scene(root, record, split):
    identifier = scene_id(record, split)
    count = len(record['img_paths'])
    for key in VIEW_FIELDS:
        if len(record[key]) != count:
            raise ValueError(f'{identifier}: {key} has {len(record[key])} entries, expected {count}')
    for image, depth in zip(record['img_paths'], record['depth_paths']):
        image = relative_file(root, image)
        depth = relative_file(root, depth)
        if not image.name.endswith('_color.png'):
            raise ValueError(f'Unexpected color filename: {image}')
        stem = image.name[:-len('_color.png')]
        if depth != image.with_name(stem + '_depth.png'):
            raise ValueError(f'RGB/depth frame mismatch: {image}, {depth}')
        pose = relative_file(root, image.with_name(stem + '_pose.npy'))
        matrix = np.load(root / pose, allow_pickle=False)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f'Invalid 4x4 pose: {pose}')
    for key, shape in (('cam2img', (3, 3)), ('lidar2cam', (4, 4))):
        matrices = np.asarray(record[key])
        if matrices.shape != (count, *shape) or not np.isfinite(matrices).all():
            raise ValueError(f'{identifier}: invalid {key} matrices: {matrices.shape}')
    directory = Path(record['img_paths'][0]).parent
    # Copy ALL files, including frames and auxiliary files absent from the pkl.
    files = set()
    for path in (root / directory).rglob('*'):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f'Linked subdirectory inside scene is unsupported: {path}')
        if path.is_file():
            files.add(path.relative_to(root))
    point_path = Path(record['lidar_points']['lidar_path'])
    if point_path.parent == Path('.'):
        point_path = Path('points') / point_path
    point_path = relative_file(root, point_path)
    if point_path.parent != Path('points') or point_path.name != f'{identifier}_points.bin':
        raise ValueError(f'{identifier}: unexpected point-cloud path {point_path}')
    features = int(record['lidar_points']['num_pts_feats'])
    point_bytes = (root / point_path).stat().st_size
    if features <= 0 or point_bytes % (features * np.dtype('float32').itemsize):
        raise ValueError(f'{identifier}: invalid point-cloud byte count or feature count')
    files.add(point_path)
    for instance in record['instances']:
        bbox = np.asarray(instance['bbox_3d'])
        if bbox.shape != (7,) or not np.isfinite(bbox).all():
            raise ValueError(f'{identifier}: expected finite 7D boxes')
    entry = dict(scene_id=identifier, views=count, instances=len(record['instances']),
                 scene_directory=directory.as_posix(), file_count=len(files),
                 point_file=point_path.as_posix())
    return copy.deepcopy(record), files, entry


def plan_split(root, split, requested, seed, skip_missing):
    with (root / f'arkit_infos_{split}.pkl').open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get('data_list'), list):
        raise ValueError(f'{split}: expected a dict with data_list and metainfo')
    if 'metainfo' not in payload:
        raise ValueError(f'{split}: missing metainfo')
    if requested > len(payload['data_list']):
        raise ValueError(f'{split}: requested {requested} scenes but only {len(payload["data_list"])} records exist')
    order = list(range(len(payload['data_list'])))
    random.Random(seed).shuffle(order)
    chosen, files, entries, skipped = [], set(), [], []
    seen = set()
    for index in order:
        if len(chosen) == requested:
            break
        record = payload['data_list'][index]
        identifier = scene_id(record, split)
        if identifier in seen:
            raise ValueError(f'{split}: duplicate scene ID {identifier}')
        seen.add(identifier)
        try:
            selected, scene_files, entry = plan_scene(root, record, split)
        except FileNotFoundError as error:
            if not skip_missing:
                raise
            skipped.append(dict(scene_id=identifier, missing_file=str(error)))
            continue
        entry['source_record_index'] = index
        chosen.append(selected)
        files.update(scene_files)
        entries.append(entry)
    if len(chosen) != requested:
        raise ValueError(f'{split}: found {len(chosen)} complete scenes, requested {requested}; '
                         f'{len(skipped)} scenes had missing required files')
    output = {key: copy.deepcopy(value) for key, value in payload.items() if key != 'data_list'}
    output['data_list'] = chosen
    return output, files, dict(source_records=len(payload['data_list']), scenes=entries, skipped=skipped)


def extract_subset(source, destination, train_scenes=2, test_scenes=1,
                   seed=2026, skip_missing=False, dry_run=False):
    if min(train_scenes, test_scenes) < 0 or train_scenes + test_scenes == 0:
        raise ValueError('Scene counts must be nonnegative and their total must be > 0')
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested directories')
    if destination.exists():
        raise FileExistsError(f'Destination already exists; choose a new directory: {destination}')
    planned, files, splits = {}, set(), {}
    for split, requested in (('train', train_scenes), ('val', test_scenes)):
        if requested == 0:
            continue
        annotation, split_files, report = plan_split(
            source, split, requested, seed, skip_missing)
        planned[split] = annotation
        files.update(split_files)
        splits[split] = report
    if 'train' in splits and 'val' in splits:
        train_ids = {entry['scene_id'] for entry in splits['train']['scenes']}
        val_ids = {entry['scene_id'] for entry in splits['val']['scenes']}
        if train_ids & val_ids:
            raise ValueError(f'Scene IDs overlap between train and val: {sorted(train_ids & val_ids)}')
    manifest = dict(development_only=True, source_root=str(source), seed=seed,
                    selection_unit='whole_scene', test_source_split='val', splits=splits,
                    files=[path.as_posix() for path in sorted(files)],
                    copied_file_bytes=sum((source / path).stat().st_size for path in files),
                    metadata_preserved=True, boxes_and_point_clouds_preserved=True)
    if dry_run:
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.', dir=destination.parent))
    try:
        for split in splits.values():
            for entry in split['scenes']:
                directory = Path(entry['scene_directory'])
                shutil.copytree(source / directory, staging / directory, copy_function=shutil.copy2)
                point = Path(entry['point_file'])
                (staging / point).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / point, staging / point)
        for split, annotation in planned.items():
            with (staging / f'arkit_infos_{split}.pkl').open('wb') as handle:
                pickle.dump(annotation, handle, protocol=4)
        (staging / 'subset_manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        if destination.exists():
            raise FileExistsError(f'Destination appeared during extraction: {destination}')
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Full ARKitScenes_processed root')
    parser.add_argument('--output', required=True, type=Path, help='New self-contained output directory')
    parser.add_argument('--train-scenes', type=int, default=2)
    parser.add_argument('--test-scenes', type=int, default=1, help='Whole scenes from the val split')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--skip-missing', action='store_true', help='Skip scenes with missing required files')
    parser.add_argument('--dry-run', action='store_true', help='Validate and report without writing files')
    args = parser.parse_args()
    try:
        report = extract_subset(args.source, args.output, args.train_scenes, args.test_scenes,
                                args.seed, args.skip_missing, args.dry_run)
    except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError) as error:
        parser.exit(1, f'Error: {error}\n')
    summary = {key: value for key, value in report.items() if key != 'files'}
    summary['copied_file_count'] = len(report['files'])
    summary['copied_file_mib'] = round(report['copied_file_bytes'] / 2**20, 2)
    summary['dry_run'] = args.dry_run
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
