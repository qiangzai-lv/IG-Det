"""Export box-selected ScanNet instance PLYs and depth-tested RGB projections.

Uses exactly the point assignment and visibility routines of the 2D annotation
generator. It does not filter instances by the annotation acceptance threshold.
Run from any directory; relative paths are resolved against the repository.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import pickle
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.scannet_3d_to_coco_bbox import (  # noqa: E402
    GeneratorConfig, VISIBLE, OCCLUDED, UNKNOWN, _PALETTE,
    _aabb_corners, _AABB_EDGES, _build_view_geometry, _prepare_objects,
    _scene_id, camera_matrix_for_view, load_axis_aligned_scene_points,
    load_depth_map, sample_view_indices,
    world_to_camera_from_aligned_pose,
)

LOGGER = logging.getLogger('visualize_training_instance_points')
GRAY = (160, 160, 160)
VERTEX_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ('instance_id', '<i4'), ('category_id', '<i4'),
])


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def category_color(label):
    return _PALETTE[int(label) % len(_PALETTE)]


def ply_header(count, edge_count=None):
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        'comment coordinates: axis-aligned ScanNet world, metres\n'
        f'element vertex {count}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'property int instance_id\nproperty int category_id\n')
    if edge_count is not None:
        header += (f'element edge {edge_count}\n'
                   'property int vertex1\nproperty int vertex2\n')
    return (header + 'end_header\n').encode('ascii')


def vertices(points, obj):
    data = np.empty(len(points), dtype=VERTEX_DTYPE)
    for axis, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, axis]
    for channel, name in enumerate(('red', 'green', 'blue')):
        data[name] = category_color(obj.label)[channel]
    data['instance_id'] = obj.instance_id
    data['category_id'] = obj.label + 1
    return data


def export_clouds(output, points, objects):
    """Save assigned instance points without downsampling or duplication."""
    instance_dir = output / 'instances'
    instance_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total = sum(len(obj.point_indices) for obj in objects)
    with (output / 'scene_instances.ply').open('wb') as combined:
        combined.write(ply_header(total))
        for obj in objects:
            data = vertices(points[obj.point_indices], obj)
            filename = f'instance_{obj.instance_id:04d}_class_{obj.label + 1:02d}.ply'
            with (instance_dir / filename).open('wb') as handle:
                handle.write(ply_header(len(data)))
                data.tofile(handle)
            data.tofile(combined)
            records.append(dict(
                instance_id=obj.instance_id, category_id=obj.label + 1,
                category_name=obj.category_name,
                color=list(category_color(obj.label)),
                point_count=len(data),
                exclusive_non_support_count=int(obj.exclusive.sum()),
                bbox_3d=obj.box.tolist(), ply=f'instances/{filename}'))
    edge_dtype = np.dtype([('vertex1', '<i4'), ('vertex2', '<i4')])
    with (output / 'scene_boxes.ply').open('wb') as handle:
        handle.write(ply_header(8 * len(objects), 12 * len(objects)))
        for obj in objects:
            vertices(_aabb_corners(obj.box), obj).tofile(handle)
        edges = np.empty(12 * len(objects), dtype=edge_dtype)
        for index in range(len(objects)):
            for offset, (first, second) in enumerate(_AABB_EDGES):
                edges[index * 12 + offset] = (index * 8 + first, index * 8 + second)
        edges.tofile(handle)
    return records


def raster_points(image, pixels, colors, depth, radius):
    """Within each pass the nearest projected point wins a pixel collision."""
    if not len(pixels):
        return
    order = np.argsort(-depth, kind='stable')
    xy = np.rint(pixels[order]).astype(np.int64)
    colors = colors[order]
    height, width = image.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x, y = xy[:, 0] + dx, xy[:, 1] + dy
            keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            # Resolve repeated pixel indices explicitly, rather than relying on
            # NumPy's unspecified repeated advanced-assignment order.
            linear = y[keep] * width + x[keep]
            _, reverse_indices = np.unique(linear[::-1], return_index=True)
            selected = len(linear) - 1 - reverse_indices
            image[y[keep][selected], x[keep][selected]] = colors[keep][selected]


def load_saved_annotations(path, root):
    """Index persisted boxes by scene and normalized RGB path, not frame order."""
    with path.open(encoding='utf-8') as handle:
        payload = json.load(handle)
    images, records = {}, {}
    for item in payload['images']:
        image_path = Path(item['file_name'])
        if image_path.is_absolute():
            image_path = image_path.relative_to(root)
        key = (str(item['scene_id']), image_path.as_posix())
        if key in records or int(item['id']) in images:
            raise ValueError(f'Duplicate saved annotation image: {key}')
        images[int(item['id'])] = key
        records[key] = dict(width=int(item['width']), height=int(item['height']), boxes=[])
    for item in payload['annotations']:
        records[images[int(item['image_id'])]]['boxes'].append(item)
    return records


def projection(root, info, points, objects, view_index, output, config, radius,
               saved_annotations):
    relative = Path(info['img_paths'][view_index])
    image_path = relative if relative.is_absolute() else root / relative
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert('RGB')).copy()
    height, width = rgb.shape[:2]
    image_key = relative.relative_to(root) if relative.is_absolute() else relative
    saved = saved_annotations[(_scene_id(info), image_key.as_posix())]
    if (saved['width'], saved['height']) != (width, height):
        raise ValueError(f'Saved annotation image size mismatch: {relative}')
    w2c = world_to_camera_from_aligned_pose(
        info['axis_align_matrix'], info['lidar2cam'][view_index])
    intrinsic = camera_matrix_for_view(info['cam2img'], view_index)
    geometry = _build_view_geometry(
        points, w2c, intrinsic, width, height,
        load_depth_map(root, relative, config.depth_scale), config)
    all_indices, all_colors, statistics = [], [], []
    for obj in objects:
        indices = obj.point_indices
        status = geometry.visibility[indices]
        inside = geometry.in_image[indices]
        count = lambda value: int(((status == value) & inside).sum())
        statistics.append(dict(
            instance_id=obj.instance_id, category_id=obj.label + 1,
            points_in_image=int(inside.sum()), visible=count(VISIBLE),
            occluded=count(OCCLUDED), unknown=count(UNKNOWN)))
        all_indices.append(indices[inside])
        all_colors.append(np.tile(
            np.array(category_color(obj.label), dtype=np.uint8), (int(inside.sum()), 1)))
    if all_indices:
        indices = np.concatenate(all_indices)
        colors = np.concatenate(all_colors)
        visible = geometry.visibility[indices] == VISIBLE
        # Draw unknown / occluded first; visible category colors take priority.
        for mask, paint in ((~visible, np.tile(GRAY, (len(indices), 1))),
                            (visible, colors)):
            raster_points(rgb, geometry.pixels[indices[mask]], paint[mask],
                          geometry.depth[indices[mask]], radius)
    rendered = Image.fromarray(rgb)
    draw = ImageDraw.Draw(rendered)
    objects_by_id = {obj.instance_id: obj for obj in objects}
    drawn_boxes = []
    for annotation in saved['boxes']:
        obj = objects_by_id[int(annotation['instance_id_3d'])]
        if int(annotation['category_id']) != obj.label + 1:
            raise ValueError(f'Saved 2D/3D category mismatch: {relative}, instance {obj.instance_id}')
        x, y, box_width, box_height = map(float, annotation['bbox'])
        box = (x, y, x + box_width, y + box_height)
        color = category_color(obj.label)
        draw.rectangle(tuple(map(float, box)), outline=color, width=2)
        text = f'{obj.category_name} #{obj.instance_id}'
        x, y = float(box[0]), float(box[1])
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill=(25, 25, 25))
        draw.text((x, y), text, fill=color)
        drawn_boxes.append(dict(annotation_id=int(annotation['id']),
                                instance_id=obj.instance_id,
                                category_id=int(annotation['category_id']),
                                bbox_xywh=list(annotation['bbox'])))
    filename = f'view_{view_index:04d}_{relative.stem}.jpg'
    rendered.save(output / filename, quality=95)
    return dict(view_index=view_index, image_path=str(relative),
                projection=filename, instances=statistics, saved_boxes=drawn_boxes)


def write_scene_index(output, summary):
    rows = []
    for obj in summary['instances']:
        color = ','.join(map(str, obj['color']))
        rows.append(
            f'<li style="color:rgb({color})">'
            f'{html.escape(obj["category_name"])} #{obj["instance_id"]} '
            f'({obj["point_count"]:,} points) '
            f'<a href="{obj["ply"]}">PLY</a></li>')
    views = ''.join(
        f'<figure><a href="{v["projection"]}"><img loading="lazy" '
        f'style="max-width:100%" src="{v["projection"]}"></a>'
        f'<figcaption>view {v["view_index"]}: '
        f'{html.escape(v["image_path"])}</figcaption></figure>'
        for v in summary['views'])
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Instance projections</title>'
        '<body style="background:#202124;color:#eee;font-family:sans-serif">'
        f'<h1>{html.escape(summary["scene_id"])}</h1>'
        '<p>Visible: category color. Occluded / unknown: gray. '
        'Rectangles: actual saved training boxes from the annotation JSON.</p>'
        '<a href="scene_instances.ply">Combined instance PLY</a> · '
        '<a href="scene_boxes.ply">3D box edges PLY</a><ul>'
        + ''.join(rows) + '</ul>' + views, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/ScanNet_processed_v2')
    parser.add_argument('--ann-file', default='scannet_infos_train_mvod_with_ids.pkl')
    parser.add_argument('--ann-file-2d', default='data/scannet_coco_v2/keypoints_bbox_train.json',
                        help='Saved COCO training boxes; relative to repository root')
    parser.add_argument('--output-dir', default='work_dirs/train_instance_points')
    parser.add_argument('--scene-id', nargs='+', help='Optional scene IDs; default: all training scenes')
    parser.add_argument('--num-views', type=int, default=100,
                        help='Uniform views per scene, capped at 100; -1 also uses this cap')
    parser.add_argument('--point-radius', type=int, default=2)
    parser.add_argument('--depth-scale', type=float, default=1000.)
    parser.add_argument('--abs-depth-tolerance', type=float, default=0.05)
    parser.add_argument('--rel-depth-tolerance', type=float, default=0.05)
    parser.add_argument('--depth-window-radius', type=int, default=1)
    args = parser.parse_args()
    if args.num_views == 0 or args.num_views < -1:
        parser.error('--num-views must be positive or -1')
    if args.point_radius < 0 or args.depth_window_radius < 0:
        parser.error('point and depth window radii must be non-negative')
    if (not np.isfinite([args.depth_scale, args.abs_depth_tolerance,
                         args.rel_depth_tolerance]).all()
            or args.depth_scale <= 0 or args.abs_depth_tolerance < 0
            or args.rel_depth_tolerance < 0):
        parser.error('depth scale must be positive; tolerances must be finite and non-negative')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    views_per_scene = min(args.num_views, 100) if args.num_views > 0 else 100
    root = resolve_path(args.data_root).resolve()
    ann = Path(args.ann_file).expanduser()
    ann = ann if ann.is_absolute() else root / ann
    output = resolve_path(args.output_dir).resolve()
    with ann.open('rb') as handle:
        payload = pickle.load(handle)
    scenes = payload['data_list']
    if args.scene_id:
        requested = set(args.scene_id)
        missing = requested - {_scene_id(info) for info in scenes}
        if missing:
            parser.error(f'unknown training scenes: {sorted(missing)}')
        scenes = [info for info in scenes if _scene_id(info) in requested]
    categories = {int(label): name for name, label in payload['metainfo']['categories'].items()}
    ann_2d = resolve_path(args.ann_file_2d).resolve()
    saved_annotations = load_saved_annotations(ann_2d, root)
    # Verify coverage before writing any output; missing images are not empty GT.
    for info in scenes:
        for view_index in sample_view_indices(
                len(info['img_paths']), views_per_scene, 'uniform', np.random.default_rng(0)):
            image_path = Path(info['img_paths'][int(view_index)])
            key_path = image_path.relative_to(root) if image_path.is_absolute() else image_path
            if (_scene_id(info), key_path.as_posix()) not in saved_annotations:
                raise ValueError(f'Image missing from saved 2D annotations: {image_path}')
    config = GeneratorConfig(
        depth_scale=args.depth_scale, abs_depth_tolerance=args.abs_depth_tolerance,
        rel_depth_tolerance=args.rel_depth_tolerance,
        depth_window_radius=args.depth_window_radius)
    output.mkdir(parents=True, exist_ok=True)
    links = []
    for index, info in enumerate(scenes):
        scene_id = _scene_id(info)
        if not scene_id or Path(scene_id).name != scene_id:
            raise ValueError(f'invalid scene ID: {scene_id!r}')
        scene_output = output / scene_id
        # A fresh directory prevents stale PLYs / views from mixing across runs.
        scene_output.mkdir(exist_ok=False)
        points = load_axis_aligned_scene_points(root, info)
        objects, _ = _prepare_objects(info, points, categories)
        LOGGER.info('[%d/%d] %s: %d points, %d instances',
                    index + 1, len(scenes), scene_id, len(points), len(objects))
        records = export_clouds(scene_output, points, objects)
        views = []
        for view_index in sample_view_indices(
                len(info['img_paths']), views_per_scene, 'uniform', np.random.default_rng(0)):
            views.append(projection(root, info, points, objects, int(view_index),
                                    scene_output, config, args.point_radius,
                                    saved_annotations))
            if len(views) == 1 or len(views) % 100 == 0:
                LOGGER.info('  %d views saved (latest view %d)', len(views), view_index)
        summary = dict(scene_id=scene_id, annotation_file=str(ann),
                       annotation_file_2d=str(ann_2d), box_source='Saved COCO bbox, no recomputation',
                       max_images_per_scene=views_per_scene,
                       point_selection='3D AABB points with unique ownership; overlaps prefer greater intersection/own-volume ratio (smaller box); no annotation rejection filter',
                       gray='OCCLUDED and UNKNOWN; only in-image projections are drawn',
                       config=vars(args), instances=records, views=views)
        (scene_output / 'summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        write_scene_index(scene_output, summary)
        links.append(f'<li><a href="{html.escape(scene_id)}/index.html">{html.escape(scene_id)}</a></li>')
        (output / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Training instances</title>'
            '<h1>Training instance point clouds and projections</h1><ul>'
            + ''.join(links) + '</ul>', encoding='utf-8')
    LOGGER.info('Complete: %s', output / 'index.html')


if __name__ == '__main__':
    main()
