"""Exact greedy farthest-point sampling and colored scene/FPS PLY export."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData, PlyElement
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger('scene_fps')
VERTEX_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ('point_type', 'u1'), ('fps_index', '<i4'), ('source_index', '<i4')])


def load_cloud(path):
    ply = PlyData.read(str(path))
    vertices = ply['vertex'].data
    names = vertices.dtype.names
    if not all(axis in names for axis in ('x', 'y', 'z')):
        raise ValueError(f'PLY has no x/y/z vertex properties: {path}')
    points = np.column_stack([vertices[key] for key in ('x', 'y', 'z')]).astype(np.float32)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f'PLY must contain nonempty finite coordinates: {path}')
    if len(points) > np.iinfo(np.int32).max:
        raise ValueError('Cloud exceeds the supported PLY source-index range')
    colors = (np.column_stack([vertices[key] for key in ('red', 'green', 'blue')]).astype(np.uint8)
              if all(key in names for key in ('red', 'green', 'blue'))
              else np.full_like(points, 160, dtype=np.uint8))
    return points, colors


@torch.no_grad()
def farthest_point_sampling(points, count, start_index=0, device='cpu'):
    """Greedy FPS over ALL input vertices; unique source indices, no random pool.

    Start at a fixed source index and iteratively maximize the squared distance
    to the closest selected point. One isotropic normalization stabilizes the
    arithmetic without changing the geometric distance ordering. Original
    coordinates are never modified. count>N returns each source index once.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('FPS needs a nonempty [N, 3] point array')
    if not np.isfinite(points).all():
        raise ValueError('FPS input coordinates must be finite')
    if count < 1 or not 0 <= start_index < len(points):
        raise ValueError('FPS count must be positive and start index must be in range')
    count = min(count, len(points))
    # Use float64 only for CPU normalization, before the float32 FPS kernels.
    normalized = points.astype(np.float64) - points[start_index].astype(np.float64)
    scale = np.abs(normalized).max()
    if scale > 0:
        normalized /= scale
    xyz = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    result = torch.empty(count, dtype=torch.long, device=device)
    nearest = torch.full((len(points),), torch.inf, dtype=torch.float32, device=device)
    used = torch.zeros(len(points), dtype=torch.bool, device=device)
    current = torch.tensor(start_index, dtype=torch.long, device=device)
    for step in range(count):
        result[step] = current
        used[current] = True
        center = xyz.index_select(0, current.reshape(1))
        distances = (xyz - center).square().sum(-1)
        torch.minimum(nearest, distances, out=nearest)
        if step + 1 < count:
            current = nearest.masked_fill(used, -1).argmax()
    return result.cpu().numpy()


def sphere_directions(count):
    if count == 0:
        return np.empty((0, 3), dtype=np.float32)
    i = np.arange(count, dtype=np.float64)
    z = 1 - 2 * (i + .5) / count
    angle = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(np.maximum(0, 1 - z * z))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)).astype(np.float32)


def export_ply(path, points, colors, source_indices, point_types, fps_indices):
    data = np.empty(len(points), dtype=VERTEX_DTYPE)
    for index, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, index]
    for index, name in enumerate(('red', 'green', 'blue')):
        data[name] = colors[:, index]
    data['source_index'] = source_indices
    data['point_type'] = point_types
    data['fps_index'] = fps_indices
    PlyData([PlyElement.describe(data, 'vertex')], text=False, byte_order='<',
            comments=['Original reconstruction coordinates and ambiguous scale preserved.',
                      'point_type: 0=scene, 1=exact FPS center, 2=display sphere surface.',
                      'fps_index: -1 for context; source_index references the input PLY.']).write(str(path))


def make_overlay(points, colors, indices, marker_color, marker_radius,
                 marker_points, context_gray=False):
    samples = points[indices]
    sample_ids = np.arange(len(indices), dtype=np.int32)
    directions = sphere_directions(marker_points) if marker_radius > 0 else sphere_directions(0)
    shell = (samples[:, None] + marker_radius * directions[None]).reshape(-1, 3)
    marker_count = len(shell)
    context_colors = np.full_like(colors, 175) if context_gray else colors
    all_points = np.concatenate((points, samples, shell))
    all_colors = np.concatenate((context_colors,
                                np.tile(marker_color, (len(samples) + marker_count, 1))))
    source_indices = np.concatenate((np.arange(len(points), dtype=np.int32),
                                     indices.astype(np.int32),
                                     np.repeat(indices.astype(np.int32), len(directions))))
    fps_indices = np.concatenate((np.full(len(points), -1, dtype=np.int32), sample_ids,
                                  np.repeat(sample_ids, len(directions))))
    types = np.concatenate((np.zeros(len(points), dtype=np.uint8),
                            np.ones(len(samples), dtype=np.uint8),
                            np.full(marker_count, 2, dtype=np.uint8)))
    return all_points, all_colors, source_indices, types, fps_indices


def preview(points, colors, samples, color, path, elev, azim, context_gray):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    take = np.linspace(0, len(points)-1, min(len(points), 200000)).round().astype(np.int64)
    context = points[take][:, [0, 2, 1]].copy()
    context[:, 2] *= -1
    selected = samples[:, [0, 2, 1]].copy()
    selected[:, 2] *= -1
    lo, hi = points.min(0)[[0, 2, 1]], points.max(0)[[0, 2, 1]]
    # Apply the display sign to the range and restore increasing z bounds.
    lo[2], hi[2] = -hi[2], -lo[2]
    middle = (lo + hi) / 2
    extent = max(float((hi-lo).max()) * .55, 1e-5)
    fig = plt.figure(figsize=(8, 8), dpi=240)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho', computed_zorder=False)
    rgb = np.full((len(take), 3), .7) if context_gray else colors[take] / 255.
    ax.scatter(*context.T, c=rgb, s=.4, linewidths=0, depthshade=False, rasterized=True, zorder=1)
    # An explicit overlay keeps even occluded samples visible in the 2D diagram.
    ax.scatter(*selected.T, c=[color / 255.], s=24, edgecolors='white',
               linewidths=.35, depthshade=False, zorder=10)
    ax.set(xlim=(middle[0]-extent, middle[0]+extent),
           ylim=(middle[1]-extent, middle[1]+extent),
           zlim=(middle[2]-extent, middle[2]+extent))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, facecolor='white', dpi=240)
    fig.savefig(Path(path).with_suffix('.pdf'), facecolor='white', dpi=240)
    plt.close(fig)


def sample_scene(source, destination, args, device):
    started = time.monotonic()
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output exists: {destination}; use a new directory or --overwrite')
    points, colors = load_cloud(source)
    indices = farthest_point_sampling(points, args.num_points, args.start_index, device)
    color = np.array([int(args.marker_color[i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    lo, hi = np.percentile(points, [1, 99], axis=0)
    radius = (args.marker_radius if args.marker_radius is not None else
              float(np.linalg.norm(hi-lo)) * args.marker_radius_ratio)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / 'fps_indices.npy', indices)
    export_ply(destination / 'fps_points.ply', points[indices],
               np.tile(color, (len(indices), 1)), indices,
               np.ones(len(indices), dtype=np.uint8), np.arange(len(indices), dtype=np.int32))
    arrays = make_overlay(points, colors, indices, color, radius, args.marker_points, args.context_gray)
    export_ply(destination / 'scene_with_fps.ply', *arrays)
    if not args.no_preview:
        preview(points, colors, points[indices], color, destination / 'scene_with_fps.png',
                args.preview_elev, args.preview_azim, args.context_gray)
    record = dict(source_ply=str(source.resolve()), scene_points=len(points),
                  requested_samples=args.num_points, sampled_points=len(indices),
                  algorithm='exact greedy FPS on all source vertices; squared Euclidean distance',
                  start_index=args.start_index, device=str(device),
                  marker_color=args.marker_color, marker_radius=radius,
                  marker_surface_points_per_sample=(args.marker_points if radius > 0 else 0),
                  combined_vertices=len(arrays[0]), context_gray=args.context_gray,
                  coordinate_frame='unchanged from source PLY',
                  preview='display x,z,-y; all-point bounds; samples overlaid even when occluded',
                  elapsed_seconds=time.monotonic()-started)
    (destination / 'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))
    LOGGER.info('%s: FPS %d/%s, combined PLY: %s (%.1fs)', source.parent.name,
                len(indices), f'{len(points):,}', destination / 'scene_with_fps.ply', record['elapsed_seconds'])
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=str(ROOT / 'vis/outputs/vggt_omega'),
                        help='A scene PLY, one reconstruction directory, or a root with scene directories')
    parser.add_argument('--scenes', nargs='+', help='Optional scene names within the input root')
    parser.add_argument('--output-dir', default=str(ROOT / 'vis/outputs/vggt_omega_fps'))
    parser.add_argument('--num-points', type=int, default=256)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--device', default='auto', help='auto, cuda, cuda:0, or cpu')
    parser.add_argument('--marker-color', default='#EF3340')
    parser.add_argument('--marker-radius', type=float, help='Sphere radius in source cloud units')
    parser.add_argument('--marker-radius-ratio', type=float, default=.006,
                        help='Default radius relative to the 1%-99% scene diagonal')
    parser.add_argument('--marker-points', type=int, default=192,
                        help='Display sphere surface vertices per sample; 0 disables sphere markers')
    parser.add_argument('--context-gray', action='store_true', help='Gray context instead of original RGB')
    parser.add_argument('--preview-elev', type=float, default=25.)
    parser.add_argument('--preview-azim', type=float, default=-65.)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.num_points < 1 or args.start_index < 0 or args.marker_points < 0:
        parser.error('Invalid sample count, start index or marker point count')
    if (not np.isfinite(args.marker_radius_ratio) or args.marker_radius_ratio < 0 or
            (args.marker_radius is not None and (not np.isfinite(args.marker_radius) or args.marker_radius < 0))):
        parser.error('Marker radius must be finite and nonnegative')
    import re
    if not re.fullmatch(r'#[0-9A-Fa-f]{6}', args.marker_color):
        parser.error('--marker-color must use #RRGGBB format')
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(4)
    source = Path(args.input_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if source.is_file():
        sources = [source]
    elif source.is_dir() and (source / 'reconstruction.ply').is_file():
        sources = [source / 'reconstruction.ply']
    elif source.is_dir():
        directories = ([source / name for name in args.scenes] if args.scenes else
                       sorted(p for p in source.iterdir() if p.is_dir()))
        sources = [folder / 'reconstruction.ply' for folder in directories
                   if (folder / 'reconstruction.ply').is_file()]
        if args.scenes and len(sources) != len(args.scenes):
            raise FileNotFoundError('A requested scene has no reconstruction.ply')
    else:
        raise FileNotFoundError(source)
    if not sources:
        raise ValueError(f'No reconstruction PLY files found under {source}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sources:
        results.append(sample_scene(path, output / path.parent.name, args, device))
        (output / 'index.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    LOGGER.info('Completed %d scenes: %s', len(results), output)


if __name__ == '__main__':
    main()
