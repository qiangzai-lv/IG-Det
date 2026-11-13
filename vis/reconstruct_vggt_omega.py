"""Reconstruct selected RGB scenes with VGGT-Omega and export paper assets.

No dataset depth, pose text, intrinsics or GT point clouds are read. All views
of one scene are inferred jointly, keeping one consistent predicted frame.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.geometry import unproject_depth_map_to_point_map_torch
from vggt_omega.utils.load_fn import (
    load_and_preprocess_images, _crop_to_supported_aspect_ratio,
    _balanced_target_shape, _max_size_target_shape)
from vggt_omega.utils.pose_enc import encoding_to_camera

LOGGER = logging.getLogger('vggt_omega_reconstruction')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VERTEX_DTYPE = np.dtype([(name, '<f4') for name in ('x', 'y', 'z')] +
                        [(name, 'u1') for name in ('red', 'green', 'blue')])


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.lower())
                 for part in re.split(r'(\d+)', Path(path).name))


def image_files(folder):
    # Intentionally non-recursive: ScanNet depth/*.png must never be RGB input.
    return sorted((p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS), key=natural_key)


def find_image_directory(scene):
    scene = Path(scene)
    for folder in (scene, scene / 'color', scene / 'rgb', scene / 'images'):
        if folder.is_dir() and image_files(folder):
            return folder
    return None


def discover_scenes(root, selected=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {root}')
    if selected:
        scenes = []
        for name in selected:
            scene = root / name
            folder = find_image_directory(scene) if scene.is_dir() else None
            if folder is None:
                raise FileNotFoundError(f'No RGB images found for requested scene: {scene}')
            scenes.append((scene.name, folder))
    elif image_files(root):
        scenes = [(root.name, root)]
    else:
        scenes = []
        for scene in sorted((p for p in root.iterdir() if p.is_dir()), key=natural_key):
            folder = find_image_directory(scene)
            if folder is not None:
                scenes.append((scene.name, folder))
        if not scenes:
            folder = find_image_directory(root)
            if folder is not None:
                scenes = [(root.name, folder)]
    if not scenes:
        raise ValueError(f'No scenes with RGB images found under {root}')
    if len({name for name, _ in scenes}) != len(scenes):
        raise ValueError('Scene names must be unique')
    return scenes


def uniform_indices(total, requested):
    if total < 1 or requested == 0 or requested < -1:
        raise ValueError('Image count must be positive, or -1 for all images')
    count = total if requested == -1 else min(total, requested)
    # No repeated frames when fewer images exist than requested.
    indices = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    assert len(np.unique(indices)) == count
    return indices


def preprocess(paths, mode, resolution):
    images = load_and_preprocess_images(
        [str(p) for p in paths], mode=mode, image_resolution=resolution, patch_size=16)
    height, width = images.shape[-2:]
    valid = np.zeros((len(paths), height, width), dtype=bool)
    metadata = []
    target_shape = _balanced_target_shape if mode == 'balanced' else _max_size_target_shape
    for index, path in enumerate(paths):
        with Image.open(path) as original:
            original_shape = [original.height, original.width]
            cropped = _crop_to_supported_aspect_ratio(original)
            h, w = target_shape(cropped.height / cropped.width, resolution, 16)
        top, left = (height - h) // 2, (width - w) // 2
        valid[index, top:top+h, left:left+w] = True
        metadata.append(dict(original_shape=original_shape, resized_shape=[h, w],
                             padding_top=top, padding_left=left))
    return images, valid, metadata


def save_ply(path, points, colors):
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('PLY points and RGB colors must have shape [N, 3]')
    data = np.empty(len(points), dtype=VERTEX_DTYPE)
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        data[name] = colors[:, i]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment VGGT-Omega predicted reconstruction coordinates; scale ambiguous\n'
              f'element vertex {len(points)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        data.tofile(stream)


def filter_cloud(point_map, rgb, depth, confidence, valid_pixels,
                 stride=2, confidence_percentile=20., max_points=1000000):
    xyz = point_map[:, ::stride, ::stride].reshape(-1, 3)
    colors = rgb[:, ::stride, ::stride].reshape(-1, 3)
    z = depth[:, ::stride, ::stride].reshape(-1)
    conf = confidence[:, ::stride, ::stride].reshape(-1)
    mask = valid_pixels[:, ::stride, ::stride].reshape(-1)
    mask = (mask & np.isfinite(xyz).all(1) & np.isfinite(z) & (z > 0)
            & np.isfinite(conf))
    if not mask.any():
        raise ValueError('No finite positive-depth reconstruction points')
    threshold = float(np.percentile(conf[mask], confidence_percentile))
    mask &= conf >= threshold
    points, colors = xyz[mask], colors[mask]
    before_cap = len(points)
    if max_points > 0 and len(points) > max_points:
        take = uniform_indices(len(points), max_points)
        points, colors = points[take], colors[take]
    return points, colors, dict(confidence_threshold=threshold,
                               point_count_before_cap=before_cap, point_count=len(points))


def save_contact_sheet(rgb, names, destination):
    thumb_w, thumb_h, columns = 192, 144, min(8, len(rgb))
    rows = (len(rgb) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * thumb_w, rows * (thumb_h + 22)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (frame, name) in enumerate(zip(rgb, names)):
        image = Image.fromarray(frame)
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + 22)
        sheet.paste(image, (x + (thumb_w-image.width)//2, y))
        draw.text((x+4, y+thumb_h+3), f'{index:03d}: {name}', fill='black')
    sheet.save(destination)


def save_preview(points, colors, destination, elev, azim, max_points=200000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indices = uniform_indices(len(points), min(len(points), max_points))
    # Display-only OpenCV-to-y-up basis change; PLY/cameras are never rotated.
    display = points[indices][:, [0, 2, 1]].copy()
    display[:, 2] *= -1
    lo, hi = np.percentile(display, [1, 99], axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi-lo).max()) * .55, 1e-6)
    fig = plt.figure(figsize=(8, 8), dpi=240)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    ax.scatter(*display.T, c=colors[indices] / 255., s=.4, linewidths=0,
               depthshade=False, rasterized=True)
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(destination, facecolor='white', dpi=240)
    fig.savefig(Path(destination).with_suffix('.pdf'), facecolor='white', dpi=240)
    plt.close(fig)


def reconstruct_scene(model, scene_name, image_dir, output_root, device, args):
    started = time.monotonic()
    available = image_files(image_dir)
    indices = uniform_indices(len(available), args.num_images)
    selected = [available[i] for i in indices]
    folder = output_root / scene_name
    if folder.exists() and any(folder.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output already exists: {folder}; use --overwrite or a new output directory')
    folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info('%s: uniformly selected %d/%d RGB images', scene_name, len(selected), len(available))
    images, valid_pixels, image_metadata = preprocess(selected, args.preprocess_mode, args.image_resolution)
    rgb = (images.permute(0, 2, 3, 1).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    with torch.inference_mode():
        predictions = model(images.to(device))
        height, width = images.shape[-2:]
        extrinsics, intrinsics = encoding_to_camera(predictions['pose_enc'], (height, width))
        depth = predictions['depth']
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        confidence = predictions['depth_conf']
        if confidence.ndim == 5 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if depth.shape != (1, len(selected), height, width) or confidence.shape != depth.shape:
            raise ValueError(f'Unexpected depth/confidence shapes {depth.shape}, {confidence.shape}')
        if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
            raise FloatingPointError('Predicted cameras contain non-finite values')
        if (intrinsics[..., 0, 0] <= 0).any() or (intrinsics[..., 1, 1] <= 0).any():
            raise ValueError('Predicted focal lengths must be positive')
        points = unproject_depth_map_to_point_map_torch(depth.float(), extrinsics, intrinsics)
        points = points[0].cpu().numpy()
        depth = depth[0].float().cpu().numpy()
        confidence = confidence[0].float().cpu().numpy()
        extrinsics_np = extrinsics[0].float().cpu().numpy()
        intrinsics_np = intrinsics[0].float().cpu().numpy()
    del predictions, extrinsics, intrinsics
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    points, colors, stats = filter_cloud(
        points, rgb, depth, confidence, valid_pixels, args.pixel_stride,
        args.confidence_percentile, args.max_points)
    save_ply(folder / 'reconstruction.ply', points, colors)
    camera_centers = -np.einsum('vji,vj->vi', extrinsics_np[:, :3, :3], extrinsics_np[:, :3, 3])
    save_ply(folder / 'camera_centers.ply', camera_centers,
             np.tile(np.array([[230, 60, 40]], dtype=np.uint8), (len(selected), 1)))
    np.savez_compressed(folder / 'cameras.npz', extrinsics_w2c=extrinsics_np,
                        intrinsics=intrinsics_np, camera_centers=camera_centers,
                        source_indices=indices, image_names=np.array([p.name for p in selected]),
                        image_size_hw=np.array(images.shape[-2:]))
    if args.save_dense:
        np.savez_compressed(folder / 'depth_confidence.npz', depth=depth,
                            confidence=confidence, valid_pixels=valid_pixels)
    image_output = folder / 'images'
    image_output.mkdir(exist_ok=True)
    if args.overwrite:
        for old_image in image_output.glob('*.png'):
            old_image.unlink()
        if not args.save_dense:
            (folder / 'depth_confidence.npz').unlink(missing_ok=True)
        if args.no_preview:
            (folder / 'reconstruction.png').unlink(missing_ok=True)
            (folder / 'reconstruction.pdf').unlink(missing_ok=True)
    for i, (frame, source) in enumerate(zip(rgb, selected)):
        Image.fromarray(frame).save(image_output / f'{i:03d}_{source.stem}.png')
    save_contact_sheet(rgb, [p.name for p in selected], folder / 'sampled_views.jpg')
    if not args.no_preview:
        save_preview(points, colors, folder / 'reconstruction.png', args.preview_elev, args.preview_azim)
    manifest = dict(scene=scene_name, source_directory=str(image_dir),
                    checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
                    device=str(device), model='VGGT-Omega',
                    coordinate_frame='predicted reconstruction world; no GT alignment; non-metric scale',
                    extrinsics_convention='OpenCV world-to-camera [R|t]',
                    available_images=len(available), requested_images=args.num_images,
                    selected_images=len(selected),
                    sampling='natural filename order, rounded linspace including endpoints',
                    selected=[dict(source=str(p), source_index=int(i), **meta)
                              for p, i, meta in zip(selected, indices, image_metadata)],
                    preprocessing=dict(mode=args.preprocess_mode, resolution=args.image_resolution,
                                       output_shape=list(images.shape), range='RGB [0,1]'),
                    filtering=dict(pixel_stride=args.pixel_stride,
                                   confidence_percentile=args.confidence_percentile,
                                   max_points=args.max_points, **stats),
                    preview=dict(elev=args.preview_elev, azim=args.preview_azim,
                                 display_axes='x,z,-y', limits='cube from 1%-99% point quantiles'),
                    elapsed_seconds=time.monotonic()-started,
                    checkpoint_size_bytes=Path(args.checkpoint).stat().st_size)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    LOGGER.info('%s: saved %s points to %s (%.1fs)', scene_name, f'{len(points):,}', folder, manifest['elapsed_seconds'])
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default='/data/data/scannet_vis')
    parser.add_argument('--scenes', nargs='+', help='Optional scene directory names; default: all scenes')
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'data/pretrain/vggt_omega_1b_512.pt'))
    parser.add_argument('--output-dir', default=str(REPO_ROOT / 'vis/outputs/vggt_omega'))
    parser.add_argument('--num-images', type=int, default=40, help='Uniformly selected images per scene; -1 means all')
    parser.add_argument('--image-resolution', type=int, default=512)
    parser.add_argument('--preprocess-mode', choices=['balanced', 'max_size'], default='max_size')
    parser.add_argument('--device', default='auto', help='auto, cuda, cuda:0, npu, cpu')
    parser.add_argument('--pixel-stride', type=int, default=2)
    parser.add_argument('--confidence-percentile', type=float, default=20.)
    parser.add_argument('--max-points', type=int, default=1000000, help='0 keeps all filtered points')
    parser.add_argument('--save-dense', action='store_true', help='Save predicted depths/confidences for re-rendering')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--preview-elev', type=float, default=25.)
    parser.add_argument('--preview-azim', type=float, default=-65.)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--list-scenes', action='store_true', help='Show selected scenes/counts without loading the model')
    args = parser.parse_args(argv)
    if args.num_images == 0 or args.num_images < -1:
        parser.error('--num-images must be positive or -1')
    if args.image_resolution <= 0 or args.image_resolution % 16:
        parser.error('--image-resolution must be a positive multiple of 16')
    if args.pixel_stride < 1 or not 0 <= args.confidence_percentile <= 100 or args.max_points < 0:
        parser.error('Invalid stride, confidence percentile or maximum point count')
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(4)
    scenes = discover_scenes(args.input_root, args.scenes)
    if args.list_scenes:
        for name, folder in scenes:
            total = len(image_files(folder))
            print(f'{name}: {total} available, {len(uniform_indices(total, args.num_images))} selected ({folder})')
        return
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.checkpoint = str(checkpoint)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            try:
                import torch_npu  # noqa: F401
                device = torch.device('npu' if torch.npu.is_available() else 'cpu')
            except ImportError:
                device = torch.device('cpu')
    else:
        if args.device.startswith('npu'):
            import torch_npu  # noqa: F401
        device = torch.device(args.device)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Strict checkpoint loading prevents accidentally using a detector checkpoint
    # or silently random-initialized camera/depth heads.
    LOGGER.info('Loading VGGT-Omega checkpoint: %s', checkpoint)
    model = VGGTOmega()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    records = []
    try:
        for name, folder in scenes:
            records.append(reconstruct_scene(model, name, folder, output, device, args))
            (output / 'index.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('CUDA memory exhausted. Reduce --num-images or --image-resolution; '
                           'views are not split into independently scaled reconstructions.') from exc
    LOGGER.info('Finished all %d scenes: %s', len(records), output)


if __name__ == '__main__':
    main()
