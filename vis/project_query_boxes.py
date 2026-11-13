"""Project NMS query boxes and colored scene points with the Blender GT camera."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def mesh_bounds(path):
    """Read ScanNet's binary PLY vertex positions for the Blender object bounds."""
    with path.open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY file: {path}')
        vertex_count = None
        vertex_properties = []
        current_element = None
        binary = False
        scalar = {'float': 'f4', 'float32': 'f4', 'double': 'f8',
                  'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
                  'ushort': 'u2', 'uint16': 'u2', 'short': 'i2', 'int16': 'i2',
                  'uint': 'u4', 'uint32': 'u4', 'int': 'i4', 'int32': 'i4'}
        while True:
            line = f.readline()
            if not line:
                raise ValueError('PLY header is incomplete')
            parts = line.decode('ascii').strip().split()
            if parts[:2] == ['format', 'binary_little_endian']:
                binary = True
            elif parts and parts[0] == 'element':
                current_element = parts[1]
                if current_element == 'vertex':
                    vertex_count = int(parts[2])
            elif parts and parts[0] == 'property' and current_element == 'vertex':
                if len(parts) != 3 or parts[1] not in scalar:
                    raise ValueError('Unsupported vertex property in PLY')
                vertex_properties.append((parts[2], '<' + scalar[parts[1]]))
            elif parts == ['end_header']:
                break
        if not binary or not vertex_count:
            raise ValueError('Expected a binary little-endian PLY with vertices')
        vertices = np.fromfile(f, dtype=np.dtype(vertex_properties), count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f'Incomplete mesh: {path}')
    positions = np.column_stack([vertices[name] for name in ('x', 'y', 'z')])
    return positions.min(axis=0), positions.max(axis=0)


def project(points, center, right, up, scale, size):
    relative = points - center
    return np.column_stack((size * (0.5 + relative @ right / scale),
                            size * (0.5 - relative @ up / scale)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', nargs='+', required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'query_nms_topk')
    parser.add_argument('--mesh-dir', type=Path, default=ROOT / 'data/scannet_val/scans')
    parser.add_argument('--output', type=Path, default=ROOT / 'projections_query')
    parser.add_argument('--resolution', type=int, default=3200)
    parser.add_argument('--point-radius', type=float, default=1.5, help='Point radius in output pixels')
    parser.add_argument('--line-width', type=int, default=6)
    args = parser.parse_args()
    if args.resolution <= 0 or args.point_radius <= 0 or args.line_width <= 0:
        parser.error('Resolution, point radius and line width must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    direction = np.array([1.15, -1.4, 2.5], dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corner_signs = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                             [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    failures = []
    for scene in args.scene:
        try:
            mesh = args.mesh_dir / scene / f'{scene}_vh_clean_2.ply'
            lo, hi = mesh_bounds(mesh)
            with np.load(args.data_dir / f'{scene}_query_nms_topk.npz', allow_pickle=False) as data:
                if str(data['coordinate_system'].item()) != 'axis_aligned_metric':
                    raise ValueError('Expected axis-aligned metric coordinates')
                transform = data['axis_alignment']
                points = data['gt_world']
                colors = data['gt_rgb']
                boxes = data['boxes_center_size_world']
                gt_boxes = data['gt_boxes_center_size_world']
            corners = (lo + hi) / 2 + corner_signs * (hi - lo) / 2
            transformed = corners @ transform[:3, :3].T + transform[:3, 3]
            low, high = transformed.min(axis=0), transformed.max(axis=0)
            center = (low + high) / 2
            scale = max(np.linalg.norm(high - low) * 1.08, 0.1)
            pixels = project(points, center, right, up, scale, args.resolution)
            depth_order = np.argsort((points - center) @ direction)
            image = Image.new('RGBA', (args.resolution, args.resolution), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            radius = args.point_radius
            for idx in depth_order:
                x, y = pixels[idx]
                if -radius <= x < args.resolution + radius and -radius <= y < args.resolution + radius:
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                                 fill=tuple(int(c) for c in colors[idx, :3]) + (255,))
            for box_set, color in ((boxes, (0, 110, 255, 255)), (gt_boxes, (255, 0, 0, 255))):
                for box in box_set:
                    box_corners = box[:3] + corner_signs * box[3:6] / 2
                    xy = project(box_corners, center, right, up, scale, args.resolution)
                    for a, b in edges:
                        draw.line((tuple(xy[a]), tuple(xy[b])), fill=color, width=args.line_width)
            destination = args.output / f'{scene}.png'
            image.save(destination)
            print(f'{scene}: {len(points)} points, {len(boxes)} query boxes, {len(gt_boxes)} GT boxes -> {destination}')
        except (OSError, ValueError, KeyError) as error:
            failures.append(scene)
            print(f'FAILED {scene}: {error}')
    if failures:
        raise SystemExit('Failed scenes: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
