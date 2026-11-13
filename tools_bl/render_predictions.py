"""Render ScanNet prediction boxes on the aligned scene mesh.

Run with Blender:
    blender -b --python render_predictions.py -- --scene scene0025_02
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import render_scannet_gt as gt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', nargs='+', required=True, help='One or more scene IDs')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'gt_iou_predictions')
    parser.add_argument('--mesh-dir', type=Path, default=ROOT / 'data/scannet_val/scans')
    parser.add_argument('--output', type=Path, default=ROOT / 'renders_predictions')
    parser.add_argument('--resolution', type=int, default=3200)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Cycles render device (default: cpu)')
    parser.add_argument('--box-radius', type=float, default=0.015)
    parser.add_argument('--raw', action='store_true', help='Render raw predictions instead of GT-IoU-selected predictions')
    parser.add_argument('--score-threshold', type=float, default=0.0)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if args.resolution <= 0 or args.samples <= 0 or args.box_radius <= 0:
        parser.error('Resolution, samples and box radius must be positive')
    args.output.mkdir(parents=True, exist_ok=True)

    original_make_boxes = gt.make_boxes

    def make_prediction_boxes(instances, radius):
        boxes = original_make_boxes(instances, radius)
        boxes.name = 'Prediction boxes'
        material = boxes.data.materials[0]
        material.name = 'Prediction blue'
        shader = next(node for node in material.node_tree.nodes if node.bl_idname == 'ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = (0.0, 0.25, 1.0, 1.0)
        (shader.inputs.get('Emission Color') or shader.inputs['Emission']).default_value = (0.0, 0.25, 1.0, 1.0)
        return boxes

    gt.make_boxes = make_prediction_boxes
    failed = []
    for scene in args.scene:
        try:
            path = args.predictions / f'{scene}_gt_iou_predictions.npz'
            with np.load(path, allow_pickle=False) as data:
                coordinate_system = str(data['coordinate_system'].item())
                if coordinate_system != 'axis_aligned_metric':
                    raise ValueError(f'Unsupported coordinate system: {coordinate_system}')
                prefix = 'raw_' if args.raw else ''
                boxes = data[prefix + 'boxes_center_size_world']
                scores = data[prefix + 'scores']
                if boxes.ndim != 2 or boxes.shape[1] != 6 or len(boxes) != len(scores):
                    raise ValueError('Inconsistent prediction box and score shapes')
                if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
                    raise ValueError('Prediction boxes or scores contain non-finite values')
                selected = boxes[scores >= args.score_threshold]
                if np.any(selected[:, 3:] <= 0):
                    raise ValueError('Prediction box dimensions must be positive')
                info = {'axis_align_matrix': data['axis_alignment'].tolist(),
                        'instances': [{'bbox_3d': box.tolist()} for box in selected]}
            print(f'{scene}: rendering {len(selected)} prediction boxes', flush=True)
            if not len(selected):
                print(f'{scene}: no boxes passed selection', file=sys.stderr, flush=True)
            gt.render(scene, info, args)
        except Exception as error:
            failed.append(scene)
            traceback.print_exc()
            print(f'FAILED {scene}: {error}', file=sys.stderr, flush=True)
    if failed:
        raise RuntimeError('Failed scenes: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
