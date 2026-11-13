"""Render ScanNet meshes and ground-truth boxes with Blender.

Example:
    blender -b -t 4 --python render_scannet_gt.py -- --scene scene0568_00
"""
import argparse
import pickle
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', nargs='+', help='Scene IDs; default: all downloaded scenes')
    parser.add_argument('--mesh-dir', type=Path, default=ROOT / 'data/scannet_val/scans')
    parser.add_argument('--annotations', type=Path, default=ROOT / 'scannet_infos_val_pts.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'renders_gt')
    parser.add_argument('--resolution', type=int, default=3200)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--box-radius', type=float, default=0.015)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])


def make_boxes(instances, radius):
    curve = bpy.data.curves.new('GT box edges', 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    for instance in instances:
        x, y, z, dx, dy, dz = instance['bbox_3d'][:6]
        corners = [Vector((x + sx * dx / 2, y + sy * dy / 2, z + sz * dz / 2))
                   for sx, sy, sz in ((-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                                      (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1))]
        for a, b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
            line = curve.splines.new('POLY')
            line.points.add(1)
            line.points[0].co = (*corners[a], 1)
            line.points[1].co = (*corners[b], 1)
    obj = bpy.data.objects.new('GT boxes', curve)
    bpy.context.collection.objects.link(obj)
    material = bpy.data.materials.new('GT box red')
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.bl_idname == 'ShaderNodeBsdfPrincipled')
    shader.inputs['Base Color'].default_value = (1, 0, 0, 1)
    emission_color = shader.inputs.get('Emission Color') or shader.inputs['Emission']
    emission_color.default_value = (1, 0, 0, 1)
    if shader.inputs.get('Emission Strength'):
        shader.inputs['Emission Strength'].default_value = 1
    curve.materials.append(material)
    return obj


def render(scene_id, info, args):
    mesh_path = args.mesh_dir / scene_id / f'{scene_id}_vh_clean_2.ply'
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.ply_import(filepath=str(mesh_path))
    mesh = bpy.context.selected_objects[0]
    mesh.matrix_world = Matrix(info['axis_align_matrix'])
    bpy.context.view_layer.update()
    color = mesh.data.color_attributes.active_color
    if color is None:
        raise ValueError(f'No vertex colors in {mesh_path}')
    mat = bpy.data.materials.new('Vertex colors')
    mat.use_nodes = True
    shader = next(node for node in mat.node_tree.nodes if node.bl_idname == 'ShaderNodeBsdfPrincipled')
    vertex = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    vertex.layer_name = color.name
    mat.node_tree.links.new(vertex.outputs['Color'], shader.inputs['Base Color'])
    mat.node_tree.links.new(vertex.outputs['Color'], shader.inputs.get('Emission Color') or shader.inputs['Emission'])
    shader.inputs['Roughness'].default_value = 0.85
    if shader.inputs.get('Emission Strength'):
        shader.inputs['Emission Strength'].default_value = 0.4
    mesh.data.materials.clear()
    mesh.data.materials.append(mat)
    make_boxes(info['instances'], args.box_radius)

    points = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    lo = Vector([min(point[k] for point in points) for k in range(3)])
    hi = Vector([max(point[k] for point in points) for k in range(3)])
    center = (lo + hi) / 2
    extent = hi - lo
    scale = max(extent)
    direction = Vector((1.15, -1.4, 2.5)).normalized()
    bpy.ops.object.camera_add(location=center + direction * scale * 2.4)
    camera = bpy.context.object
    camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max(extent.length * 1.08, 0.1)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.bl_idname == 'ShaderNodeBackground')
    background.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1)
    background.inputs['Strength'].default_value = 0.5
    for direction, power, size in [((2,-3,5),1200,1.8), ((-4,-1,3),800,2.5), ((1,4,4),950,2.0)]:
        bpy.ops.object.light_add(type='AREA', location=center + Vector(direction).normalized() * scale * 1.5)
        light = bpy.context.object
        light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()
        light.data.energy = power * 0.4 * (scale / 5.0) ** 2
        light.data.shape = 'DISK'
        light.data.size = size * scale
    scene.view_settings.view_transform = 'Standard'
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = str(args.output / f'{scene_id}.png')
    bpy.ops.render.render(write_still=True)
    print(f'RENDERED {scene_id}: {scene.render.filepath}', flush=True)


def main():
    args = arguments()
    if args.resolution <= 0 or args.samples <= 0 or args.box_radius <= 0:
        raise ValueError('Resolution, samples and box radius must be positive')
    with args.annotations.open('rb') as source:
        infos = pickle.load(source)
    by_scene = {Path(item['lidar_points']['lidar_path']).stem: item for item in infos['data_list']}
    scenes = args.scene or sorted(p.name for p in args.mesh_dir.iterdir()
                                   if (p / f'{p.name}_vh_clean_2.ply').is_file())
    if not scenes:
        raise RuntimeError(f'No downloaded meshes found in {args.mesh_dir}')
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for scene_id in scenes:
        try:
            render(scene_id, by_scene[scene_id], args)
        except Exception as error:
            failed.append(scene_id)
            print(f'FAILED {scene_id}: {error}', file=sys.stderr, flush=True)
    if failed:
        raise RuntimeError(f'Failed scenes: {", ".join(failed)}')


if __name__ == '__main__':
    main()
