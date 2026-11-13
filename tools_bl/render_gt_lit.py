import bpy, sys
from pathlib import Path
from mathutils import Vector

root = Path(__file__).resolve().parent
out = root / 'renders_lit_color'
out.mkdir(exist_ok=True)
args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
paths = sorted(root.glob('scene*/mesh_with_gt_bbox.ply'))
if args:
    paths = [p for p in paths if p.parent.name in args]
for ply in paths:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.ply_import(filepath=str(ply.parent / 'aligned_mesh.ply'))
    obj = bpy.context.selected_objects[0]
    color = obj.data.color_attributes.active_color
    if color is None:
        raise RuntimeError(f'No vertex color in {ply}')
    mat = bpy.data.materials.new('Vertex colors')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    vertex = nodes.new('ShaderNodeVertexColor')
    vertex.layer_name = color.name
    mat.node_tree.links.new(vertex.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.85
    mat.node_tree.links.new(vertex.outputs['Color'], bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = 0.25
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    bpy.ops.wm.ply_import(filepath=str(ply.parent / 'gt_bbox.ply'))
    boxes = bpy.context.selected_objects[0]
    red = bpy.data.materials.new('GT box red')
    red.use_nodes = True
    nodes = red.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    emission.inputs['Strength'].default_value = 1.0
    red.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    boxes.data.materials.clear()
    boxes.data.materials.append(red)
    points = [o.matrix_world @ Vector(c) for o in (obj, boxes) for c in o.bound_box]
    lo = Vector([min(p[k] for p in points) for k in range(3)])
    hi = Vector([max(p[k] for p in points) for k in range(3)])
    center = (lo + hi) / 2
    extent = hi - lo
    scale = max(extent)
    direction = Vector((1.15, -1.4, 1.6)).normalized()
    bpy.ops.object.camera_add(location=center + direction * scale * 2.4)
    camera = bpy.context.object
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max(extent.length * 1.08, 0.1)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.25
    for direction, power, size in [((2,-3,5), 1200, 1.8), ((-4,-1,3), 800, 2.5), ((1,4,4), 950, 2.0)]:
        bpy.ops.object.light_add(type='AREA', location=center + Vector(direction).normalized()*scale*1.5)
        light = bpy.context.object
        light.rotation_euler = (center-light.location).to_track_quat('-Z','Y').to_euler()
        light.data.energy = power * 0.25 * (scale / 5.0)**2
        light.data.shape = 'DISK'
        light.data.size = size * scale
    scene.view_settings.view_transform = 'Standard'
    scene.render.resolution_x = 3200
    scene.render.resolution_y = 3200
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = str(out / (ply.parent.name + '.png'))
    bpy.ops.render.render(write_still=True)
    print('RENDERED', scene.render.filepath, flush=True)
