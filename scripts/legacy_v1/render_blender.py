"""Blender headless：导入 MHR 网格，三点布光，渲染三视图。"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
obj_path = argv[0]
out_dir = Path(argv[1])
samples = int(argv[2]) if len(argv) > 2 else 96
res = int(argv[3]) if len(argv) > 3 else 1080

out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj_path)

objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
print(f"[blender] imported {len(objs)} mesh objects")
mesh_obj = objs[0]

for o in bpy.context.selected_objects:
    o.select_set(False)
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

co = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
zvals = [p.z for p in co]
if (sum(zvals) / len(zvals)) < 0:
    print("[blender] 轴系为 Y-down，绕 X 轴校正 180 度")
    bpy.ops.transform.rotate(value=math.pi, orient_axis="X")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
xs = [v.x for v in bbox]; ys = [v.y for v in bbox]; zs = [v.z for v in bbox]
center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
height = max(zs) - min(zs)
size = max(max(xs) - min(xs), max(ys) - min(ys), height)
print(f"[blender] bbox size {size:.3f} height {height:.3f} center {tuple(round(c,3) for c in center)}")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.render.resolution_x = res
scene.render.resolution_y = res
scene.render.film_transparent = False

world = bpy.data.worlds.new("W")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
scene.world = world

mat = bpy.data.materials.new("skin")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.78, 0.62, 0.55, 1)
bsdf.inputs["Roughness"].default_value = 0.55
if "Subsurface Weight" in bsdf.inputs:
    bsdf.inputs["Subsurface Weight"].default_value = 0.12
mesh_obj.data.materials.append(mat)

key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
key.data.energy = 400
key.data.size = size * 1.2
key.location = center + Vector((size * 1.2, -size * 1.4, size * 1.1))
key.rotation_euler = (Vector(center) - key.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(key)

fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "AREA"))
fill.data.energy = 120
fill.data.size = size * 1.5
fill.location = center + Vector((-size * 1.3, -size * 1.0, size * 0.3))
fill.rotation_euler = (Vector(center) - fill.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(fill)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def shoot(name, direction):
    d = Vector(direction).normalized()
    cam.location = center + d * size * 1.9
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_dir / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {name}.png")


shoot("front", (0, -1, 0.12))
shoot("side45", (0.75, -0.75, 0.15))
shoot("back", (0, 1, 0.12))
print("[blender] done")
