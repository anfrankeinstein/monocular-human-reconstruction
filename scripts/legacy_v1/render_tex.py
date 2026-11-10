"""Blender：用带 UV 的网格 + 皮肤贴图渲染。"""
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
data_path, tex_path, out_dir = argv[0], argv[1], Path(argv[2])
samples = int(argv[3]) if len(argv) > 3 else 64
res = int(argv[4]) if len(argv) > 4 else 900
exposure = float(argv[5]) if len(argv) > 5 else 0.0
# 光强标定：面积光 500W 在 2.3m 处辐照度约 30，比"1.0"高两个数量级，
# 实测 light_mul=0.10（key 50W / fill 16W）时体表渲染中位色与照片皮肤中位色一致
light_mul = float(argv[6]) if len(argv) > 6 else 0.10
look = argv[7] if len(argv) > 7 else "None"
vt = argv[8] if len(argv) > 8 else "Standard"
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)

d = np.load(data_path)
verts = d["v"].astype(np.float32)
faces = d["faces"].astype(np.int32)
tex = d["tex"].astype(np.float32)
tfaces = d["tfaces"].astype(np.int32)

me = bpy.data.meshes.new("body")
verts_b = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=1)
me.from_pydata(verts_b.tolist(), [], faces.tolist())
me.update()
obj = bpy.data.objects.new("body", me)
bpy.context.scene.collection.objects.link(obj)

uv_layer = me.uv_layers.new(name="UVMap")
loops = me.loops
for poly in me.polygons:
    for k in range(poly.loop_total):
        li = poly.loop_start + k
        vi = loops[li].vertex_index
        tri_t = tfaces[poly.index]
        uv_layer.data[li].uv = (float(tex[tri_t[k], 0]), float(tex[tri_t[k], 1]))
me.update()

# 网格在 UV 岛边界处存在重复顶点，法线被拆开 → 平滑着色出现硬边（踝、肘一圈亮环）。
# 焊接重复顶点并统一法线朝向后重新平滑。
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=0.0005)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
print(f"[blender] 焊接后 顶点 {len(me.vertices)} 面 {len(me.polygons)}")

mat = bpy.data.materials.new("skin")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
img = bpy.data.images.load(tex_path)
tex_node = nt.nodes.new("ShaderNodeTexImage")
tex_node.image = img
tex_node.interpolation = "Smart"
nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
if os.environ.get("STB_EMIT"):
    nt.links.remove(nt.links[0])
    em = nt.nodes.new("ShaderNodeEmission")
    out_node = nt.nodes["Material Output"]
    nt.links.new(tex_node.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out_node.inputs["Surface"])
    print("[blender] 自发光模式（纯贴图预览）")
def envc(k, d):
    return float(os.environ.get(k, d))


bsdf.inputs["Roughness"].default_value = envc("STB_ROUGH", 0.62)
# 次表面：默认半径是 1 米，对 1.2 米高的人体等于全透光 → 必须改成皮肤量级
if "Subsurface Weight" in bsdf.inputs:
    bsdf.inputs["Subsurface Weight"].default_value = envc("STB_SSS", 0.16)
if "Subsurface Radius" in bsdf.inputs:
    bsdf.inputs["Subsurface Radius"].default_value = (0.010, 0.005, 0.0025)
if "Subsurface Scale" in bsdf.inputs:
    bsdf.inputs["Subsurface Scale"].default_value = 1.0
if "Specular IOR Level" in bsdf.inputs:
    bsdf.inputs["Specular IOR Level"].default_value = envc("STB_SPEC", 0.18)
me.materials.append(mat)

for poly in me.polygons:
    poly.use_smooth = True

co = np.array([obj.matrix_world @ v.co for v in me.vertices])
if co[:, 2].mean() < 0:
    print("[blender] 轴系校正 180 度")
    obj.rotation_euler = (math.pi, 0, 0)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    co = np.array([obj.matrix_world @ v.co for v in me.vertices])
print(f"[blender] z range {co[:,2].min():.3f}..{co[:,2].max():.3f}")

mn, mx = co.min(axis=0), co.max(axis=0)
center = Vector(((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2))
size = float(max(mx - mn))

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.render.resolution_x = res
scene.render.resolution_y = res
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = vt
scene.view_settings.exposure = exposure
try:
    scene.view_settings.look = look
except TypeError:
    print(f"[blender] look '{look}' 不可用，保持默认")

world = bpy.data.worlds.new("W")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.08, 0.08, 0.09, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
scene.world = world

key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
key.data.energy = 500 * light_mul
key.data.size = size * 1.3
key.location = center + Vector((size * 0.9, -size * 1.5, size * 0.9))
key.rotation_euler = (Vector(center) - key.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(key)

fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "AREA"))
fill.data.energy = 160 * light_mul
fill.data.size = size * 1.6
fill.location = center + Vector((-size * 1.2, -size * 1.1, size * 0.4))
fill.rotation_euler = (Vector(center) - fill.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(fill)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def shoot(name, direction, dist=1.85, target=None, fsize=None):
    tgt = Vector(target) if target is not None else center
    s = float(fsize) if fsize is not None else size
    dvec = Vector(direction).normalized()
    cam.location = tgt + dvec * s * dist
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_dir / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {name}.png")


# 头部中心：最高 12% 顶点（z 向上）
zmax = co[:, 2].max()
head = co[co[:, 2] > zmax - size * 0.12].mean(axis=0)
print(f"[blender] head center {head.round(3)}")

shoot("tex_front", (0, -1, 0.1))
shoot("tex_3q", (0.75, -1, 0.12), dist=1.8)
shoot("tex_face", (0.05, -1, 0.12), dist=1.55, target=head, fsize=size * 0.34)
print("[blender] done")
