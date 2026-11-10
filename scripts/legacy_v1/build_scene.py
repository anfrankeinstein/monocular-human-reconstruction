"""S8 场景：木墙 + 木地板 + 地毯 + 床（素色被 + 格子毯）+ 玩偶，复刻照片的机位与闪光灯。

用法：
  blender -b -P build_scene.py -- <mesh_uv.npz> <skin_atlas.png> <tex_dir> <out_dir> [res_y]
"""
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

sys.path.insert(0, str(Path(__file__).parent))
import char_lib as CL

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MESH, TEX, TEXDIR, OUT = argv[0], argv[1], Path(argv[2]), Path(argv[3])
RES_Y = int(argv[4]) if len(argv) > 4 else 720
OUT.mkdir(parents=True, exist_ok=True)

FLOOR_Z = 0.668
RUG_Z = 0.679

# ------------------------------------------------------------------ 场景骨架
CL.clear_scene()
sc = CL.setup_render(samples=64, res=RES_Y, transparent=False, view="Standard")
sc.render.resolution_x = int(RES_Y * 1402 / 1122)

body, B, F = CL.build_body(MESH, TEX)
head, z_top, crn = CL.head_info(B)
CL.build_hair(body, B, CL.hair_material())


# ------------------------------------------------------------------ 材质
def tex_mat(name, path, rough=0.85, spec=0.3, tint=(1.0, 1.0, 1.0)):
    """贴图材质。UV 由 box() 按世界坐标做盒式投影，尺寸以米为单位。"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    img = bpy.data.images.load(str(path))
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = img
    tx.interpolation = "Smart"
    tx.extension = "REPEAT"
    if tint != (1.0, 1.0, 1.0):
        mul = nt.nodes.new("ShaderNodeMixRGB")
        mul.blend_type = "MULTIPLY"
        mul.inputs["Fac"].default_value = 1.0
        mul.inputs["Color2"].default_value = (*tint, 1.0)
        nt.links.new(tx.outputs["Color"], mul.inputs["Color1"])
        nt.links.new(mul.outputs["Color"], b.inputs["Base Color"])
    else:
        nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    return m


def proc_wood(name, c1=(0.075, 0.048, 0.030), c2=(0.020, 0.011, 0.007),
              scale=3.2, distortion=6.0, rough=0.62, band_dir="Z"):
    """程序化木纹：Wave 条纹 + 噪声扰动。照片木纹只有 195×130，平铺到整面墙会糊成色块，改用程序化。"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    if band_dir == "X":
        mp.inputs["Scale"].default_value = (1.0, 1.0, 0.18)   # 竖向木板
    elif band_dir == "Y":
        mp.inputs["Scale"].default_value = (1.0, 1.0, 0.18)
    else:
        mp.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale * 2.4
    nz.inputs["Detail"].default_value = 8.0
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = band_dir
    wave.inputs["Scale"].default_value = scale
    wave.inputs["Distortion"].default_value = distortion
    wave.inputs["Detail"].default_value = 3.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c2, 1.0)
    ramp.color_ramp.elements[1].color = (*c1, 1.0)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.28
    return m


def flat_mat(name, rgb, rough=0.9, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    if sheen and "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    return m


M_WOOD_FLOOR = proc_wood("wood_floor", c1=(0.21, 0.13, 0.075), c2=(0.080, 0.045, 0.028),
                         scale=6.0, distortion=2.0, band_dir="X", rough=0.5)
M_WOOD_WALL = proc_wood("wood_wall", c1=(0.088, 0.058, 0.038), c2=(0.026, 0.015, 0.009),
                        scale=9.0, distortion=1.5, band_dir="X")
M_CARPET = tex_mat("carpet", TEXDIR / "carpet.png", rough=0.95, spec=0.1)
M_PLAID = tex_mat("plaid", TEXDIR / "plaid.png", rough=0.9, spec=0.15)
M_DUVET = tex_mat("duvet", TEXDIR / "duvet.png", rough=0.95, spec=0.1)


# ------------------------------------------------------------------ 几何
def box(name, x0, x1, y0, y1, z0, z1, mat, tile=0.5):
    """按世界坐标做盒式 UV 投影：每面挑两个世界轴当 u/v，tile 为贴图覆盖的米数。"""
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], quads)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    for pi, q in enumerate(quads):
        for k, vi in enumerate(q):
            x, y, z = verts[vi]
            if pi in (0, 1):
                u, v = x, y
            elif pi in (2, 4):
                u, v = x, z
            else:
                u, v = y, z
            uvl.data[pi * 4 + k].uv = (u / tile, v / tile)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat)
    return ob


def displace(ob, amp=0.02, scale=6.0):
    for v in ob.data.vertices:
        p = np.array(v.co[:], np.float32)
        n = 0.0
        n += 1.00 * noise.noise(Vector(p) * scale)
        n += 0.50 * noise.noise(Vector(p) * scale * 2.3 + Vector((11, 3, 7)))
        v.co = Vector((p + np.array([0, 0, 1.0]) * n * amp).tolist())


# 地板 / 墙
box("floor", -6, 6, -6, 6, FLOOR_Z - 0.02, FLOOR_Z, M_WOOD_FLOOR, tile=0.55)
box("wall_back", -6, 6, 1.62, 1.70, FLOOR_Z, 3.2, M_WOOD_WALL, tile=0.95)
box("wall_left", -2.60, -2.52, -2.5, 1.70, FLOOR_Z, 3.2, M_WOOD_WALL, tile=0.95)

# 地毯
rug = box("rug", -1.20, 1.05, -1.30, 0.45, FLOOR_Z, RUG_Z, M_CARPET, tile=0.78)

# 床：素色被在左，格子毯在右
box("bed_L", -1.35, 0.16, 0.28, 1.60, FLOOR_Z, 0.88, M_DUVET, tile=0.34)
box("bed_R", 0.16, 1.90, 0.28, 1.60, FLOOR_Z, 0.86, M_PLAID, tile=0.40)
duv = box("duvet_mound", -1.35, 0.16, 0.22, 1.60, 0.88, 1.02, M_DUVET, tile=0.34)
bpy.context.view_layer.objects.active = duv
sm = duv.modifiers.new("sub", "SUBSURF"); sm.subdivision_type = "SIMPLE"; sm.levels = 4
bpy.ops.object.modifier_apply(modifier="sub")
displace(duv, amp=0.085, scale=2.4)
pl = box("plaid_mound", 0.16, 1.90, 0.22, 1.60, 0.86, 1.00, M_PLAID, tile=0.40)
bpy.context.view_layer.objects.active = pl
sm = pl.modifiers.new("sub", "SUBSURF"); sm.subdivision_type = "SIMPLE"; sm.levels = 4
bpy.ops.object.modifier_apply(modifier="sub")
displace(pl, amp=0.070, scale=2.8)

# 玩偶
M_TEDDY = flat_mat("teddy", (0.62, 0.42, 0.45), rough=0.95, sheen=0.6)
M_WHITE = flat_mat("plush_white", (0.72, 0.70, 0.66), rough=0.95, sheen=0.6)
M_BLUE = flat_mat("plush_blue", (0.30, 0.36, 0.42), rough=0.95, sheen=0.6)


def plush(name, loc, r, mat, ears=True):
    objs = []
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=r, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = (1.0, 0.9, 1.15)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = True
    objs.append(o)
    hz = loc[2] + r * 1.15
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=r * 0.72,
                                         location=(loc[0], loc[1] - r * 0.15, hz))
    h = bpy.context.object
    h.name = name + "_head"
    h.data.materials.append(mat)
    for p in h.data.polygons:
        p.use_smooth = True
    objs.append(h)
    if ears:
        for sx in (-1, 1):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, radius=r * 0.30,
                                                 location=(loc[0] + sx * r * 0.60,
                                                           loc[1] - r * 0.10,
                                                           hz + r * 0.58))
            e = bpy.context.object
            e.name = f"{name}_ear{sx}"
            e.data.materials.append(mat)
            for p in e.data.polygons:
                p.use_smooth = True
            objs.append(e)
    return objs


plush("teddy_pink", (-0.72, 0.50, 0.98), 0.115, M_TEDDY)
plush("plush_white", (-0.02, 1.02, 1.10), 0.088, M_WHITE)
plush("plush_blue", (0.70, 0.62, 0.96), 0.092, M_BLUE)


# ------------------------------------------------------------------ 灯光：机顶闪光
def flash_look(cam_pos, power=90.0, size=0.06):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.16, 0.17, 0.21, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.55
    sc.world = w

    ld = bpy.data.lights.new("flash", "POINT")
    ld.energy = power
    ld.shadow_soft_size = size
    ld.color = (0.96, 0.98, 1.0)
    o = bpy.data.objects.new("flash", ld)
    o.location = Vector(cam_pos) + Vector((0.0, 0.06, 0.10))
    o.visible_camera = False
    sc.collection.objects.link(o)

    ld2 = bpy.data.lights.new("bounce", "AREA")
    ld2.energy = 55.0
    ld2.size = 2.0
    ld2.color = (1.0, 0.93, 0.85)
    o2 = bpy.data.objects.new("bounce", ld2)
    o2.location = (0.0, -0.30, 2.35)
    o2.rotation_euler = (Vector((0, 0.9, 0.9)) - o2.location).to_track_quat("-Z", "Y").to_euler()
    o2.visible_camera = False
    sc.collection.objects.link(o2)


CAM_POS = Vector((-0.068, -2.363, 1.175))
flash_look(CAM_POS)

cam = CL.add_camera(lens=46.1)
cam.location = CAM_POS
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)

sc.render.filepath = str(OUT / "scene_match.png")
bpy.ops.render.render(write_still=True)
print("[blender] wrote scene_match.png")

# 侧一点机位看空间关系
cam.location = CAM_POS + Vector((0.55, 0.30, 0.10))
cam.rotation_euler = (math.radians(90.0), 0.0, math.radians(-14.0))
sc.render.filepath = str(OUT / "scene_side.png")
bpy.ops.render.render(write_still=True)
print("[blender] wrote scene_side.png")
print("[blender] done")
