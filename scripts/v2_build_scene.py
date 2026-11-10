"""M4 场景复刻 v2：三点接触标定真实俯拍地面倾角 + 完整着装穿鞋角色入场渲染。

解决核心缺陷（对应 OPTIMIZATION_SPEC.md M4 / RC-6）：
1. 由左鞋底、右鞋底与臀部坐骨最低点精确求解实拍地面平面（法向约向相机前上方倾斜 ~16°），
   将整个卧室几何挂在地面局部坐标系下，使双脚与臀部同时严丝合缝贴合地毯，床铺自然退居身后作靠背，
   彻底消除旧版 S8 中人物悬空 17cm、“坐在床顶垂腿”的透视错位。
2. 接入完整 v2 角色（LOD0 皮肤/衣物图集 + 体积外推 + 骨骼对齐双丸子头 + 高帮帆布鞋）。
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import char_lib_v2 as CL2


def tex_mat(name, path, rough=0.88, spec=0.22, tint=(1.0, 1.0, 1.0)):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    img = bpy.data.images.load(str(Path(path).resolve()))
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
              scale=3.2, distortion=6.0, rough=0.60, band_dir="X"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 0.18) if band_dir in ("X", "Y") else (1.0, 1.0, 1.0)
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
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.26
    return m


def make_room_box(name, parent, x0, x1, y0, y1, z0, z1, mat, tile=0.5):
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
    ob.parent = parent
    me.materials.append(mat)
    return ob


def displace_local_z(ob, amp=0.06, scale=2.6):
    for v in ob.data.vertices:
        p = np.array(v.co[:], np.float32)
        n = noise.noise(Vector(p) * scale) + 0.5 * noise.noise(Vector(p) * scale * 2.2 + Vector((7, 3, 11)))
        v.co = Vector((p[0], p[1], p[2] + n * amp))


def add_plush(name, parent, loc, r, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=18, radius=r, location=(0, 0, 0))
    body = bpy.context.object
    body.name = name
    body.scale = (1.0, 0.88, 1.12)
    bpy.ops.object.transform_apply(scale=True)
    body.parent = parent
    body.location = loc
    body.data.materials.append(mat)
    for p in body.data.polygons:
        p.use_smooth = True

    hz = loc[2] + r * 1.12
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=r * 0.72, location=(0, 0, 0))
    head = bpy.context.object
    head.name = f"{name}_head"
    head.parent = parent
    head.location = (loc[0], loc[1] - r * 0.14, hz)
    head.data.materials.append(mat)
    for p in head.data.polygons:
        p.use_smooth = True

    for sx in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, radius=r * 0.28, location=(0, 0, 0))
        ear = bpy.context.object
        ear.name = f"{name}_ear_{sx}"
        ear.parent = parent
        ear.location = (loc[0] + sx * r * 0.58, loc[1] - r * 0.10, hz + r * 0.56)
        ear.data.materials.append(mat)
        for p in ear.data.polygons:
            p.use_smooth = True


def compute_contact_floor_frame(info: dict, shoe_objs: list):
    """由左右鞋底最低点与臀部坐骨最低点精确求解地面坐标系（原点在臀部正下方的地毯顶面）。"""
    B = info["B"]
    rest = info["rest_v"]

    # 提取左右鞋底网格的最低接触点（在相机竖直向近似下取最低 3% 均值）
    def bottom_of_objs(prefix):
        pts = []
        for ob in shoe_objs:
            if ob.name.startswith(prefix):
                pts.extend([np.array(ob.matrix_world @ v.co, dtype=np.float64) for v in ob.data.vertices])
        arr = np.asarray(pts)
        z_cut = np.percentile(arr[:, 2], 5)
        return arr[arr[:, 2] <= z_cut].mean(axis=0)

    p_L = bottom_of_objs("sole_bot_L")
    p_R = bottom_of_objs("sole_bot_R")

    # 臀部坐骨接触点：T-pose 中骨盆/臀底区域 (ry ∈ [79, 88], rz < 2) 在当前姿态下的最低簇
    hip_sel = (rest[:, 1] >= 79.0) & (rest[:, 1] <= 88.0) & (rest[:, 2] < 2.0) & (np.abs(rest[:, 0]) < 13.0)
    hip_pts = B[hip_sel].astype(np.float64)
    z_hip_cut = np.percentile(hip_pts[:, 2], 8)
    p_H = hip_pts[hip_pts[:, 2] <= z_hip_cut].mean(axis=0)

    # 通过 p_L, p_R, p_H 三点的精确地面法向量（朝上 +Z）
    v1 = p_L - p_R
    v2 = p_H - 0.5 * (p_L + p_R)
    n_floor = np.cross(v1, v2)
    if n_floor[2] < 0:
        n_floor = -n_floor
    n_floor /= np.linalg.norm(n_floor) + 1e-9

    # 房间横向轴与世界 +X 对齐（保持后墙木板竖直与原图一致），正交化出房间进深轴 +Y_room
    e_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e_x = e_x - n_floor * np.dot(e_x, n_floor)
    e_x /= np.linalg.norm(e_x) + 1e-9
    e_y = np.cross(n_floor, e_x)
    e_y /= np.linalg.norm(e_y) + 1e-9

    pitch_deg = float(np.degrees(np.arccos(np.clip(n_floor[2], -1.0, 1.0))))
    print(f"[floor] 三点接触平面拟合完成: p_L={p_L.round(3).tolist()} p_R={p_R.round(3).tolist()} "
          f"p_H={p_H.round(3).tolist()} -> 俯拍倾角={pitch_deg:.2f}°")

    origin = p_H.copy()  # 地毯上表面原点位于臀部接触点
    # 计算全身顶点在房间局部坐标系下的最深后背坐标 y_back_max，确保床沿紧贴后背且 0 穿模
    basis = np.column_stack([e_x, e_y, n_floor])
    b_local = (B.astype(np.float64) - origin[None, :]) @ basis
    y_back_max = float(np.percentile(b_local[:, 1], 99.8))
    print(f"[floor] 角色局部纵深范围 y_local=[{b_local[:,1].min():.3f}, {y_back_max:.3f}]m")

    mat4 = Matrix((
        (e_x[0], e_y[0], n_floor[0], origin[0]),
        (e_x[1], e_y[1], n_floor[1], origin[1]),
        (e_x[2], e_y[2], n_floor[2], origin[2]),
        (0.0,    0.0,    0.0,        1.0),
    ))
    return mat4, pitch_deg, y_back_max, p_L, p_R, p_H


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mesh_npz = argv[0] if len(argv) > 0 else r"work\q004-base\v2\mesh_lod0_v2.npz"
    tex_png = argv[1] if len(argv) > 1 else r"work\q004-base\v2\tex\char_atlas_v2.png"
    tex_dir = (Path(argv[2]) if len(argv) > 2 else Path(r"work\q004-base\scene\tex")).resolve()
    out_dir = (Path(argv[3]) if len(argv) > 3 else Path(r"work\q004-base\renders\OPT_SCENE")).resolve()
    res_y = int(argv[4]) if len(argv) > 4 else 840
    samples = int(argv[5]) if len(argv) > 5 else 96
    out_dir.mkdir(parents=True, exist_ok=True)

    CL2.clear_scene()
    info_peek = np.load(mesh_npz, allow_pickle=True)
    w_orig, h_orig = int(info_peek["orig_img_size"][0]), int(info_peek["orig_img_size"][1])
    res_x = int(round(res_y * w_orig / h_orig))
    sc = CL2.setup_render(samples=samples, res_x=res_x, res_y=res_y, transparent=False, view="Standard")

    info = CL2.build_body_v2(mesh_npz, tex_png, use_inflate=True)
    CL2.build_hair_v2(info)
    shoe_objs = CL2.build_shoes_v2(info)

    # 求解真实三点接触倾斜地面坐标系
    room_mat, _, y_back_max, p_L, p_R, p_H = compute_contact_floor_frame(info, shoe_objs)
    room_root = bpy.data.objects.new("room_root", None)
    sc.collection.objects.link(room_root)
    room_root.matrix_world = room_mat

    # 材质构建
    m_wood_floor = proc_wood("wood_floor", c1=(0.20, 0.12, 0.070), c2=(0.075, 0.042, 0.025),
                             scale=6.0, distortion=2.0, band_dir="X", rough=0.48)
    m_wood_wall = proc_wood("wood_wall", c1=(0.088, 0.055, 0.034), c2=(0.022, 0.012, 0.007),
                            scale=18.0, distortion=0.8, band_dir="X", rough=0.60)
    m_carpet = tex_mat("carpet", tex_dir / "carpet.png", rough=0.95, spec=0.08)
    m_plaid = tex_mat("plaid", tex_dir / "plaid.png", rough=0.90, spec=0.14)
    m_duvet = tex_mat("duvet", tex_dir / "duvet.png", rough=0.95, spec=0.10)

    # 在 room_root 局部坐标构建房间：z_local = 0 即地毯顶面（臀部与双鞋底精确贴地！）
    rug_thick = 0.011
    bed_y0 = max(0.35, y_back_max + 0.022)
    bed_y1 = bed_y0 + 1.32

    make_room_box("floor", room_root, -6.0, 6.0, -6.0, 6.0, -rug_thick - 0.03, -rug_thick, m_wood_floor, tile=0.55)
    make_room_box("rug", room_root, -1.05, 1.25, -1.45, bed_y0 + 0.08, -rug_thick, 0.0, m_carpet, tile=0.76)

    # 后方与左侧深色竖纹护墙木板
    make_room_box("wall_back", room_root, -6.0, 6.0, bed_y1, bed_y1 + 0.08, -rug_thick, 3.0, m_wood_wall, tile=0.95)
    make_room_box("wall_left", room_root, -2.45, -2.37, -2.8, bed_y1 + 0.08, -rug_thick, 3.0, m_wood_wall, tile=0.95)

    # 床铺精确位于角色后背之后 y_local ∈ [bed_y0, bed_y1]：0 穿模且视觉上自然背靠床沿
    make_room_box("bed_L", room_root, -1.32, 0.16, bed_y0, bed_y1, -rug_thick, 0.27, m_duvet, tile=0.36)
    make_room_box("bed_R", room_root, 0.16, 1.88, bed_y0, bed_y1, -rug_thick, 0.26, m_plaid, tile=0.40)

    duv = make_room_box("duvet_mound", room_root, -1.32, 0.16, bed_y0 + 0.01, bed_y1, 0.27, 0.45, m_duvet, tile=0.36)
    bpy.context.view_layer.objects.active = duv
    sm = duv.modifiers.new("sub", "SUBSURF")
    sm.subdivision_type = "SIMPLE"
    sm.levels = 4
    bpy.ops.object.modifier_apply(modifier="sub")
    displace_local_z(duv, amp=0.065, scale=2.4)
    for p in duv.data.polygons:
        p.use_smooth = True

    pl = make_room_box("plaid_mound", room_root, 0.16, 1.88, bed_y0 + 0.01, bed_y1, 0.26, 0.43, m_plaid, tile=0.40)
    bpy.context.view_layer.objects.active = pl
    sm2 = pl.modifiers.new("sub", "SUBSURF")
    sm2.subdivision_type = "SIMPLE"
    sm2.levels = 4
    bpy.ops.object.modifier_apply(modifier="sub")
    displace_local_z(pl, amp=0.058, scale=2.7)
    for p in pl.data.polygons:
        p.use_smooth = True

    # 毛绒玩偶（放置在床面局部高度 z_local ≈ 0.46..0.55）
    m_teddy = CL2.make_simple_mat("teddy_pink", (0.64, 0.44, 0.46), rough=0.95, spec=0.08, sheen=0.55)
    m_white = CL2.make_simple_mat("plush_white", (0.74, 0.71, 0.65), rough=0.95, spec=0.08, sheen=0.55)
    m_blue = CL2.make_simple_mat("plush_blue", (0.28, 0.35, 0.43), rough=0.95, spec=0.08, sheen=0.55)
    add_plush("teddy_pink", room_root, (-0.68, bed_y0 + 0.38, 0.48), 0.115, m_teddy)
    add_plush("plush_white", room_root, (0.04, bed_y0 + 0.80, 0.54), 0.088, m_white)
    add_plush("plush_blue", room_root, (0.66, bed_y0 + 0.36, 0.45), 0.092, m_blue)

    # 相机与机顶闪光灯精确复原（从 params 读取真实 cam_t 与焦距换算镜头）
    cam_t = info["cam_t"]
    focal_px = info["focal"]
    cam_pos = Vector((-float(cam_t[0]), -float(cam_t[2]), float(cam_t[1])))
    lens_mm = 36.0 * focal_px / float(w_orig)

    w_world = bpy.data.worlds.new("RoomWorld")
    w_world.use_nodes = True
    w_world.node_tree.nodes["Background"].inputs[0].default_value = (0.14, 0.14, 0.16, 1.0)
    w_world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    sc.world = w_world

    ld_flash = bpy.data.lights.new("on_camera_flash", "POINT")
    ld_flash.energy = 86.0
    ld_flash.shadow_soft_size = 0.065
    ld_flash.color = (1.0, 0.96, 0.92)
    ob_flash = bpy.data.objects.new("on_camera_flash", ld_flash)
    ob_flash.location = cam_pos + Vector((0.0, 0.06, 0.09))
    ob_flash.visible_camera = False
    sc.collection.objects.link(ob_flash)

    ld_fill = bpy.data.lights.new("warm_bounce", "AREA")
    ld_fill.energy = 45.0
    ld_fill.size = 2.2
    ld_fill.color = (1.0, 0.92, 0.83)
    ob_fill = bpy.data.objects.new("warm_bounce", ld_fill)
    ob_fill.location = cam_pos + Vector((0.0, 1.8, 1.25))
    ob_fill.rotation_euler = (Vector(info["head_center"].tolist()) - ob_fill.location).to_track_quat("-Z", "Y").to_euler()
    ob_fill.visible_camera = False
    sc.collection.objects.link(ob_fill)

    cam = CL2.add_camera(lens=lens_mm, sensor_width=36.0)
    cam.location = cam_pos
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    sc.render.filepath = str(out_dir / "scene_match.png")
    bpy.ops.render.render(write_still=True)
    print(f"[scene_v2] wrote {out_dir / 'scene_match.png'} (lens={lens_mm:.2f}mm)")

    # 斜侧面接触检视机位：对准臀部与双脚中心，直观验证双脚鞋底+臀部贴合地毯且背部靠床零穿模
    mid_contact = 0.5 * (p_H + 0.5 * (p_L + p_R))
    side_target = Vector((float(mid_contact[0]), float(mid_contact[1]) + 0.08, float(mid_contact[2]) + 0.30))
    cam.data.lens = 35.0
    cam.location = side_target + Vector((1.35, -1.18, 0.32))
    cam.rotation_euler = (side_target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(out_dir / "scene_side.png")
    bpy.ops.render.render(write_still=True)
    print(f"[scene_v2] wrote {out_dir / 'scene_side.png'}")


if __name__ == "__main__":
    main()
