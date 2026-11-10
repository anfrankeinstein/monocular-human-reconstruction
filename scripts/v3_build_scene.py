"""v3 卧室实景复刻：三点接触标定真实俯拍地面倾角 + 独立物理布料壳体角色入场渲染。

在 v2 基础上升级：
1. 角色采用解耦纯肤色 LOD0 身体 + Blender CLOTH 物理模拟后的 3D 衬衫壳与牛仔短裤壳（含古巴翻领、纽扣、裤耳、毛边裤口）。
2. 臀部接地点基于实体牛仔短裤壳体外底面计算，后背床沿安全间隙同时联合身体与宽松衬衫后摆顶点计算，确保松弛衣摆靠床零穿模。
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import char_lib_v2 as CL2
import char_lib_v3 as CL3
import hair_lib_v3 as HL3
import shoe_lib_v3 as SL3
import v2_build_scene as S2


def compute_contact_floor_frame_v3(info: dict, g_info: dict, shoe_info: dict):
    """由左右鞋底接地面最低点与实体牛仔短裤臀底最低点精确求解地面坐标系，并联合宽松衬衫后背防穿模。"""
    def bottom_of_pts(arr: np.ndarray):
        z_cut = np.percentile(arr[:, 2], 8)
        return arr[arr[:, 2] <= z_cut].mean(axis=0)

    p_L = bottom_of_pts(shoe_info["sole_bottom_L"])
    p_R = bottom_of_pts(shoe_info["sole_bottom_R"])

    # 臀部接触点：取牛仔短裤壳体臀底区域最低簇（算上短裤半厚 1.4mm），使短裤外表面严丝合缝贴合地毯
    dn_v = g_info["shorts_v_bl"].astype(np.float64)
    dn_r = g_info["shorts_rest"]
    hip_sel = (dn_r[:, 1] >= 79.0) & (dn_r[:, 1] <= 88.0) & (dn_r[:, 2] < 2.0) & (np.abs(dn_r[:, 0]) < 13.0)
    hip_pts = dn_v[hip_sel]
    z_hip_cut = np.percentile(hip_pts[:, 2], 8)
    p_H = hip_pts[hip_pts[:, 2] <= z_hip_cut].mean(axis=0)

    v1 = p_L - p_R
    v2 = p_H - 0.5 * (p_L + p_R)
    n_floor = np.cross(v1, v2)
    if n_floor[2] < 0:
        n_floor = -n_floor
    n_floor /= np.linalg.norm(n_floor) + 1e-9

    # 减去牛仔短裤 SOLIDIFY 半厚 (1.4mm)，使实体布料外表面贴地零穿插
    p_H = p_H - n_floor * 0.0014

    e_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e_x = e_x - n_floor * np.dot(e_x, n_floor)
    e_x /= np.linalg.norm(e_x) + 1e-9
    e_y = np.cross(n_floor, e_x)
    e_y /= np.linalg.norm(e_y) + 1e-9

    pitch_deg = float(np.degrees(np.arccos(np.clip(n_floor[2], -1.0, 1.0))))
    print(f"[floor_v3] 三点接触平面拟合完成: p_L={p_L.round(3).tolist()} p_R={p_R.round(3).tolist()} "
          f"p_H={p_H.round(3).tolist()} -> 俯拍倾角={pitch_deg:.2f}°")

    origin = p_H.copy()
    basis = np.column_stack([e_x, e_y, n_floor])
    all_verts = np.vstack([info["B"].astype(np.float64), g_info["garment_verts_bl"].astype(np.float64)])
    c_local = (all_verts - origin[None, :]) @ basis
    y_back_max = float(np.percentile(c_local[:, 1], 99.8))
    print(f"[floor_v3] 角色+物理服装局部纵深范围 y_local=[{c_local[:, 1].min():.3f}, {y_back_max:.3f}]m")

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
    sim_npz = argv[1] if len(argv) > 1 else r"work\q004-base\v3\garments_sim_v3.npz"
    char_tex_dir = (Path(argv[2]) if len(argv) > 2 else Path(r"work\q004-base\v3\tex")).resolve()
    scene_tex_dir = (Path(argv[3]) if len(argv) > 3 else Path(r"work\q004-base\scene\tex")).resolve()
    out_dir = (Path(argv[4]) if len(argv) > 4 else Path(r"work\q004-base\renders\OPT_SCENE_V3")).resolve()
    res_y = int(argv[5]) if len(argv) > 5 else 840
    samples = int(argv[6]) if len(argv) > 6 else 96
    out_dir.mkdir(parents=True, exist_ok=True)

    CL2.clear_scene()
    info_peek = np.load(mesh_npz, allow_pickle=True)
    w_orig, h_orig = int(info_peek["orig_img_size"][0]), int(info_peek["orig_img_size"][1])
    res_x = int(round(res_y * w_orig / h_orig))
    sc = CL2.setup_render(samples=samples, res_x=res_x, res_y=res_y, transparent=False, view="Standard")

    info = CL3.build_pure_skin_body_v3(mesh_npz, str(char_tex_dir / "skin_body_v3.png"))
    g_info = CL3.build_simulated_garments_v3(
        sim_npz,
        str(char_tex_dir / "shirt_atlas_v3.png"),
        str(char_tex_dir / "shorts_atlas_v3.png"),
    )
    HL3.build_hair_v3(info)
    shoe_info = SL3.build_shoes_v3(mesh_npz, char_tex_dir)

    room_mat, _, y_back_max, p_L, p_R, p_H = compute_contact_floor_frame_v3(info, g_info, shoe_info)
    room_root = bpy.data.objects.new("room_root", None)
    sc.collection.objects.link(room_root)
    room_root.matrix_world = room_mat

    m_wood_floor = S2.proc_wood("wood_floor", c1=(0.20, 0.12, 0.070), c2=(0.075, 0.042, 0.025),
                                scale=6.0, distortion=2.0, band_dir="X", rough=0.48)
    m_wood_wall = S2.proc_wood("wood_wall", c1=(0.088, 0.055, 0.034), c2=(0.022, 0.012, 0.007),
                               scale=18.0, distortion=0.8, band_dir="X", rough=0.60)
    m_carpet = S2.tex_mat("carpet", scene_tex_dir / "carpet.png", rough=0.95, spec=0.08)
    m_plaid = S2.tex_mat("plaid", scene_tex_dir / "plaid.png", rough=0.90, spec=0.14)
    m_duvet = S2.tex_mat("duvet", scene_tex_dir / "duvet.png", rough=0.95, spec=0.10)

    rug_thick = 0.011
    bed_y0 = max(0.35, y_back_max + 0.024)
    bed_y1 = bed_y0 + 1.32

    S2.make_room_box("floor", room_root, -6.0, 6.0, -6.0, 6.0, -rug_thick - 0.03, -rug_thick, m_wood_floor, tile=0.55)
    S2.make_room_box("rug", room_root, -1.05, 1.25, -1.45, bed_y0 + 0.08, -rug_thick, 0.0, m_carpet, tile=0.76)

    S2.make_room_box("wall_back", room_root, -6.0, 6.0, bed_y1, bed_y1 + 0.08, -rug_thick, 3.0, m_wood_wall, tile=0.95)
    S2.make_room_box("wall_left", room_root, -2.45, -2.37, -2.8, bed_y1 + 0.08, -rug_thick, 3.0, m_wood_wall, tile=0.95)

    S2.make_room_box("bed_L", room_root, -1.32, 0.16, bed_y0, bed_y1, -rug_thick, 0.27, m_duvet, tile=0.36)
    S2.make_room_box("bed_R", room_root, 0.16, 1.88, bed_y0, bed_y1, -rug_thick, 0.26, m_plaid, tile=0.40)

    duv = S2.make_room_box("duvet_mound", room_root, -1.32, 0.16, bed_y0 + 0.01, bed_y1, 0.27, 0.45, m_duvet, tile=0.36)
    bpy.context.view_layer.objects.active = duv
    sm = duv.modifiers.new("sub", "SUBSURF")
    sm.subdivision_type = "SIMPLE"
    sm.levels = 4
    bpy.ops.object.modifier_apply(modifier="sub")
    S2.displace_local_z(duv, amp=0.065, scale=2.4)
    for p in duv.data.polygons:
        p.use_smooth = True

    pl = S2.make_room_box("plaid_mound", room_root, 0.16, 1.88, bed_y0 + 0.01, bed_y1, 0.26, 0.43, m_plaid, tile=0.40)
    bpy.context.view_layer.objects.active = pl
    sm2 = pl.modifiers.new("sub", "SUBSURF")
    sm2.subdivision_type = "SIMPLE"
    sm2.levels = 4
    bpy.ops.object.modifier_apply(modifier="sub")
    S2.displace_local_z(pl, amp=0.058, scale=2.7)
    for p in pl.data.polygons:
        p.use_smooth = True

    m_teddy = CL2.make_simple_mat("teddy_pink", (0.64, 0.44, 0.46), rough=0.95, spec=0.08, sheen=0.55)
    m_white = CL2.make_simple_mat("plush_white", (0.74, 0.71, 0.65), rough=0.95, spec=0.08, sheen=0.55)
    m_blue = CL2.make_simple_mat("plush_blue", (0.28, 0.35, 0.43), rough=0.95, spec=0.08, sheen=0.55)
    S2.add_plush("teddy_pink", room_root, (-0.68, bed_y0 + 0.38, 0.48), 0.115, m_teddy)
    S2.add_plush("plush_white", room_root, (0.04, bed_y0 + 0.80, 0.54), 0.088, m_white)
    S2.add_plush("plush_blue", room_root, (0.66, bed_y0 + 0.36, 0.45), 0.092, m_blue)

    cam_t = info["cam_t"]
    focal_px = info["focal"]
    cam_pos = Vector((-float(cam_t[0]), -float(cam_t[2]), float(cam_t[1])))
    lens_mm = 36.0 * focal_px / float(w_orig)

    w_world = bpy.data.worlds.new("RoomWorldV3")
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
    print(f"[scene_v3] wrote {out_dir / 'scene_match.png'} (lens={lens_mm:.2f}mm)")

    mid_contact = 0.5 * (p_H + 0.5 * (p_L + p_R))
    side_target = Vector((float(mid_contact[0]), float(mid_contact[1]) + 0.08, float(mid_contact[2]) + 0.30))
    cam.data.lens = 35.0
    cam.location = side_target + Vector((1.35, -1.18, 0.32))
    cam.rotation_euler = (side_target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(out_dir / "scene_side.png")
    bpy.ops.render.render(write_still=True)
    print(f"[scene_v3] wrote {out_dir / 'scene_side.png'}")


if __name__ == "__main__":
    main()
