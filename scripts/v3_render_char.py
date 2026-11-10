"""v3 角色多视角与服装特写渲染入口（独立 3D 服装壳体 + Blender CLOTH 物理模拟）。

用法：
  "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" -b -P scripts/v3_render_char.py -- ^
      work/q004-base/v2/mesh_lod0_v2.npz ^
      work/q004-base/v3/garments_sim_v3.npz ^
      work/q004-base/v3/tex ^
      work/q004-base/renders/OPT_CHAR_V3 96 1000
"""
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import char_lib_v2 as CL2
import char_lib_v3 as CL3
import hair_lib_v3 as HL3
import shoe_lib_v3 as SL3


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mesh_npz = argv[0] if len(argv) > 0 else r"work\q004-base\v2\mesh_lod0_v2.npz"
    sim_npz = argv[1] if len(argv) > 1 else r"work\q004-base\v3\garments_sim_v3.npz"
    tex_dir = Path(argv[2]) if len(argv) > 2 else Path(r"work\q004-base\v3\tex")
    out_dir = Path(argv[3]) if len(argv) > 3 else Path(r"work\q004-base\renders\OPT_CHAR_V3")
    samples = int(argv[4]) if len(argv) > 4 else 96
    res = int(argv[5]) if len(argv) > 5 else 1000
    out_dir.mkdir(parents=True, exist_ok=True)

    CL2.clear_scene()
    CL2.setup_render(samples=samples, res_x=res, res_y=res, transparent=True, view="Standard")

    info = CL3.build_pure_skin_body_v3(mesh_npz, str(tex_dir / "skin_body_v3.png"))
    g_info = CL3.build_simulated_garments_v3(
        sim_npz,
        str(tex_dir / "shirt_atlas_v3.png"),
        str(tex_dir / "shorts_atlas_v3.png"),
    )
    HL3.build_hair_v3(info)
    shoe_info = SL3.build_shoes_v3(mesh_npz, tex_dir)

    B = info["B"]
    mn, mx = B.min(axis=0), B.max(axis=0)
    center = Vector(((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0))
    size = float(max(mx - mn))
    head_target = Vector(info["head_center"].tolist())

    # 服装壳体近景聚焦中心（覆盖古巴翻领、宽袖悬空、露脐下摆阴影与高腰牛仔短裤）
    gv = g_info["garment_verts_bl"]
    g_mn, g_mx = gv.min(axis=0), gv.max(axis=0)
    garment_target = Vector((
        float(0.5 * (g_mn[0] + g_mx[0])),
        float(0.5 * (g_mn[1] + g_mx[1])),
        float(0.52 * g_mx[2] + 0.48 * g_mn[2]),
    ))
    garment_fsize = float(max(g_mx - g_mn)) * 0.92

    key_l, fill_l, rim_l = CL2.add_studio_lights(center, size, key=52.0, fill=18.0, rim=18.0)
    cam = CL2.add_camera(lens=50.0)

    # 与 SHOE1 / OPT_CHAR 严格一致的 5 个标准机位
    CL2.shoot(cam, "hair_front", out_dir, center, (0.0, -1.0, 0.10), size)
    CL2.shoot(cam, "hair_3q", out_dir, center, (0.75, -1.0, 0.12), size, dist=1.80)
    CL2.shoot(cam, "hair_back", out_dir, center, (-0.35, 1.0, 0.14), size, dist=1.90)
    CL2.shoot(cam, "hair_side", out_dir, center, (-1.0, -0.35, 0.10), size, dist=1.80)
    head_face_target = head_target + Vector(info["head_up"].tolist()) * 0.024
    CL2.shoot(cam, "hair_face", out_dir, head_face_target, (0.05, -1.0, 0.10), size, dist=1.55, fsize=size * 0.335)

    # 新增服装与写实毛发微距特写机位
    head_crown_target = head_target + Vector(info["head_up"].tolist()) * 0.032
    CL2.shoot(cam, "garment_closeup", out_dir, garment_target, (0.52, -1.0, 0.14), size, dist=1.65, fsize=garment_fsize)
    CL2.shoot(cam, "garment_back_closeup", out_dir, garment_target, (-0.38, 1.0, 0.14), size, dist=1.65, fsize=garment_fsize)
    CL2.shoot(cam, "hair_3q_closeup", out_dir, head_crown_target, (0.62, -1.0, 0.18), size, dist=1.55, fsize=size * 0.345)
    CL2.shoot(cam, "hair_back_closeup", out_dir, head_crown_target, (-0.32, 1.0, 0.20), size, dist=1.55, fsize=size * 0.345)

    # 新增帆布鞋与堆堆白袜独立 PBR 微距特写机位（完整框入圆头包边至后跟橡胶标，避免前掌高光过曝）
    key_l.data.energy = 38.0
    fill_l.data.energy = 22.0
    rim_l.data.energy = 34.0
    fr_r = shoe_info["feet"]["R"]
    med_view_dir = 0.80 * fr_r["med_dir_3d"] + 0.42 * fr_r["e_fwd"] + 0.20 * fr_r["e_shin"]
    CL2.shoot(cam, "shoe_medial_closeup", out_dir, Vector(fr_r["focus_center"].tolist()),
              tuple(med_view_dir.tolist()), size, dist=1.66, fsize=0.305)

    fr_l = shoe_info["feet"]["L"]
    heel_view_dir = -0.82 * fr_l["e_fwd"] + 0.46 * fr_l["med_dir_3d"] + 0.22 * fr_l["e_shin"]
    CL2.shoot(cam, "shoe_heel_closeup", out_dir, Vector(fr_l["heel_focus"].tolist()),
              tuple(heel_view_dir.tolist()), size, dist=1.62, fsize=0.235)

    print("[v3_render_char] 全部 11 个视角渲染完成")


if __name__ == "__main__":
    main()
