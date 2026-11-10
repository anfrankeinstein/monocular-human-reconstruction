"""M3 角色多视角渲染入口（对标旧版 SHOE1 全部 5 个机位，便于逐视角同框对比）。

用法：
  "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" -b -P scripts/v2_render_char.py -- ^
      work/q004-base/v2/mesh_lod0_v2.npz work/q004-base/v2/tex/char_atlas_v2.png ^
      work/q004-base/renders/OPT_CHAR 96 1000
"""
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import char_lib_v2 as CL2


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mesh_npz = argv[0] if len(argv) > 0 else r"work\q004-base\v2\mesh_lod0_v2.npz"
    tex_png = argv[1] if len(argv) > 1 else r"work\q004-base\v2\tex\char_atlas_v2.png"
    out_dir = Path(argv[2]) if len(argv) > 2 else Path(r"work\q004-base\renders\OPT_CHAR")
    samples = int(argv[3]) if len(argv) > 3 else 96
    res = int(argv[4]) if len(argv) > 4 else 1000
    out_dir.mkdir(parents=True, exist_ok=True)

    CL2.clear_scene()
    CL2.setup_render(samples=samples, res_x=res, res_y=res, transparent=True, view="Standard")

    info = CL2.build_body_v2(mesh_npz, tex_png, use_inflate=True)
    CL2.build_hair_v2(info)
    CL2.build_shoes_v2(info)

    B = info["B"]
    mn, mx = B.min(axis=0), B.max(axis=0)
    center = Vector(((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0))
    size = float(max(mx - mn))
    head_target = Vector(info["head_center"].tolist())

    CL2.add_studio_lights(center, size, key=52.0, fill=18.0, rim=18.0)
    cam = CL2.add_camera(lens=50.0)

    CL2.shoot(cam, "hair_front", out_dir, center, (0.0, -1.0, 0.10), size)
    CL2.shoot(cam, "hair_3q", out_dir, center, (0.75, -1.0, 0.12), size, dist=1.80)
    CL2.shoot(cam, "hair_back", out_dir, center, (-0.35, 1.0, 0.14), size, dist=1.90)
    CL2.shoot(cam, "hair_side", out_dir, center, (-1.0, -0.35, 0.10), size, dist=1.80)
    CL2.shoot(cam, "hair_face", out_dir, head_target, (0.05, -1.0, 0.10), size, dist=1.55, fsize=size * 0.30)
    print("[v2_render_char] 全部 5 个视角渲染完成")


if __name__ == "__main__":
    main()
