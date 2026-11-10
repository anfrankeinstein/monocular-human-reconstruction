"""S7 头发入口：调用 char_lib 的共享实现。

用法：
  blender -b -P build_hair.py -- <mesh_uv.npz> <skin_atlas.png> <out_dir> [samples] [res]
"""
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import char_lib as CL

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MESH, TEX, OUT = argv[0], argv[1], Path(argv[2])
SAMPLES = int(argv[3]) if len(argv) > 3 else 64
RES = int(argv[4]) if len(argv) > 4 else 1000
OUT.mkdir(parents=True, exist_ok=True)

CL.clear_scene()
sc = CL.setup_render(samples=SAMPLES, res=RES, transparent=True, view="Standard")
INFLATE = argv[5] if len(argv) > 5 else None
body, B, F = CL.build_body(MESH, TEX, inflate=INFLATE)
head, z_top, crn = CL.head_info(B)
CL.build_hair(body, B, CL.hair_material())
if os.environ.get("STB_SHOES", "1") == "1":
    CL.build_shoes(B, F, float(B[:, 2].min()))

co = np.array([v.co[:] for v in body.data.vertices])
mn, mx = co.min(axis=0), co.max(axis=0)
center = Vector(((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2))
size = float(max(mx - mn))
CL.add_lights(center, size, key=50.0, fill=16.0)
cam = CL.add_camera(50.0)

CL.shoot(cam, "hair_front", OUT, center, (0, -1, 0.1), size)
CL.shoot(cam, "hair_3q", OUT, center, (0.75, -1, 0.12), size, dist=1.8)
CL.shoot(cam, "hair_back", OUT, center, (-0.35, 1, 0.14), size, dist=1.9)
CL.shoot(cam, "hair_side", OUT, center, (-1, -0.35, 0.10), size, dist=1.8)
CL.shoot(cam, "hair_face", OUT, head, (0.05, -1, 0.10), size, dist=1.55, fsize=size * 0.30)
print("[blender] done")
