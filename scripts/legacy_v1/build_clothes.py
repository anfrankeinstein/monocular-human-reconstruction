"""S8.5 服装：把照片里"非皮肤的面"挑出来，细分后重新按皮肤 mask 裁剪，再加厚成壳。

为什么要细分：皮肤 mask 在 2D 上是平滑的，但身体网格的面约 1–2cm，
逐面二值化会把衣服边缘切成一圈锯齿。先 2 级细分再判定，边界细 4 倍。

用法：
  blender -b -P build_clothes.py -- <mesh_uv.npz> <skin_atlas.png> <cloth_atlas.png> <out_dir> \
                                  <flags.npz> <skin_mask.npy> [samples] [res]
"""
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

sys.path.insert(0, str(Path(__file__).parent))
import char_lib as CL

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MESH, SKIN_TEX, CLOTH_TEX, OUT = argv[0], argv[1], argv[2], Path(argv[3])
FLAGS, MASK_NPY = argv[4], argv[5]
SAMPLES = int(argv[6]) if len(argv) > 6 else 64
RES = int(argv[7]) if len(argv) > 7 else 1000
OUT.mkdir(parents=True, exist_ok=True)

CAM_T = np.array([0.06798, 1.17540, 2.36280])
FOCAL = 1795.69
IMG_W, IMG_H = 1402, 1122
SUBDIV = 2

CL.clear_scene()
sc = CL.setup_render(samples=SAMPLES, res=RES, transparent=True, view="Standard")
body, B, F = CL.build_body(MESH, SKIN_TEX)
head, z_top, crn = CL.head_info(B)
CL.build_hair(body, B, CL.hair_material())

d = np.load(MESH)
UV = d["tex"].astype(np.float32)
TF = d["tfaces"].astype(np.int32)
fl = np.load(FLAGS)
facing = fl["facing"].astype(bool)
is_skin = fl["is_skin"].astype(bool)
cloth_faces = facing & (~is_skin)

# 先把服装区域向外扩 1 圈再细分：否则细分只发生在区域内部，
# 外边界仍是原始面边 → 裁剪后照样是锯齿。
from collections import defaultdict
v2f = defaultdict(list)
for fi in range(len(F)):
    for vi in F[fi]:
        v2f[vi].append(fi)
cur = set(int(x) for x in np.where(cloth_faces)[0])
for _ in range(1):
    nxt = set(cur)
    for fi in cur:
        nxt.update(v2f[vi] for vi in F[fi]) if False else None
        for vi in F[fi]:
            nxt.update(v2f[vi])
    cur = nxt
idx = np.array(sorted(cur))
vids = sorted({int(v) for i in idx for v in F[i]})
remap = {v: k for k, v in enumerate(vids)}
verts = [B[v].tolist() for v in vids]
tris = [[remap[int(v)] for v in F[i]] for i in idx]
uvs = [UV[TF[i]].tolist() for i in idx]
print(f"[cloth] 衣物面 {len(idx)}  顶点 {len(vids)}")

me = bpy.data.meshes.new("cloth")
me.from_pydata(verts, [], tris)
me.update()
uvl = me.uv_layers.new(name="UVMap")
for li, poly in enumerate(me.polygons):
    t = uvs[li]
    for k in range(poly.loop_total):
        uvl.data[poly.loop_start + k].uv = (float(t[k][0]), float(t[k][1]))

# ---------------------------------------------------------------- 细分
bm = bmesh.new()
bm.from_mesh(me)
for _ in range(SUBDIV):
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
bm.to_mesh(me)
bm.free()
print(f"[cloth] 细分后 {len(me.polygons)} 面")

# ---------------------------------------------------------------- 按皮肤 mask 裁剪
mask = np.load(MASK_NPY)                       # uint8, (H, W)
co = np.array([v.co[:] for v in me.vertices], np.float64)
# Blender (x, y, z) → SAM 相机系 (x, -z, y)：SAM 的 y 向下、z 是深度
ps = np.stack([co[:, 0], -co[:, 2], co[:, 1]], axis=1) + CAM_T
z = np.clip(ps[:, 2], 1e-3, None)
u = FOCAL * ps[:, 0] / z + IMG_W / 2.0
vv = FOCAL * ps[:, 1] / z + IMG_H / 2.0

keep = set()
for pp in me.polygons:
    vi = list(pp.vertices)
    uc = u[vi].mean()
    vc = vv[vi].mean()
    xi = int(np.clip(uc, 0, IMG_W - 1))
    yi = int(np.clip(vc, 0, IMG_H - 1))
    if mask[yi, xi] == 0:
        keep.add(tuple(sorted(vi)))
print(f"[cloth] 裁剪后 {len(keep)} 面")

bm = bmesh.new()
bm.from_mesh(me)
bm.faces.ensure_lookup_table()
drop = [f for f in bm.faces if tuple(sorted(vv2.index for vv2 in f.verts)) not in keep]
bmesh.ops.delete(bm, geom=drop, context="FACES")
bm.to_mesh(me)
bm.free()
me.update()

# ---------------------------------------------------------------- 外推
nrm = np.zeros((len(me.vertices), 3), np.float32)
for poly in me.polygons:
    nn = np.array(poly.normal[:], np.float32)
    for vi in poly.vertices:
        nrm[vi] += nn
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8

for i, vtx in enumerate(me.vertices):
    pp = np.array(vtx.co[:], np.float32)
    zz = pp[2]
    t = 0.020 if zz > 1.15 else (0.014 if zz > 0.86 else 0.010)
    wob = 0.0012 * noise.noise(Vector(pp) * 30.0)
    vtx.co = Vector((pp + nrm[i] * (t + wob)).tolist())

for poly in me.polygons:
    poly.use_smooth = True

sol = bpy.data.objects.new("cloth_shell", me)
bpy.context.scene.collection.objects.link(sol)

m = bpy.data.materials.new("cloth")
m.use_nodes = True
nt = m.node_tree
bsdf = nt.nodes["Principled BSDF"]
img = bpy.data.images.load(CLOTH_TEX)
tx = nt.nodes.new("ShaderNodeTexImage")
tx.image = img
tx.interpolation = "Smart"
nt.links.new(tx.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.88
if "Specular IOR Level" in bsdf.inputs:
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
tc = nt.nodes.new("ShaderNodeTexCoord")
nz = nt.nodes.new("ShaderNodeTexNoise")
nz.inputs["Scale"].default_value = 220.0
bump = nt.nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.2
nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
me.materials.append(m)
sd = sol.modifiers.new("solid", "SOLIDIFY")
sd.thickness = 0.0025
sd.offset = -1.0

# ---------------------------------------------------------------- 灯光 / 相机
co2 = np.array([v.co[:] for v in body.data.vertices])
mn, mx = co2.min(axis=0), co2.max(axis=0)
center = Vector(((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2))
size = float(max(mx - mn))
CL.add_lights(center, size, key=50.0, fill=16.0)
cam = CL.add_camera(50.0)

CL.shoot(cam, "cloth_front", OUT, center, (0, -1, 0.1), size)
CL.shoot(cam, "cloth_3q", OUT, center, (0.75, -1, 0.12), size, dist=1.8)
CL.shoot(cam, "cloth_face", OUT, head, (0.05, -1, 0.10), size, dist=1.55, fsize=size * 0.30)
print("[blender] done")
