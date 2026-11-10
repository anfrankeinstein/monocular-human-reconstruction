"""角色构建共享库：身体 + 皮肤材质 + 头发。供 build_hair.py / build_scene.py 复用。

坐标系：Blender，+Z 上，-Y 为人物正面。角色是"body-centric"的，脚底在 z≈0.679。
"""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

SKIN_ROUGH = 0.62
SKIN_SPEC = 0.18
SKIN_SSS = 0.16
HAIR_RGB = (0.0165, 0.0135, 0.0125)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(samples=64, res=1000, transparent=True, view="Standard", exposure=0.0):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.film_transparent = transparent
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.view_settings.view_transform = view
    sc.view_settings.exposure = exposure
    sc.view_settings.look = "None"
    return sc


def _principled(mat):
    return mat.node_tree.nodes["Principled BSDF"]


def build_body(mesh_npz, tex_png, name="body", inflate=None):
    """返回 (obj, verts_blender, faces)"""
    d = np.load(mesh_npz)
    V = d["v"].astype(np.float32)
    F = d["faces"].astype(np.int32)
    UV = d["tex"].astype(np.float32)
    TF = d["tfaces"].astype(np.int32)
    B = np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1)

    me = bpy.data.meshes.new(name)
    me.from_pydata(B.tolist(), [], F.tolist())
    me.update()

    # 衣物体积：在焊接之前按索引套用外推场（焊接会改变顶点索引）
    if inflate:
        disp = np.load(inflate).astype(np.float32)
        assert len(disp) == len(B), f"inflate 长度 {len(disp)} != 顶点数 {len(B)}"
        nrm = np.zeros_like(B, dtype=np.float32)
        for tri in F:
            n = np.cross(B[tri[1]] - B[tri[0]], B[tri[2]] - B[tri[0]])
            for vi in tri:
                nrm[vi] += n
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
        for i, vtx in enumerate(me.vertices):
            vtx.co = Vector((B[i] + nrm[i] * disp[i]).tolist())
        me.update()
        print(f"[body] 衣物体积外推 最大 {float(disp.max())*1000:.1f}mm")

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)

    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for k in range(poly.loop_total):
            li = poly.loop_start + k
            t = TF[poly.index]
            uvl.data[li].uv = (float(UV[t[k], 0]), float(UV[t[k], 1]))
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0005)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    b = _principled(mat)
    img = bpy.data.images.load(tex_png)
    tn = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tn.image = img
    tn.interpolation = "Smart"
    mat.node_tree.links.new(tn.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = SKIN_ROUGH
    b.inputs["Specular IOR Level"].default_value = SKIN_SPEC
    if "Subsurface Weight" in b.inputs:
        b.inputs["Subsurface Weight"].default_value = SKIN_SSS
        b.inputs["Subsurface Radius"].default_value = (0.010, 0.005, 0.0025)
        b.inputs["Subsurface Scale"].default_value = 1.0
    me.materials.append(mat)
    return obj, B, F


def head_info(B):
    z_top = float(B[:, 2].max())
    crn = B[int(np.argmax(B[:, 2]))]
    head = Vector((float(crn[0]), float(crn[1]), z_top - 0.082))
    return head, z_top, crn


def hairline_z(head, y, x=0.0):
    # 前额发际线压到眉毛上方约 1.585，后颈约 1.50；再压低就会遮住眼睛
    base = float(head.z) - 0.005 - 0.44 * (y - float(head.y))
    # 沿 x 做低频起伏，让刘海下沿呈一缕一缕的轮廓，而不是一条直边
    # 低频决定每缕的长度（做成不规则的一缕一缕），高频只做毛边
    return (base + 0.011 * np.sin(x * 23.0 + 0.4) + 0.006 * np.sin(x * 61.0 + 2.1)
            + 0.0022 * np.sin(x * 137.0))


def hair_material(name="hair", rgb=HAIR_RGB, rough=0.44, bump_scale=260.0, bump_strength=0.22):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = _principled(m)
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.32
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = 0.18
        b.inputs["Sheen Roughness"].default_value = 0.35
        b.inputs["Sheen Tint"].default_value = (0.45, 0.36, 0.30, 1.0)
    nt = m.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = bump_scale
    nz.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def build_hair(body_obj, B, mat, thick=0.0090, r_head=0.108, cards=True, n_lock=300):
    """以头皮面片为基底生成发帽 + 双丸子 + 发圈。返回对象列表。"""
    head, z_top, _ = head_info(B)
    me = body_obj.data

    def keep(c):
        if (Vector((c[0], c[1], c[2])) - head).length > r_head * 1.24:
            return False
        return c[2] > hairline_z(head, c[1], c[0])

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    drop = [f for f in bm.faces if not keep(f.calc_center_median())]
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    cap = bpy.data.meshes.new("hair_cap")
    bm.to_mesh(cap)
    bm.free()

    cap_obj = bpy.data.objects.new("hair_cap", cap)
    bpy.context.scene.collection.objects.link(cap_obj)
    cap.update()
    for p in cap.polygons:
        p.use_smooth = True

    nrm = np.zeros((len(cap.vertices), 3), np.float32)
    for poly in cap.polygons:
        n = np.array(poly.normal[:], np.float32)
        for vi in poly.vertices:
            nrm[vi] += n
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8

    for i, vtx in enumerate(cap.vertices):
        p = np.array(vtx.co[:], np.float32)
        front = float(np.clip((float(head.y) - p[1]) / 0.075, 0.0, 1.0))
        bulge = 0.0045 * front * front
        d_edge = p[2] - hairline_z(head, p[1], p[0])
        # 边缘厚度收到 3mm，否则刘海下沿是一圈厚帽檐
        edge = float(np.clip(d_edge / 0.028, 0.0, 1.0))
        fade = float(np.clip(d_edge / 0.022, 0.0, 1.0))
        clump = 0.0050 * noise.noise(Vector(p) * 60.0) * fade
        t_now = 0.003 + (thick + bulge + clump - 0.003) * (edge ** 0.7)
        vtx.co = Vector((p + nrm[i] * t_now).tolist())

    solid = cap_obj.modifiers.new("solid", "SOLIDIFY")
    solid.thickness = 0.006
    solid.offset = -1.0

    objs = [cap_obj]

    def add_bun(name, offset, radius, squash=0.86):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius)
        ob = bpy.context.object
        ob.name = name
        ob.location = Vector(head) + Vector(offset)
        ob.scale = (1.0, 1.0, squash)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bm2 = bmesh.new()
        bm2.from_mesh(ob.data)
        for vtx in bm2.verts:
            p = np.array(vtx.co[:], np.float32)
            n = np.array(vtx.normal[:], np.float32)
            d2 = 0.0055 * noise.noise(Vector(p) * 28.0 + Vector((offset[0] * 30, 0, 0)))
            vtx.co = Vector((p + n * d2).tolist())
        bm2.to_mesh(ob.data)
        bm2.free()
        for p in ob.data.polygons:
            p.use_smooth = True
        return ob

    R = 0.046
    objs.append(add_bun("bun_L", (-0.070, 0.042, 0.080), R))
    objs.append(add_bun("bun_R", (0.064, 0.042, 0.086), R * 0.95))

    def add_torus(name, loc, major, minor, rot):
        bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                         major_segments=40, minor_segments=12,
                                         location=loc, rotation=rot)
        ob = bpy.context.object
        ob.name = name
        for p in ob.data.polygons:
            p.use_smooth = True
        return ob

    objs.append(add_torus("tie_L", Vector(head) + Vector((-0.062, 0.042, 0.046)),
                          0.024, 0.005, (math.radians(74), 0, math.radians(-20))))
    objs.append(add_torus("tie_R", Vector(head) + Vector((0.057, 0.042, 0.052)),
                          0.023, 0.005, (math.radians(74), 0, math.radians(20))))

    for ob in objs:
        ob.data.materials.append(mat)

    if cards:
        base = len(objs)
        objs += build_hair_cards(head, mat, n_lock=n_lock)
        objs += build_flyaways(head, mat, n=10)
        print(f"[hair] 发片 {len(objs) - base - 10} 条 + 飞发 10 条（发帽 {len(cap.polygons)} 面）")

    return objs


def add_lights(target, size, key=50.0, fill=16.0, world=(0.08, 0.08, 0.09), world_strength=0.5):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*world, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = world_strength
    sc.world = w

    k = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    k.data.energy = key
    k.data.size = size * 1.3
    k.location = Vector(target) + Vector((size * 0.9, -size * 1.5, size * 0.9))
    k.rotation_euler = (Vector(target) - k.location).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(k)

    f = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "AREA"))
    f.data.energy = fill
    f.data.size = size * 1.6
    f.location = Vector(target) + Vector((-size * 1.2, -size * 1.1, size * 0.4))
    f.rotation_euler = (Vector(target) - f.location).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(f)
    return k, f


def add_camera(lens=50.0):
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def shoot(cam, name, out_dir, target, direction, size, dist=1.85, fsize=None):
    tgt = Vector(target)
    s = float(fsize) if fsize is not None else size
    dvec = Vector(direction).normalized()
    cam.location = tgt + dvec * s * dist
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.render.filepath = str(out_dir / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {name}.png")


# ------------------------------------------------------------------ 发片（hair cards）
RX_H, RY_H, RZ_H = 0.083, 0.096, 0.091


def _on_head(head, az_deg, el_deg, out):
    az, el = math.radians(az_deg), math.radians(el_deg)
    x = (RX_H + out) * math.cos(el) * math.sin(az)
    y = -(RY_H + out) * math.cos(el) * math.cos(az)
    z = (RZ_H + out) * math.sin(el)
    return np.array([head.x, head.y, head.z]) + np.array([x, y, z])


def build_hair_cards(head, mat, n_lock=300, steps=9, seed=7,
                     w_root=0.0042, w_tip=0.0012, out_root=0.003, out_tip=0.010):
    """沿头皮撒发缕，逐条生成带状卡片。比整片壳更接近真实发型的剪影。

    az = 0 为正前方(-Y)，el = 90 为头顶。前额发缕 el 收到 14°（眉毛上方），
    两侧到 -8°，后颈到 -26°。
    """
    rng = np.random.default_rng(seed)
    objs = []
    # 前额更密（刘海是视觉重点），其余均匀
    az_list = []
    for _ in range(int(n_lock * 0.38)):
        az_list.append(rng.uniform(-62, 62))
    for _ in range(n_lock - len(az_list)):
        az_list.append(rng.uniform(-180, 180))

    for li, az0 in enumerate(az_list):
        front = abs(az0) < 62
        el0 = rng.uniform(52, 72)
        if front:
            el1 = rng.uniform(9, 20)
        elif abs(az0) < 115:
            el1 = rng.uniform(-14, -2)
        else:
            el1 = rng.uniform(-30, -16)
        drift = rng.uniform(-26, 26)
        wr = w_root * rng.uniform(0.72, 1.25)
        wt = w_tip * rng.uniform(0.7, 1.3)

        pts, nrms = [], []
        for i in range(steps):
            t = i / (steps - 1)
            el = el0 + (el1 - el0) * t
            az = az0 + drift * (t * t) * (1.0 if li % 2 else -1.0) * 1.0
            out = out_root + (out_tip - out_root) * t
            pts.append(_on_head(head, az, el, out))
            c = np.array([head.x, head.y, head.z])
            nrms.append((pts[-1] - c) / (np.linalg.norm(pts[-1] - c) + 1e-8))

        verts, quads = [], []
        for i, (p0, nrm) in enumerate(zip(pts, nrms)):
            t = i / (steps - 1)
            tang = (pts[min(i + 1, steps - 1)] - pts[max(i - 1, 0)])
            tang /= np.linalg.norm(tang) + 1e-8
            side = np.cross(tang, nrm)
            side /= np.linalg.norm(side) + 1e-8
            w = wr * (1 - t) + wt * t
            verts.append((p0 - side * w).tolist())
            verts.append((p0 + side * w).tolist())
        for i in range(steps - 1):
            a = 2 * i
            quads.append((a, a + 1, a + 3, a + 2))

        me = bpy.data.meshes.new(f"lock_{li}")
        me.from_pydata(verts, [], quads)
        me.update()
        ob = bpy.data.objects.new(f"lock_{li}", me)
        bpy.context.scene.collection.objects.link(ob)
        for poly in me.polygons:
            poly.use_smooth = True
        me.materials.append(mat)
        objs.append(ob)
    return objs


def build_flyaways(head, mat, n=10, seed=11):
    """轮廓外的碎发/飞发，破掉整片壳的呆板边缘。"""
    rng = np.random.default_rng(seed)
    objs = []
    for i in range(n):
        az0 = rng.uniform(-180, 180)
        el0 = rng.uniform(30, 70)
        drift = rng.uniform(-40, 40)
        pts = []
        for k in range(4):
            t = k / 3
            el = el0 - 40 * t
            az = az0 + drift * t * t
            out = 0.010 + 0.016 * t * t
            pts.append(_on_head(head, az, el, out))
        verts = []
        for k, p0 in enumerate(pts):
            t = k / 3
            w = 0.0018 * (1 - 0.6 * t)
            nrm = (p0 - np.array([head.x, head.y, head.z]))
            nrm /= np.linalg.norm(nrm) + 1e-8
            tang = (pts[min(k + 1, 3)] - pts[max(k - 1, 0)])
            tang /= np.linalg.norm(tang) + 1e-8
            side = np.cross(tang, nrm)
            side /= np.linalg.norm(side) + 1e-8
            verts += [(p0 - side * w).tolist(), (p0 + side * w).tolist()]
        quads = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(3)]
        me = bpy.data.meshes.new(f"fly_{i}")
        me.from_pydata(verts, [], quads)
        me.update()
        ob = bpy.data.objects.new(f"fly_{i}", me)
        bpy.context.scene.collection.objects.link(ob)
        for poly in me.polygons:
            poly.use_smooth = True
        me.materials.append(mat)
        objs.append(ob)
    return objs


# ------------------------------------------------------------------ 鞋
def _hull2d(pts):
    """Andrew monotone chain，返回逆时针凸包。"""
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _offset_poly(poly, d):
    """把凸包沿外法线外扩 d（逆时针多边形）。"""
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1, p2 = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        e0 = p1 - p0
        e1 = p2 - p1
        n0 = np.array([e0[1], -e0[0]])
        n1 = np.array([e1[1], -e1[0]])
        n0 /= np.linalg.norm(n0) + 1e-9
        n1 /= np.linalg.norm(n1) + 1e-9
        v = n0 + n1
        v /= np.linalg.norm(v) + 1e-9
        out.append(p1 + v * (d / max(v @ n0, 0.35)))
    return np.array(out)


def _mat(name, rgb, rough=0.6, spec=0.35, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    if sheen and "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    return m


def build_shoes(B, F, floor_z, upper_top=0.115, sole_h=0.032, toe_in=0.010):
    """按左右脚做两只低模运动鞋：厚鞋底 + 鞋面壳 + 鞋带 + 白袜。

    B/F 是身体网格（Blender 坐标）。脚底在 floor_z。
    """
    objs = []
    zc = B[:, 2]
    foot = np.where(zc < floor_z + 0.20)[0]
    if len(foot) < 50:
        print("[shoes] 找不到脚部顶点")
        return objs
    xs = B[foot, 0]
    mid = 0.5 * (xs.min() + xs.max())
    M_SOLE = _mat("sole", (0.74, 0.74, 0.72), rough=0.62, spec=0.30)
    M_UP = _mat("shoe_upper", (0.84, 0.84, 0.82), rough=0.86, spec=0.16, sheen=0.3)
    M_LACE = _mat("lace", (0.90, 0.90, 0.88), rough=0.92, spec=0.1)
    M_SOCK = _mat("sock", (0.86, 0.86, 0.84), rough=0.95, spec=0.08, sheen=0.35)

    for side, sel in (("L", xs < mid), ("R", xs >= mid)):
        vids = foot[sel]
        if len(vids) < 30:
            continue
        zlo = B[vids, 2].min()
        # ---- 鞋底：足印凸包挤出 ----
        contact = vids[B[vids, 2] < zlo + 0.055]
        if len(contact) < 10:
            contact = vids
        poly = _offset_poly(_hull2d(B[contact][:, :2]), toe_in)
        n = len(poly)
        rings = []
        for zz, shrink in ((zlo - 0.001, 0.30), (zlo + 0.006, 0.0),
                           (zlo + sole_h, 0.10), (zlo + sole_h + 0.004, 0.22)):
            c = poly.mean(axis=0)
            rings.append(np.column_stack([c + (poly - c) * (1 - shrink),
                                          np.full(n, zz)]))
        verts, faces = [], []
        for r in rings:
            verts += r.tolist()
        for k in range(len(rings) - 1):
            a, b = k * n, (k + 1) * n
            for i in range(n):
                j = (i + 1) % n
                faces.append((a + i, a + j, b + j, b + i))
        faces.append(tuple(range(n - 1, -1, -1)))                      # 底盖
        top0 = (len(rings) - 1) * n
        faces.append(tuple(range(top0, top0 + n)))                      # 顶盖
        me = bpy.data.meshes.new(f"sole_{side}")
        me.from_pydata(verts, [], faces)
        me.update()
        ob = bpy.data.objects.new(f"sole_{side}", me)
        bpy.context.scene.collection.objects.link(ob)
        for pp in me.polygons:
            pp.use_smooth = True
        me.materials.append(M_SOLE)
        objs.append(ob)

        # ---- 鞋面：脚背壳 ----
        top_z = zlo + upper_top
        sel_f = [fi for fi in range(len(F))
                 if all(v in set(vids.tolist()) for v in F[fi])
                 and B[F[fi]][:, 2].mean() > zlo + sole_h * 0.6
                 and B[F[fi]][:, 2].mean() < top_z]
        if len(sel_f) > 20:
            vv = sorted({int(v) for fi in sel_f for v in F[fi]})
            rmap = {v: k for k, v in enumerate(vv)}
            vts = [B[v].tolist() for v in vv]
            tris = [[rmap[int(v)] for v in F[fi]] for fi in sel_f]
            me = bpy.data.meshes.new(f"upper_{side}")
            me.from_pydata(vts, [], tris)
            me.update()
            nrm = np.zeros((len(me.vertices), 3), np.float32)
            for pp in me.polygons:
                nn = np.array(pp.normal[:], np.float32)
                for vi in pp.vertices:
                    nrm[vi] += nn
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
            for i, vtx in enumerate(me.vertices):
                q = np.array(vtx.co[:], np.float32)
                vtx.co = Vector((q + nrm[i] * 0.010).tolist())
            ob = bpy.data.objects.new(f"upper_{side}", me)
            bpy.context.scene.collection.objects.link(ob)
            for pp in me.polygons:
                pp.use_smooth = True
            me.materials.append(M_UP)
            sd = ob.modifiers.new("solid", "SOLIDIFY")
            sd.thickness = 0.004
            sd.offset = -1.0
            objs.append(ob)

            # ---- 鞋带：跨足背的几根细圆柱 ----
            seg = [v for v in vv if B[v, 2] > zlo + 0.045]
            if len(seg) > 10:
                for k, frac in enumerate((0.35, 0.48, 0.61, 0.74)):
                    zz = zlo + 0.045 + (top_z - zlo - 0.05) * frac
                    band = [v for v in seg if abs(B[v, 2] - zz) < 0.012]
                    if len(band) < 4:
                        continue
                    pts = B[band]
                    c = pts.mean(axis=0)
                    d = pts[np.argmax(np.linalg.norm(pts - c, axis=1))] - c
                    length = 2 * np.linalg.norm(d)
                    bpy.ops.mesh.primitive_cylinder_add(
                        vertices=8, radius=0.0032, depth=length,
                        location=(c[0], c[1], c[2] + 0.008))
                    ob2 = bpy.context.object
                    ob2.name = f"lace_{side}_{k}"
                    v_dir = Vector((d[0], d[1], 0.0)).normalized()
                    ob2.rotation_mode = "QUATERNION"
                    ob2.rotation_quaternion = v_dir.to_track_quat("Z", "Y")
                    ob2.data.materials.append(M_LACE)
                    objs.append(ob2)

        # ---- 白袜：踝上一圈 ----
        sock_f = [fi for fi in range(len(F))
                  if all(v in set(vids.tolist()) for v in F[fi])
                  and B[F[fi]][:, 2].mean() >= top_z - 0.005
                  and B[F[fi]][:, 2].mean() < top_z + 0.075]
        if len(sock_f) > 10:
            vv = sorted({int(v) for fi in sock_f for v in F[fi]})
            rmap = {v: k for k, v in enumerate(vv)}
            me = bpy.data.meshes.new(f"sock_{side}")
            me.from_pydata([B[v].tolist() for v in vv], [],
                           [[rmap[int(v)] for v in F[fi]] for fi in sock_f])
            me.update()
            nrm = np.zeros((len(me.vertices), 3), np.float32)
            for pp in me.polygons:
                nn = np.array(pp.normal[:], np.float32)
                for vi in pp.vertices:
                    nrm[vi] += nn
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
            for i, vtx in enumerate(me.vertices):
                q = np.array(vtx.co[:], np.float32)
                vtx.co = Vector((q + nrm[i] * 0.0065).tolist())
            ob = bpy.data.objects.new(f"sock_{side}", me)
            bpy.context.scene.collection.objects.link(ob)
            for pp in me.polygons:
                pp.use_smooth = True
            me.materials.append(M_SOCK)
            objs.append(ob)

    print(f"[shoes] 生成 {len(objs)} 个部件")
    return objs
