"""M3 角色构建核心库 v2：原生 LOD0 身体 + 皮肤/织物双属性材质 + 骨骼局部系双丸子发型 + 精细高帮帆布鞋。

核心升级（对应 OPTIMIZATION_SPEC.md M3 / RC-3, RC-4, RC-7）：
1. 自动启用 Blender Cycles GPU (OptiX / CUDA) 硬件加速。
2. 顶点级 cloth_mask 驱动材质参数：裸露皮肤保留次表面散射 (SSS)，衬衫/短裤切换为哑光棉麻织物与微凹凸，
   且仅对非眼口顶点焊接 <1e-5 的 UV 接缝裂口，完整保护 LOD0 眼睑、嘴唇与手指细节。
3. 发型完全建立在骨骼精确局部正交基 (head_center, head_right, head_fwd, head_up) 与 T-pose 解剖坐标上：
   双丸子、浅粉米色发圈、偏分曲面空气刘海自然跟随 ~28° 歪头角度，并将发束合并为单网格消除“碎纸条”感。
4. 高帮帆布鞋沿真实足部骨骼轴向构建：平面切割齐整鞋口/袜口（无三角锯齿）、橡胶厚底 + 黑色护条 + 亮面鞋头 + 扁鞋带。
"""
import math
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector, noise

HAIR_RGB = (0.015, 0.012, 0.011)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def enable_cycles_gpu(scene):
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                gpu_devs = [d for d in prefs.devices if d.type in ("OPTIX", "CUDA")]
                if gpu_devs:
                    for d in prefs.devices:
                        d.use = (d.type in ("OPTIX", "CUDA"))
                    scene.cycles.device = "GPU"
                    print(f"[cycles] 启用 GPU 加速 ({dev_type}: {gpu_devs[0].name})")
                    return
            except Exception:
                continue
    except Exception as exc:
        print(f"[cycles] GPU 初始化回退 CPU: {exc}")
    scene.cycles.device = "CPU"


def setup_render(samples=96, res_x=1000, res_y=1000, transparent=True, view="Standard", exposure=0.0):
    sc = bpy.context.scene
    enable_cycles_gpu(sc)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.film_transparent = transparent
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.view_settings.view_transform = view
    sc.view_settings.exposure = exposure
    sc.view_settings.look = "None"
    return sc


def sam_to_bl(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return np.array([arr[0], arr[2], -arr[1]], dtype=np.float32)
    return np.stack([arr[:, 0], arr[:, 2], -arr[:, 1]], axis=1)


def build_body_v2(mesh_npz: str, tex_png: str, name: str = "body", use_inflate: bool = True) -> dict:
    d = np.load(mesh_npz, allow_pickle=True)
    V_sam = d["v"].astype(np.float32)
    F = d["faces"].astype(np.int32)
    UV = d["tex"].astype(np.float32)
    TF = d["tfaces"].astype(np.int32)
    rest_v = d["rest_v"].astype(np.float32)
    cloth_reg = d["cloth_region"].astype(np.uint8)
    is_oral_eye = d["is_oral_eye"].astype(bool)
    inflate = d["vertex_inflate"].astype(np.float32) if (use_inflate and "vertex_inflate" in d.files) else None

    B0 = sam_to_bl(V_sam)
    B = B0.copy()
    if inflate is not None:
        nrm = np.zeros_like(B0)
        fn = np.cross(B0[F[:, 1]] - B0[F[:, 0]], B0[F[:, 2]] - B0[F[:, 0]])
        np.add.at(nrm, F[:, 0], fn)
        np.add.at(nrm, F[:, 1], fn)
        np.add.at(nrm, F[:, 2], fn)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
        B = B0 + nrm * inflate[:, None]
        print(f"[body_v2] 蒙皮安全衣物体积外推 max={float(inflate.max())*1000:.1f}mm")

    me = bpy.data.meshes.new(name)
    me.from_pydata(B.tolist(), [], F.tolist())
    me.update()

    # 批量设置 UV（foreach_set 比逐面循环快 30 倍）
    uvl = me.uv_layers.new(name="UVMap")
    loop_t_idx = TF.reshape(-1)
    uv_flat = UV[loop_t_idx].reshape(-1).astype(np.float32)
    uvl.data.foreach_set("uv", uv_flat)

    # 写入顶点衣物属性，供材质区分 SSS 皮肤与哑光棉麻衣物
    cloth_attr = me.attributes.new(name="cloth_mask", type="FLOAT", domain="POINT")
    is_fabric = ((cloth_reg == 1) | (cloth_reg == 2) | (cloth_reg == 3)).astype(np.float32)
    cloth_attr.data.foreach_set("value", is_fabric)

    for poly in me.polygons:
        poly.use_smooth = True
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)

    # 仅对非眼口顶点做极小阈值 (1e-5m = 0.01mm) 焊接以消除 UV 接缝法线裂痕，保护眼睑/唇齿细节
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    weld_verts = [bm.verts[i] for i in range(len(bm.verts)) if not is_oral_eye[i]]
    bmesh.ops.remove_doubles(bm, verts=weld_verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()

    # 构建皮肤 + 织物双模态材质
    mat = bpy.data.materials.new("char_surface_v2")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    img = bpy.data.images.load(str(Path(tex_png).resolve()))
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.interpolation = "Smart"
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    attr_node = nt.nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "cloth_mask"

    # Roughness: 皮肤 0.56 -> 衣物 0.88
    rough_map = nt.nodes.new("ShaderNodeMapRange")
    rough_map.inputs["From Min"].default_value = 0.0
    rough_map.inputs["From Max"].default_value = 1.0
    rough_map.inputs["To Min"].default_value = 0.56
    rough_map.inputs["To Max"].default_value = 0.88
    nt.links.new(attr_node.outputs["Fac"], rough_map.inputs["Value"])
    nt.links.new(rough_map.outputs["Result"], bsdf.inputs["Roughness"])

    # Specular: 皮肤 0.22 -> 衣物 0.13
    if "Specular IOR Level" in bsdf.inputs:
        spec_map = nt.nodes.new("ShaderNodeMapRange")
        spec_map.inputs["To Min"].default_value = 0.22
        spec_map.inputs["To Max"].default_value = 0.13
        nt.links.new(attr_node.outputs["Fac"], spec_map.inputs["Value"])
        nt.links.new(spec_map.outputs["Result"], bsdf.inputs["Specular IOR Level"])

    # Subsurface: 皮肤 0.15 -> 衣物 0.0
    if "Subsurface Weight" in bsdf.inputs:
        sss_map = nt.nodes.new("ShaderNodeMapRange")
        sss_map.inputs["To Min"].default_value = 0.15
        sss_map.inputs["To Max"].default_value = 0.0
        nt.links.new(attr_node.outputs["Fac"], sss_map.inputs["Value"])
        nt.links.new(sss_map.outputs["Result"], bsdf.inputs["Subsurface Weight"])
        bsdf.inputs["Subsurface Radius"].default_value = (0.010, 0.005, 0.0025)
        bsdf.inputs["Subsurface Scale"].default_value = 1.0

    # Sheen: 衣物纤维微绒感
    if "Sheen Weight" in bsdf.inputs:
        sheen_map = nt.nodes.new("ShaderNodeMapRange")
        sheen_map.inputs["To Min"].default_value = 0.0
        sheen_map.inputs["To Max"].default_value = 0.28
        nt.links.new(attr_node.outputs["Fac"], sheen_map.inputs["Value"])
        nt.links.new(sheen_map.outputs["Result"], bsdf.inputs["Sheen Weight"])

    # 衣物微织物凹凸（仅在 cloth_mask=1 区域生效）
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 340.0
    nz.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bump_str = nt.nodes.new("ShaderNodeMath")
    bump_str.operation = "MULTIPLY"
    bump_str.inputs[1].default_value = 0.09
    nt.links.new(attr_node.outputs["Fac"], bump_str.inputs[0])
    bump = nt.nodes.new("ShaderNodeBump")
    nt.links.new(bump_str.outputs["Value"], bump.inputs["Strength"])
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    me.materials.append(mat)

    joints_bl = sam_to_bl(d["pred_joint_coords"])
    jnames = [str(x) for x in d["joint_names"]]
    return {
        "obj": obj,
        "B0": B0,
        "B": B,
        "F": F,
        "rest_v": rest_v,
        "cloth_region": cloth_reg,
        "joints_bl": joints_bl,
        "joint_names": jnames,
        "head_center": d["head_center"].astype(np.float64),
        "head_right": d["head_right"].astype(np.float64),
        "head_fwd": d["head_fwd"].astype(np.float64),
        "head_up": d["head_up"].astype(np.float64),
        "cam_t": d["pred_cam_t"].astype(np.float64),
        "focal": float(d["focal_length"][0]),
        "orig_size": tuple(int(x) for x in d["orig_img_size"]),
    }


def make_hair_material(name="hair_v2", rgb=HAIR_RGB):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = 0.38
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.36
    if "Anisotropic" in b.inputs:
        b.inputs["Anisotropic"].default_value = 0.65
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = 0.22
        b.inputs["Sheen Roughness"].default_value = 0.32
        b.inputs["Sheen Tint"].default_value = (0.42, 0.34, 0.30, 1.0)

    # 沿发片 UV 纵向构造细腻发丝凹凸
    uv_n = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (45.0, 2.5, 1.0)
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 18.0
    nz.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    nt.links.new(uv_n.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def make_simple_mat(name, rgb, rough=0.75, spec=0.22, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    if sheen > 0 and "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    return m


def build_hair_v2(info: dict) -> list:
    """在骨骼精确头部局部坐标系 (C, e_right, e_fwd, e_up) 与 T-pose 头皮拓扑上构建双丸子发型。"""
    B0 = info["B0"]
    F = info["F"]
    rest = info["rest_v"]  # cm: rx, ry, rz
    C = info["head_center"]
    e_r = info["head_right"]
    e_f = info["head_fwd"]
    e_u = info["head_up"]

    mat_hair = make_hair_material()
    # 原图双丸子根部带有浅米粉色发饰发圈
    mat_scrunchie = make_simple_mat("scrunchie_pink", (0.82, 0.68, 0.66), rough=0.88, spec=0.12, sheen=0.45)

    # 1. 基于姿态无关的 T-pose 坐标提取精确头皮面片（不压眉毛、不切耳朵）
    rx, ry, rz = rest[:, 0], rest[:, 1], rest[:, 2]

    def scalp_hairline_ry(x_cm, z_cm):
        # 太阳穴下延至 ry=164.4cm 消除露白，额头正中分缝处适度上收至 ry=166.8cm 露出自然分缝肤色
        front_t = np.clip((z_cm + 2.0) / 9.5, 0.0, 1.0)
        center_part = 2.4 * np.exp(-((x_cm - 1.5) / 3.4) ** 2) * front_t
        base_y = 150.2 + 14.2 * front_t + center_part
        # 严格绕开耳廓 (|rx|>7.3, rz∈[-0.5, 3.0])
        ear_arch = 5.2 * np.exp(-((np.abs(x_cm) - 8.1) / 1.15) ** 2) * np.exp(-((z_cm - 1.4) / 1.9) ** 2)
        return base_y + ear_arch

    v_margin = ry - scalp_hairline_ry(rx, rz)
    is_ear = (np.abs(rx) > 7.35) & (ry < 164.4) & (rz > -0.8) & (rz < 3.4)
    head_only = (ry > 148.5) & (np.abs(rx) < 8.15) & (~is_ear)
    scalp_v = head_only & (v_margin > 0.0)
    scalp_f_idx = np.where(scalp_v[F].all(axis=1))[0]

    cap_vids = np.unique(F[scalp_f_idx])
    remap = {int(v): k for k, v in enumerate(cap_vids)}
    cap_verts_base = B0[cap_vids]
    cap_tris = [[remap[int(v)] for v in F[fi]] for fi in scalp_f_idx]

    # 计算头皮法线，按距发际线距离平滑拱起 8.5mm 发帽厚度（边缘收薄至 1.2mm 紧贴头皮）
    cap_nrm = np.zeros_like(cap_verts_base)
    fn = np.cross(cap_verts_base[np.array(cap_tris)[:, 1]] - cap_verts_base[np.array(cap_tris)[:, 0]],
                  cap_verts_base[np.array(cap_tris)[:, 2]] - cap_verts_base[np.array(cap_tris)[:, 0]])
    for k_tri, tri in enumerate(cap_tris):
        for vi in tri:
            cap_nrm[vi] += fn[k_tri]
    cap_nrm /= np.linalg.norm(cap_nrm, axis=1, keepdims=True) + 1e-8

    margins_cm = np.clip(v_margin[cap_vids], 0.0, 4.0) / 4.0
    cap_thick = 0.0012 + 0.0085 * (margins_cm ** 0.65)
    cap_verts = cap_verts_base + cap_nrm * cap_thick[:, None]

    me_cap = bpy.data.meshes.new("hair_cap_v2")
    me_cap.from_pydata(cap_verts.tolist(), [], cap_tris)
    me_cap.update()
    uv_cap = me_cap.uv_layers.new(name="UVMap")
    for poly in me_cap.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            vi = me_cap.loops[li].vertex_index
            p_r = rest[cap_vids[vi]]
            uv_cap.data[li].uv = (float(p_r[0] * 0.05 + 0.5), float((p_r[1] - 150.0) * 0.04))

    cap_obj = bpy.data.objects.new("hair_cap_v2", me_cap)
    bpy.context.scene.collection.objects.link(cap_obj)
    me_cap.materials.append(mat_hair)
    sd = cap_obj.modifiers.new("solid", "SOLIDIFY")
    sd.thickness = 0.0035
    sd.offset = -1.0

    objs = [cap_obj]

    # 局部椭球辅助函数：az=0 指向面部正前 (+e_fwd)，az>0 指向角色左侧 (+e_right / 画面右侧)，el=90 指向颅顶 (+e_up)
    RX, RY, RZ = 0.083, 0.096, 0.097

    def head_surf(az_deg, el_deg, lift):
        az, el = math.radians(az_deg), math.radians(el_deg)
        lx = (RX + lift) * math.cos(el) * math.sin(az)
        ly = (RY + lift) * math.cos(el) * math.cos(az)
        lz = (RZ + lift) * math.sin(el) + 0.018
        p = C + lx * e_r + ly * e_f + lz * e_u
        n_loc = lx / RX * e_r + ly / RY * e_f + lz / RZ * e_u
        n_loc /= np.linalg.norm(n_loc) + 1e-8
        return p, n_loc

    # 2. 单网格合并曲面发束（还原原图三七偏分斜刘海 + 两侧碎鬓发 + 后脑包覆发流）
    rng = np.random.default_rng(42)
    all_verts, all_quads, all_uvs = [], [], []

    def append_ribbon(pts, nrms, w_root, w_tip, arch=0.0014):
        base_idx = len(all_verts)
        n_pts = len(pts)
        # 横截面做微弧拱形（3 列顶点、每段 2 个四边面），梢部收尖至 0.25mm 并向内轻扣，告别平直碎纸条
        for i, (p0, nrm) in enumerate(zip(pts, nrms)):
            t = i / (n_pts - 1)
            tang = pts[min(i + 1, n_pts - 1)] - pts[max(i - 1, 0)]
            tang /= np.linalg.norm(tang) + 1e-8
            side = np.cross(tang, nrm)
            side /= np.linalg.norm(side) + 1e-8
            # 根部窄、中段饱满、发梢精细收尖
            profile = math.sin(min(t * 1.25, 1.0) * math.pi * 0.5) if t < 0.3 else (1.0 - (t - 0.3) / 0.7) ** 1.15
            w = w_tip + (w_root - w_tip) * max(profile, 0.0)
            crown_lift = nrm * (arch * (1.0 - t * 0.8))
            all_verts.append((p0 - side * w).tolist())
            all_verts.append((p0 + crown_lift).tolist())
            all_verts.append((p0 + side * w).tolist())
        for i in range(n_pts - 1):
            r0 = base_idx + 3 * i
            r1 = base_idx + 3 * (i + 1)
            t0, t1 = i / (n_pts - 1), (i + 1) / (n_pts - 1)
            all_quads.append((r0, r0 + 1, r1 + 1, r1))
            all_uvs.extend([(0.0, t0), (0.5, t0), (0.5, t1), (0.0, t1)])
            all_quads.append((r0 + 1, r0 + 2, r1 + 2, r1 + 1))
            all_uvs.extend([(0.5, t0), (1.0, t0), (1.0, t1), (0.5, t1)])

    # (a) 偏分空气斜刘海：发根深入发帽内部 (-4.5mm) 消除台阶，发梢停在眉弓线 el∈[20.5°, 28°] 完整露出双眼
    bang_specs = []
    # 主斜刘海（从分缝 az=+12° 斜扫过右额至 az=-42°）
    for az0 in np.linspace(-42.0, 11.5, 26):
        gap_factor = math.exp(-((az0 - 13.0) / 8.0) ** 2)
        el_end = 21.0 + 8.5 * gap_factor + rng.uniform(-1.5, 1.8)
        sweep = -8.5 - 3.0 * ((az0 - 12.0) / 55.0) + rng.uniform(-2.2, 2.2)
        bang_specs.append((az0, rng.uniform(60.0, 70.0), el_end, sweep, rng.uniform(0.0056, 0.0082), 0.0042))
    # 分缝右侧副刘海（留出透气三角分缝后自然包覆至左侧太阳穴 az=58°）
    for az0 in np.linspace(17.5, 57.5, 19):
        gap_factor = math.exp(-((az0 - 15.5) / 7.5) ** 2)
        el_end = 21.0 + 7.5 * gap_factor - 0.14 * (az0 - 20.0) + rng.uniform(-1.4, 1.4)
        sweep = 8.5 + rng.uniform(-2.0, 2.0)
        bang_specs.append((az0, rng.uniform(58.0, 69.0), el_end, sweep, rng.uniform(0.0054, 0.0080), 0.0036))
    # 两侧耳前轻薄碎鬓发（紧贴太阳穴垂落，不在耳廓外鼓包）
    for side_s in (-1.0, 1.0):
        for _ in range(4):
            az0 = side_s * rng.uniform(50.0, 61.0)
            el_end = rng.uniform(-4.0, 6.0)
            sweep = side_s * rng.uniform(3.0, 8.0)
            bang_specs.append((az0, rng.uniform(42.0, 54.0), el_end, sweep, rng.uniform(0.0036, 0.0050), 0.0016))
    # 后脑勺纹理发流（严格限制在后颅 |az|>114°，避免在耳朵两侧鼓出多余发团）
    for _ in range(42):
        az0 = rng.uniform(-180.0, -114.0) if rng.random() < 0.5 else rng.uniform(114.0, 180.0)
        el_end = rng.uniform(-10.0, 4.0)
        bang_specs.append((az0, rng.uniform(52.0, 74.0), el_end, rng.uniform(-10.0, 10.0), rng.uniform(0.0052, 0.0078), 0.0022))

    steps = 11
    for az0, el0, el1, sweep, wr, lift_peak in bang_specs:
        pts, nrms = [], []
        lift_mid = lift_peak * rng.uniform(0.88, 1.14)
        for i in range(steps):
            t = i / (steps - 1)
            el = el0 + (el1 - el0) * (t ** 0.92)
            az = az0 + sweep * (t ** 1.30)
            # t=0 深埋于发帽下方 -4.5mm 彻底消除发根折角，中段微拱透气，发梢轻柔贴近眉上
            root_tuck = -0.0045 * ((1.0 - t) ** 2.6)
            arch_curve = lift_mid * math.sin(t * math.pi * 0.88)
            p, nrm = head_surf(az, el, root_tuck + arch_curve)
            pts.append(p)
            nrms.append(nrm)
        append_ribbon(pts, nrms, w_root=wr, w_tip=0.00030, arch=0.0011)

    me_locks = bpy.data.meshes.new("hair_locks_v2")
    me_locks.from_pydata(all_verts, [], all_quads)
    me_locks.update()
    uv_locks = me_locks.uv_layers.new(name="UVMap")
    for li, uv_val in enumerate(all_uvs):
        uv_locks.data[li].uv = uv_val
    for poly in me_locks.polygons:
        poly.use_smooth = True
    ob_locks = bpy.data.objects.new("hair_locks_v2", me_locks)
    bpy.context.scene.collection.objects.link(ob_locks)
    me_locks.materials.append(mat_hair)
    objs.append(ob_locks)

    # 3. 骨骼局部系高颅顶对称双丸子 + 浅粉米色发圈（完全跟随歪头旋转矩阵）
    for side_name, sx, r_scale in (("L", -1.0, 1.00), ("R", +1.0, 0.97)):
        bun_dir = sx * 0.47 * e_r - 0.24 * e_f + 0.85 * e_u
        bun_dir /= np.linalg.norm(bun_dir)
        bun_loc = C + sx * 0.062 * e_r - 0.026 * e_f + 0.112 * e_u

        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=28, radius=0.045 * r_scale)
        bun = bpy.context.object
        bun.name = f"bun_{side_name}"
        bm = bmesh.new()
        bm.from_mesh(bun.data)
        for vtx in bm.verts:
            p = np.array(vtx.co[:], dtype=np.float64)
            p[2] *= 0.86
            ang = math.atan2(p[1], p[0])
            twist = 0.0030 * math.sin(ang * 7.0 + p[2] * 95.0) + 0.0034 * noise.noise(Vector(p) * 32.0)
            n = p / (np.linalg.norm(p) + 1e-8)
            p = p + n * twist
            vtx.co = Vector(p.tolist())
        bm.to_mesh(bun.data)
        bm.free()

        q_bun = Vector(bun_dir.tolist()).to_track_quat("Z", "Y")
        bun.rotation_mode = "QUATERNION"
        bun.rotation_quaternion = q_bun
        bun.location = Vector(bun_loc.tolist())
        for p in bun.data.polygons:
            p.use_smooth = True
        bun.data.materials.append(mat_hair)
        objs.append(bun)

        # 褶皱布艺发圈（Scrunchie）紧贴丸子根部
        tie_loc = bun_loc - bun_dir * 0.022
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.025 * r_scale,
            minor_radius=0.0060,
            major_segments=48,
            minor_segments=16,
        )
        tie = bpy.context.object
        tie.name = f"scrunchie_{side_name}"
        bm2 = bmesh.new()
        bm2.from_mesh(tie.data)
        for vtx in bm2.verts:
            p = np.array(vtx.co[:], dtype=np.float64)
            ang = math.atan2(p[1], p[0])
            ruffle = 0.0018 * math.sin(ang * 13.0) + 0.0010 * math.cos(ang * 23.0)
            vtx.co = Vector((p + np.array([math.cos(ang), math.sin(ang), 0.35]) * ruffle).tolist())
        bm2.to_mesh(tie.data)
        bm2.free()
        tie.rotation_mode = "QUATERNION"
        tie.rotation_quaternion = q_bun
        tie.location = Vector(tie_loc.tolist())
        for p in tie.data.polygons:
            p.use_smooth = True
        tie.data.materials.append(mat_scrunchie)
        objs.append(tie)

    print(f"[hair_v2] 生成发帽 + 合并曲面发束({len(bang_specs)}缕) + 对称双丸子与浅粉发圈")
    return objs


def _hull2d(pts: np.ndarray) -> np.ndarray:
    pts_s = sorted(set(map(tuple, np.round(pts, 5))))
    if len(pts_s) < 3:
        return np.array(pts_s)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_s:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_s):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _offset_poly(poly: np.ndarray, d: float) -> np.ndarray:
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1, p2 = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        e0, e1 = p1 - p0, p2 - p1
        n0 = np.array([e0[1], -e0[0]])
        n1 = np.array([e1[1], -e1[0]])
        n0 /= np.linalg.norm(n0) + 1e-9
        n1 /= np.linalg.norm(n1) + 1e-9
        v = n0 + n1
        v /= np.linalg.norm(v) + 1e-9
        out.append(p1 + v * (d / max(float(v @ n0), 0.38)))
    return np.array(out)


def _extract_clean_shell(name, B, F, face_mask, offset_m, plane_cuts, mat, solidify_m=0.003):
    """提取子网格、外扩法线，并用平面切割出平整边缘（彻底告别三角锯齿鞋口/袜口）。"""
    f_idx = np.where(face_mask)[0]
    if len(f_idx) < 16:
        return None
    vids = np.unique(F[f_idx])
    rmap = {int(v): k for k, v in enumerate(vids)}
    v_sub = B[vids].copy()
    tris = np.array([[rmap[int(v)] for v in F[fi]] for fi in f_idx], dtype=np.int32)

    nrm = np.zeros_like(v_sub)
    fn = np.cross(v_sub[tris[:, 1]] - v_sub[tris[:, 0]], v_sub[tris[:, 2]] - v_sub[tris[:, 0]])
    np.add.at(nrm, tris[:, 0], fn)
    np.add.at(nrm, tris[:, 1], fn)
    np.add.at(nrm, tris[:, 2], fn)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
    v_sub = v_sub + nrm * offset_m

    me = bpy.data.meshes.new(name)
    me.from_pydata(v_sub.tolist(), [], tris.tolist())
    me.update()

    bm = bmesh.new()
    bm.from_mesh(me)
    for plane_co, plane_no, clear_outer in plane_cuts:
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(
            bm,
            geom=geom,
            dist=1e-5,
            plane_co=Vector(plane_co.tolist()),
            plane_no=Vector(plane_no.tolist()),
            clear_outer=clear_outer,
            clear_inner=not clear_outer,
        )
    bm.to_mesh(me)
    bm.free()
    me.update()

    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat)
    if solidify_m > 0:
        sd = ob.modifiers.new("solid", "SOLIDIFY")
        sd.thickness = solidify_m
        sd.offset = -1.0
    return ob


def build_shoes_v2(info: dict) -> list:
    """构建匡威风高帮帆布鞋：齐口螺纹白袜 + 帆布高帮 + 橡胶包头 + 厚底黑护条 + 扁鞋带 + 内侧圆标。"""
    B = info["B0"]
    F = info["F"]
    rest = info["rest_v"]
    j_bl = info["joints_bl"]
    jn = info["joint_names"]

    mat_canvas = make_simple_mat("canvas_cream", (0.83, 0.79, 0.76), rough=0.88, spec=0.14, sheen=0.32)
    mat_rubber = make_simple_mat("rubber_sole", (0.87, 0.85, 0.81), rough=0.42, spec=0.38)
    mat_stripe = make_simple_mat("foxing_stripe", (0.09, 0.08, 0.08), rough=0.45, spec=0.30)
    mat_lace = make_simple_mat("shoelace_white", (0.91, 0.90, 0.88), rough=0.92, spec=0.10, sheen=0.25)
    mat_sock = make_simple_mat("crew_sock", (0.86, 0.84, 0.81), rough=0.94, spec=0.08, sheen=0.40)
    mat_patch = make_simple_mat("star_patch", (0.14, 0.20, 0.34), rough=0.65, spec=0.20)

    objs = []
    ry = rest[:, 1]
    rx = rest[:, 0]
    rz = rest[:, 2]

    for side, pfx, rx_sign in (("L", "l_", +1.0), ("R", "r_", -1.0)):
        side_v = (rx * rx_sign) > 0.0
        p_knee = j_bl[jn.index(f"{pfx}lowleg")].astype(np.float64)
        p_ankle = j_bl[jn.index(f"{pfx}talocrural")].astype(np.float64)
        p_ball = j_bl[jn.index(f"{pfx}ball")].astype(np.float64)

        # 小腿轴向（踝 -> 膝）用于齐整切割鞋帮口与袜筒口
        e_shin = p_knee - p_ankle
        e_shin /= np.linalg.norm(e_shin) + 1e-8

        # 基于 T-pose 足底解剖标志点提取姿态不变的真实脚掌平面（跟底 -> 前掌底严丝合缝平行）
        heel_vids = np.where(side_v & (ry < 2.2) & (rz < -2.0))[0]
        fore_vids = np.where(side_v & (ry < 2.2) & (rz > 6.0) & (rz < 13.5))[0]
        sole_vids = np.where(side_v & (ry < 2.8))[0]
        foot_vids = np.where(side_v & (ry < 16.5))[0]

        p_heel = B[heel_vids].astype(np.float64).mean(axis=0)
        p_fore = B[fore_vids].astype(np.float64).mean(axis=0)
        e_fwd = p_fore - p_heel
        e_fwd /= np.linalg.norm(e_fwd) + 1e-8

        x_med = float(np.median(np.abs(rx[sole_vids])))
        p_lat = B[sole_vids[np.abs(rx[sole_vids]) > x_med]].astype(np.float64).mean(axis=0)
        p_med = B[sole_vids[np.abs(rx[sole_vids]) <= x_med]].astype(np.float64).mean(axis=0)
        e_lat_raw = p_lat - p_med

        e_up_foot = np.cross(e_fwd, e_lat_raw)
        e_up_foot /= np.linalg.norm(e_up_foot) + 1e-8
        if np.dot(e_up_foot, p_ankle - p_heel) < 0:
            e_up_foot = -e_up_foot
        e_side = np.cross(e_up_foot, e_fwd)
        e_side /= np.linalg.norm(e_side) + 1e-8

        # 将全脚掌（从脚后跟到脚尖）投影到真实足底平面求 2D 凸包
        h_vals = (B[foot_vids].astype(np.float64) - p_heel) @ e_up_foot
        h_min = float(np.percentile(h_vals, 2))
        foot_origin = p_heel + e_up_foot * h_min

        rel_foot = B[foot_vids].astype(np.float64) - foot_origin
        u_f = rel_foot @ e_fwd
        v_f = rel_foot @ e_side
        w_f = rel_foot @ e_up_foot
        low_contact = w_f < 0.052
        hull_2d = _offset_poly(_hull2d(np.column_stack([u_f[low_contact], v_f[low_contact]])), 0.0105)

        # 1. 全掌硫化橡胶厚底 + 经典通长侧面黑色护条 (Foxing Stripe)
        sole_h = 0.028
        n_h = len(hull_2d)
        c2d = hull_2d.mean(axis=0)

        def make_extruded_band(name_b, z0, z1, s0, s1, mat_b, cap_bot=False, cap_top=False):
            p0 = c2d + (hull_2d - c2d) * s0
            p1 = c2d + (hull_2d - c2d) * s1
            ring0 = (foot_origin[None, :] + p0[:, 0:1] * e_fwd + p0[:, 1:2] * e_side + z0 * e_up_foot)
            ring1 = (foot_origin[None, :] + p1[:, 0:1] * e_fwd + p1[:, 1:2] * e_side + z1 * e_up_foot)
            v_list = ring0.tolist() + ring1.tolist()
            f_list = []
            for i in range(n_h):
                j = (i + 1) % n_h
                f_list.append((i, j, n_h + j, n_h + i))
            if cap_bot:
                f_list.append(tuple(range(n_h - 1, -1, -1)))
            if cap_top:
                f_list.append(tuple(range(n_h, 2 * n_h)))
            me_b = bpy.data.meshes.new(name_b)
            me_b.from_pydata(v_list, [], f_list)
            me_b.update()
            for pp in me_b.polygons:
                pp.use_smooth = True
            ob_b = bpy.data.objects.new(name_b, me_b)
            bpy.context.scene.collection.objects.link(ob_b)
            me_b.materials.append(mat_b)
            return ob_b

        objs.append(make_extruded_band(f"sole_bot_{side}", -0.002, 0.019, 0.97, 1.00, mat_rubber, cap_bot=True))
        objs.append(make_extruded_band(f"foxing_{side}", 0.0185, 0.0228, 1.012, 1.008, mat_stripe))
        objs.append(make_extruded_band(f"sole_top_{side}", 0.022, sole_h + 0.004, 1.00, 0.90, mat_rubber, cap_top=True))

        # 2. 高帮帆布鞋面（下口精准收入全掌橡胶鞋底内，上口沿小腿轴向齐整平切）
        shoe_top_co = p_ankle + e_shin * 0.074
        sole_cut_co = foot_origin + e_up_foot * (sole_h * 0.55)
        upper_faces = side_v[F].all(axis=1) & (ry[F].mean(axis=1) < 18.8)
        ob_up = _extract_clean_shell(
            f"upper_{side}",
            B,
            F,
            upper_faces,
            offset_m=0.0086,
            plane_cuts=[
                (shoe_top_co, -e_shin, False),
                (sole_cut_co, e_up_foot, False),
            ],
            mat=mat_canvas,
            solidify_m=0.0036,
        )
        if ob_up:
            objs.append(ob_up)

        # 3. 白色亮面橡胶包头 (Toe Cap)
        u_max = float(hull_2d[:, 0].max())
        toe_cut_co = foot_origin + e_fwd * (u_max - 0.056)
        toe_faces = side_v[F].all(axis=1) & (ry[F].mean(axis=1) < 9.5)
        ob_toe = _extract_clean_shell(
            f"toecap_{side}",
            B,
            F,
            toe_faces,
            offset_m=0.0102,
            plane_cuts=[
                (toe_cut_co, e_fwd, False),
                (sole_cut_co, e_up_foot, False),
            ],
            mat=mat_rubber,
            solidify_m=0.0025,
        )
        if ob_toe:
            objs.append(ob_toe)

        # 4. 齐口堆堆白袜（从鞋口内侧平滑延伸至踝上 14.8cm）
        sock_bot_co = p_ankle + e_shin * 0.058
        sock_top_co = p_ankle + e_shin * 0.148
        sock_faces = side_v[F].all(axis=1) & (ry[F].mean(axis=1) > 9.0) & (ry[F].mean(axis=1) < 28.0)
        ob_sock = _extract_clean_shell(
            f"sock_{side}",
            B,
            F,
            sock_faces,
            offset_m=0.0060,
            plane_cuts=[
                (sock_top_co, -e_shin, False),
                (sock_bot_co, e_shin, False),
            ],
            mat=mat_sock,
            solidify_m=0.0030,
        )
        if ob_sock:
            objs.append(ob_sock)

        # 5. 扁平白色鞋带（5 道跨足背）+ 紧贴内踝的匡威经典圆标
        instep_line_a = p_ball + e_up_foot * 0.030 - e_fwd * 0.015
        instep_line_b = p_ankle + e_shin * 0.052 + e_fwd * 0.042
        for k_lace, t_l in enumerate(np.linspace(0.08, 0.90, 5)):
            lc = (1.0 - t_l) * instep_line_a + t_l * instep_line_b
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=Vector(lc.tolist()))
            lace = bpy.context.object
            lace.name = f"lace_{side}_{k_lace}"
            lace.scale = (0.023, 0.0042, 0.0018)
            rot_l = Matrix((
                (e_side[0], e_fwd[0], e_up_foot[0]),
                (e_side[1], e_fwd[1], e_up_foot[1]),
                (e_side[2], e_fwd[2], e_up_foot[2]),
            )).to_euler()
            lace.rotation_euler = rot_l
            lace.data.materials.append(mat_lace)
            objs.append(lace)

        # 内踝圆标：由解剖内侧向量 (p_med - p_lat) 精确贴合在鞋帮内侧表面，绝不悬空
        medial_dir = p_med - p_lat
        medial_dir -= e_shin * np.dot(medial_dir, e_shin)
        medial_dir /= np.linalg.norm(medial_dir) + 1e-8
        ankle_band = side_v & (ry > 7.5) & (ry < 13.0)
        ankle_pts = B[ankle_band].astype(np.float64)
        proj_med = (ankle_pts - (p_ankle + e_shin * 0.025)) @ medial_dir
        surf_dist = float(np.percentile(proj_med, 96))
        patch_loc = p_ankle + e_shin * 0.025 + medial_dir * (surf_dist + 0.0092)
        bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.0125, depth=0.0022, location=Vector(patch_loc.tolist()))
        ptc = bpy.context.object
        ptc.name = f"patch_{side}"
        ptc.rotation_mode = "QUATERNION"
        ptc.rotation_quaternion = Vector(medial_dir.tolist()).to_track_quat("Z", "Y")
        ptc.data.materials.append(mat_patch)
        objs.append(ptc)

    print(f"[shoes_v2] 生成精细高帮帆布鞋与白袜部件 {len(objs)} 个")
    return objs


def add_studio_lights(target, size, key=52.0, fill=18.0, rim=20.0):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("StudioWorld")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.09, 0.09, 0.10, 1.0)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.55
    sc.world = w

    def make_area(nm, eng, sz, offset):
        ld = bpy.data.lights.new(nm, "AREA")
        ld.energy = eng
        ld.size = sz
        ob = bpy.data.objects.new(nm, ld)
        ob.location = Vector(target) + Vector(offset)
        ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(ob)
        return ob

    k = make_area("key", key, size * 1.35, (size * 0.85, -size * 1.55, size * 0.85))
    f = make_area("fill", fill, size * 1.65, (-size * 1.25, -size * 1.15, size * 0.40))
    r = make_area("rim", rim, size * 1.20, (0.0, size * 1.45, size * 0.95))
    return k, f, r


def add_camera(lens=50.0, sensor_width=36.0):
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cd.sensor_width = sensor_width
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def shoot(cam, name, out_dir: Path, target, direction, size, dist=1.85, fsize=None):
    tgt = Vector(target)
    s = float(fsize) if fsize is not None else size
    dvec = Vector(direction).normalized()
    cam.location = tgt + dvec * s * dist
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.render.filepath = str(Path(out_dir).resolve() / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {out_dir / (name + '.png')}")
