"""v3 角色装配与材质核心库：纯肤色 SSS 身体 + 独立物理布料衬衫壳体 + 重磅水洗牛仔短裤壳体（含纽扣/裤耳细节）。"""
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import char_lib_v2 as CL2


def build_pure_skin_body_v3(mesh_npz: str, skin_png: str, name: str = "body_v3") -> dict:
    """构建零膨胀纯解剖精度 LOD0 身体网格（服装已完全解耦为独立 3D 物理壳体）。"""
    d = np.load(str(Path(mesh_npz).resolve()), allow_pickle=True)
    V_sam = d["v"].astype(np.float32)
    F = d["faces"].astype(np.int32)
    UV = d["tex"].astype(np.float32)
    TF = d["tfaces"].astype(np.int32)
    rest_v = d["rest_v"].astype(np.float32)
    is_oral_eye = d["is_oral_eye"].astype(bool)

    B0 = CL2.sam_to_bl(V_sam)

    me = bpy.data.meshes.new(name)
    me.from_pydata(B0.tolist(), [], F.tolist())
    me.update()

    uvl = me.uv_layers.new(name="UVMap")
    uv_flat = UV[TF.reshape(-1)].reshape(-1).astype(np.float32)
    uvl.data.foreach_set("uv", uv_flat)

    for poly in me.polygons:
        poly.use_smooth = True
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)

    # 仅对非眼口顶点焊接 <1e-5 的 UV 接缝裂口，确保体表法线丝滑无接缝且眼睑/唇齿毫发无损
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    weld_verts = [bm.verts[i] for i in range(len(bm.verts)) if not is_oral_eye[i]]
    bmesh.ops.remove_doubles(bm, verts=weld_verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()

    # 纯净真人次表面散射皮肤材质（无需任何织物妥协混合）
    mat = bpy.data.materials.new("pure_skin_sss_v3")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    img = bpy.data.images.load(str(Path(skin_png).resolve()))
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.interpolation = "Smart"
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.52
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.24
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.17
        bsdf.inputs["Subsurface Radius"].default_value = (0.010, 0.005, 0.0025)
        bsdf.inputs["Subsurface Scale"].default_value = 1.0

    me.materials.append(mat)

    joints_bl = CL2.sam_to_bl(d["pred_joint_coords"])
    jnames = [str(x) for x in d["joint_names"]]
    return {
        "obj": obj,
        "B0": B0,
        "B": B0,
        "F": F,
        "rest_v": rest_v,
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


def _make_shirt_fabric_material(shirt_png: str) -> bpy.types.Material:
    """轻盈棉麻/人造丝古巴领印花衬衫材质（细腻经纬织纹凹凸 + 纤维微光泽）。"""
    m = bpy.data.materials.new("shirt_shell_mat_v3")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]

    img = bpy.data.images.load(str(Path(shirt_png).resolve()))
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = img
    tx.interpolation = "Smart"
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])

    b.inputs["Roughness"].default_value = 0.84
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.14
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = 0.32
        b.inputs["Sheen Roughness"].default_value = 0.45

    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 420.0
    nz.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.065
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _make_denim_fabric_material(shorts_png: str) -> bpy.types.Material:
    """12oz 重磅水洗斜纹牛仔布材质（接缝压线凹凸 + 纬向白棉绒感边缘光泽）。"""
    m = bpy.data.materials.new("shorts_shell_mat_v3")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]

    img = bpy.data.images.load(str(Path(shorts_png).resolve()))
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = img
    tx.interpolation = "Smart"
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])

    b.inputs["Roughness"].default_value = 0.90
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.12
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = 0.42
        b.inputs["Sheen Roughness"].default_value = 0.55
        b.inputs["Sheen Tint"].default_value = (0.88, 0.92, 1.0, 1.0)

    # 利用贴图明暗（压线/毛边/斜纹）驱动真实三维压线凹凸
    rgb2bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(tx.outputs["Color"], rgb2bw.inputs["Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(rgb2bw.outputs["Val"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _compute_normals_np(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.zeros_like(v, dtype=np.float64)
    v64 = v.astype(np.float64)
    fn = np.cross(v64[f[:, 1]] - v64[f[:, 0]], v64[f[:, 2]] - v64[f[:, 0]])
    np.add.at(n, f[:, 0], fn)
    np.add.at(n, f[:, 1], fn)
    np.add.at(n, f[:, 2], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return n.astype(np.float32)


def _create_shell_object(name: str, verts: np.ndarray, faces: np.ndarray, uv: np.ndarray,
                         tfaces: np.ndarray, mat: bpy.types.Material,
                         thickness_m: float, smooth_iters: int = 2) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()

    uvl = me.uv_layers.new(name="UVMap")
    uv_flat = uv[tfaces.reshape(-1)].reshape(-1).astype(np.float32)
    uvl.data.foreach_set("uv", uv_flat)

    for p in me.polygons:
        p.use_smooth = True
    me.update()

    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat)

    if smooth_iters > 0:
        cs = ob.modifiers.new("corr_smooth", "CORRECTIVE_SMOOTH")
        cs.iterations = smooth_iters
        cs.factor = 0.42
        cs.smooth_type = "SIMPLE"

    # 先在单层流形曲面上做 1 阶曲面细分，再沿平滑法线赋予均匀布料实体厚度（use_even_offset=False 杜绝尖角除零毛刺）
    sub = ob.modifiers.new("subsurf_cloth", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 1

    sd = ob.modifiers.new("solidify_cloth", "SOLIDIFY")
    sd.thickness = thickness_m
    sd.offset = 0.0
    sd.use_rim = True
    sd.use_even_offset = False
    return ob


def build_simulated_garments_v3(sim_npz: str, shirt_png: str, shorts_png: str) -> dict:
    """加载物理解算后的三维衬衫壳体与牛仔短裤壳体，赋予实体厚度、细分曲面及五金纽扣/裤耳配件。"""
    ds = np.load(str(Path(sim_npz).resolve()), allow_pickle=True)

    v_sh = ds["shirt_v_bl"].astype(np.float32)
    f_sh = ds["shirt_f"].astype(np.int32)
    uv_sh = ds["shirt_uv"].astype(np.float32)
    tf_sh = ds["shirt_tf"].astype(np.int32)
    r_sh = ds["shirt_rest"].astype(np.float32)

    v_dn = ds["shorts_v_bl"].astype(np.float32)
    f_dn = ds["shorts_f"].astype(np.int32)
    uv_dn = ds["shorts_uv"].astype(np.float32)
    tf_dn = ds["shorts_tf"].astype(np.int32)
    r_dn = ds["shorts_rest"].astype(np.float32)

    mat_shirt = _make_shirt_fabric_material(shirt_png)
    mat_shorts = _make_denim_fabric_material(shorts_png)

    ob_shirt = _create_shell_object(
        "shirt_shell_v3", v_sh, f_sh, uv_sh, tf_sh, mat_shirt, thickness_m=0.0018, smooth_iters=2
    )
    ob_shorts = _create_shell_object(
        "shorts_shell_v3", v_dn, f_dn, uv_dn, tf_dn, mat_shorts, thickness_m=0.0028, smooth_iters=2
    )

    detail_objs = []
    n_sh = _compute_normals_np(v_sh, f_sh)
    n_dn = _compute_normals_np(v_dn, f_dn)

    # 1. 衬衫前门襟珍珠母贝小纽扣（3 颗沿正中门襟均匀排列于下摆上方）
    mat_pearl = CL2.make_simple_mat("pearl_button", (0.86, 0.83, 0.80), rough=0.25, spec=0.45)
    for k_btn, ry_target in enumerate((124.2, 121.0, 117.8)):
        front_sh = (r_sh[:, 2] > 3.0) & (np.abs(r_sh[:, 0] + 0.4) < 2.2) & (np.abs(r_sh[:, 1] - ry_target) < 1.8)
        if np.any(front_sh):
            idx_b = np.where(front_sh)[0]
            best = idx_b[int(np.argmin((r_sh[idx_b, 0] + 0.4) ** 2 + (r_sh[idx_b, 1] - ry_target) ** 2))]
            p_btn = v_sh[best] + n_sh[best] * 0.0020
            bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.0046, depth=0.0018, location=Vector(p_btn.tolist()))
            btn = bpy.context.object
            btn.name = f"shirt_button_{k_btn}"
            btn.rotation_mode = "QUATERNION"
            btn.rotation_quaternion = Vector(n_sh[best].tolist()).to_track_quat("Z", "Y")
            btn.data.materials.append(mat_pearl)
            detail_objs.append(btn)

    # 2. 牛仔短裤高腰头正中古铜金属工字扣（还原原图肚脐正下方标志性圆扣！）
    mat_brass = bpy.data.materials.new("antique_brass_btn")
    mat_brass.use_nodes = True
    bsdf_br = mat_brass.node_tree.nodes["Principled BSDF"]
    bsdf_br.inputs["Base Color"].default_value = (0.24, 0.19, 0.14, 1.0)
    bsdf_br.inputs["Metallic"].default_value = 0.82
    bsdf_br.inputs["Roughness"].default_value = 0.36

    hip_lift_arr = 2.1 * np.exp(-((np.abs(r_dn[:, 0]) - 11.5) / 4.8) ** 2) * (0.5 + 0.5 * np.tanh(r_dn[:, 2] / 4.0))
    wb_top_arr = 99.2 - 2.2 * np.tanh(r_dn[:, 2] / 4.5) + hip_lift_arr
    wb_front = (r_dn[:, 2] > 4.0) & (np.abs(r_dn[:, 0]) < 2.0) & (np.abs(r_dn[:, 1] - (wb_top_arr - 1.5)) < 1.5)
    if np.any(wb_front):
        idx_w = np.where(wb_front)[0]
        best_w = idx_w[int(np.argmin(r_dn[idx_w, 0] ** 2 + (r_dn[idx_w, 1] - (wb_top_arr[idx_w] - 1.5)) ** 2))]
        p_wb = v_dn[best_w] + n_dn[best_w] * 0.0026
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.0078, depth=0.0028, location=Vector(p_wb.tolist()))
        jbtn = bpy.context.object
        jbtn.name = "jeans_waist_button"
        jbtn.rotation_mode = "QUATERNION"
        jbtn.rotation_quaternion = Vector(n_dn[best_w].tolist()).to_track_quat("Z", "Y")
        jbtn.data.materials.append(mat_brass)
        detail_objs.append(jbtn)

    # 3. 牛仔短裤 5 条立体裤耳 (Belt Loops)：自适应前低后高人体工学高腰沿
    loop_specs = [(-6.2, +1.0), (+6.2, +1.0), (-13.5, 0.0), (+13.5, 0.0), (0.0, -1.0)]
    for k_lp, (lx_t, z_sign) in enumerate(loop_specs):
        if z_sign > 0:
            sel = (r_dn[:, 2] > 2.0) & (np.abs(r_dn[:, 0] - lx_t) < 2.2)
        elif z_sign < 0:
            sel = (r_dn[:, 2] < -2.0) & (np.abs(r_dn[:, 0] - lx_t) < 2.2)
        else:
            sel = (np.abs(r_dn[:, 2]) < 5.0) & (np.sign(r_dn[:, 0]) == np.sign(lx_t)) & (np.abs(r_dn[:, 0]) > 10.0)
        top_cand = np.where(sel & (np.abs(r_dn[:, 1] - (wb_top_arr - 0.5)) < 1.3))[0]
        bot_cand = np.where(sel & (np.abs(r_dn[:, 1] - (wb_top_arr - 3.3)) < 1.3))[0]
        if len(top_cand) > 0 and len(bot_cand) > 0:
            it = top_cand[int(np.argmin(np.abs(r_dn[top_cand, 0] - lx_t)))]
            ib = bot_cand[int(np.argmin(np.abs(r_dn[bot_cand, 0] - lx_t)))]
            pt_a = v_dn[it] + n_dn[it] * 0.0028
            pt_b = v_dn[ib] + n_dn[ib] * 0.0028
            mid = 0.5 * (pt_a + pt_b)
            n_m = 0.5 * (n_dn[it] + n_dn[ib])
            n_m /= np.linalg.norm(n_m) + 1e-8
            up_v = pt_a - pt_b
            h_len = float(np.linalg.norm(up_v))
            if h_len > 0.012:
                up_v /= h_len
                side_v = np.cross(n_m, up_v)
                side_v /= np.linalg.norm(side_v) + 1e-8
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=Vector((mid + n_m * 0.0016).tolist()))
                lp = bpy.context.object
                lp.name = f"belt_loop_{k_lp}"
                lp.scale = (0.0085, h_len * 1.05, 0.0032)
                rot_m = Matrix((
                    (side_v[0], up_v[0], n_m[0]),
                    (side_v[1], up_v[1], n_m[1]),
                    (side_v[2], up_v[2], n_m[2]),
                )).to_euler()
                lp.rotation_euler = rot_m
                lp.data.materials.append(mat_shorts)
                detail_objs.append(lp)

    all_garment_verts = np.vstack([v_sh, v_dn])
    print(f"[garments_v3] 已装配实体物理壳体: shirt({len(v_sh)}v) + shorts({len(v_dn)}v) + 配件({len(detail_objs)}个)")
    return {
        "shirt_obj": ob_shirt,
        "shorts_obj": ob_shorts,
        "detail_objs": detail_objs,
        "garment_verts_bl": all_garment_verts,
        "shirt_v_bl": v_sh,
        "shorts_v_bl": v_dn,
        "shorts_rest": r_dn,
    }
