"""M1-B (envs/mhr0)：原生 MHR LOD0 前向 + 精确 3D 解剖语义分区 + 蒙皮约束衣物体积场。

输出单一自包含数据包 work/q004-base/v2/mesh_lod0_v2.npz，供后续贴图合成、角色构建与场景渲染共用。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_MHR = Path(r"Q:\stb-assets\repos\MHR")
if str(REPO_MHR) not in sys.path:
    sys.path.insert(0, str(REPO_MHR))


def smooth_scalar_on_mesh(val: np.ndarray, faces: np.ndarray, iters: int = 14, alpha: float = 0.65) -> np.ndarray:
    """用 PyTorch 稀疏邻接在 7.3 万顶点网格上快速做拉普拉斯平滑。"""
    f = torch.as_tensor(faces, dtype=torch.long)
    e0 = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    e1 = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    n = len(val)
    deg = torch.zeros(n, dtype=torch.float32)
    deg.scatter_add_(0, e0, torch.ones_like(e0, dtype=torch.float32))
    deg = deg.clamp(min=1.0)

    x = torch.as_tensor(val, dtype=torch.float32)
    for _ in range(iters):
        nb_sum = torch.zeros(n, dtype=torch.float32)
        nb_sum.scatter_add_(0, e0, x[e1])
        x = (1.0 - alpha) * x + alpha * (nb_sum / deg)
    return x.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=r"models\mhr-assets\assets")
    ap.add_argument("--params", default=r"work\q004-base\v2\params_v2.npz")
    ap.add_argument("--out", default=r"work\q004-base\v2\mesh_lod0_v2.npz")
    args = ap.parse_args()

    from mhr.mhr import MHR

    d = np.load(args.params, allow_pickle=True)
    sh = torch.as_tensor(d["shape_params"], dtype=torch.float32).view(1, -1)
    mp = torch.as_tensor(d["mhr_model_params"], dtype=torch.float32).view(1, -1)
    ex = torch.as_tensor(d["expr_params"], dtype=torch.float32).view(1, -1)

    print("[mhr0] 加载原生 MHR LOD0（开启 pose correctives）...")
    m0 = MHR.from_files(folder=Path(args.assets), device=torch.device("cpu"),
                        lod=0, wants_pose_correctives=True)
    with torch.no_grad():
        v0_cm, skel0 = m0(sh, mp, ex)

    # 转换到 SAM 相机米制坐标：(x_cm/100, -y_cm/100, -z_cm/100)
    v0 = v0_cm[0].numpy().astype(np.float32) / 100.0
    v0[:, 1] *= -1.0
    v0[:, 2] *= -1.0

    j0 = skel0[0, :, :3].numpy().astype(np.float32) / 100.0
    j0[:, 1] *= -1.0
    j0[:, 2] *= -1.0

    mesh0 = m0.character.mesh
    faces = np.asarray(mesh0.faces, dtype=np.int32)
    tex = np.asarray(mesh0.texcoords, dtype=np.float32)
    tfaces = np.asarray(mesh0.texcoord_faces, dtype=np.int32)
    rest = np.asarray(mesh0.vertices, dtype=np.float32)  # (73639, 3), T-pose cm: +Y up, +Z fwd

    jn = list(m0.character.skeleton.joint_names)
    sw_idx = np.asarray(m0.character.skin_weights.index, dtype=np.int32)    # (N, 8)
    sw_val = np.asarray(m0.character.skin_weights.weight, dtype=np.float32) # (N, 8)
    n_verts = len(rest)

    def sum_joint_weights(keywords):
        match = np.array([any(k in name.lower() for k in keywords) for name in jn], dtype=bool)
        hit = match[sw_idx]
        return (sw_val * hit).sum(axis=1)

    w_oral_eye = sum_joint_weights(["eye", "teeth", "tongue"])
    w_head_neck = sum_joint_weights(["head", "neck", "jaw", "eye", "teeth", "tongue",
                                     "brow", "cheek", "lip", "ear", "nose"])
    w_forearm_hand = sum_joint_weights(["lowarm", "wrist", "thumb", "index", "middle", "ring", "pinky"])
    w_uparm = sum_joint_weights(["uparm", "clavicle"])
    w_torso = sum_joint_weights(["spine"])
    w_pelvis_upleg = sum_joint_weights(["root", "upleg"])
    w_lowleg = sum_joint_weights(["lowleg"])
    w_foot = sum_joint_weights(["foot", "talocrural", "subtalar", "transversetarsal", "ball"])

    is_oral_eye = (w_oral_eye > 0.25)
    # 3D 面部核心五官区（眉眼鼻唇齿舌，处于额头刘海线 rest_y=163.6cm 以下）：保留瞳孔/眼线/嘴唇暗部细节
    is_face_core = (
        ((rest[:, 1] >= 148.5) & (rest[:, 1] <= 163.6) &
         (np.abs(rest[:, 0]) <= 6.8) & (rest[:, 2] >= 5.0))
        | is_oral_eye
    )

    rx, ry, rz = rest[:, 0], rest[:, 1], rest[:, 2]
    ax = np.abs(rx)

    # 1 = 宽松短袖印花衬衫（含后背与短袖，排除前胸 V 领裸露三角区）
    vneck_half_w = np.clip((ry - 128.5) / (143.5 - 128.5), 0.0, 1.0) * 5.8
    in_vneck = (rz > 2.5) & (ry > 128.5) & (ax < vneck_half_w)
    is_shirt = (
        ((w_torso + w_uparm) > 0.45)
        & (w_head_neck < 0.35)
        & (w_forearm_hand < 0.40)
        & (ry >= 117.8)
        & (ry <= 143.5)
        & (ax <= 29.2)
        & (~in_vneck)
        & (~is_oral_eye)
    )

    # 2 = 高腰牛仔短裤（含臀部与裤腿斜切边）
    shorts_hem_y = 79.5 + 0.42 * np.clip(ax - 4.0, 0.0, 14.0)
    is_shorts = (
        ((w_pelvis_upleg + w_torso) > 0.50)
        & (w_lowleg < 0.30)
        & (ry <= 95.8)
        & (ry >= shorts_hem_y)
        & (~is_oral_eye)
    )

    # 3 = 中筒白袜与高帮帆布鞋区域
    is_sock_shoe = ((w_foot > 0.30) | ((w_lowleg > 0.40) & (ry <= 23.0))) & (~is_oral_eye)

    cloth_region = np.zeros(n_verts, dtype=np.uint8)
    cloth_region[is_shirt] = 1
    cloth_region[is_shorts] = 2
    cloth_region[is_sock_shoe] = 3

    # 蒙皮约束下的衣物体积外推场（米）
    disp0 = np.zeros(n_verts, dtype=np.float32)
    # 衬衫体积：躯干 18mm，袖管 15mm，带自然垂坠波纹，袖口与下摆先平滑过渡
    drape = 0.0022 * np.sin(rx * 0.42 + ry * 0.18) + 0.0015 * np.cos(ry * 0.35 - rz * 0.28)
    shirt_amp = np.where(ax < 16.5, 0.0185, 0.0150) + drape
    disp0[is_shirt] = np.clip(shirt_amp[is_shirt], 0.006, 0.024)
    # 短裤体积：9.5mm
    disp0[is_shorts] = 0.0095

    disp_smooth = smooth_scalar_on_mesh(disp0, faces, iters=16, alpha=0.65)
    # 严格锁定头颈、前臂、手腕、手指、小腿与双脚外推量为 0（彻底根除手腕 24mm 鼓包）
    strict_zero = np.clip(w_forearm_hand * 1.5 + w_head_neck * 1.5 + w_lowleg * 1.5 + w_foot * 1.5, 0.0, 1.0)
    vertex_inflate = (disp_smooth * (1.0 - strict_zero)).astype(np.float32)
    vertex_inflate[vertex_inflate < 0.0004] = 0.0

    wrist_max_mm = float(vertex_inflate[w_forearm_hand > 0.5].max()) * 1000.0
    print(f"[semantic] 顶点数={n_verts}  皮肤={(cloth_region==0).mean():.1%}  "
          f"衬衫={(cloth_region==1).mean():.1%}  短裤={(cloth_region==2).mean():.1%}  "
          f"鞋袜={(cloth_region==3).mean():.1%}")
    print(f"[inflate] 最大外推={float(vertex_inflate.max())*1000:.1f}mm  "
          f"前臂/手腕最大外推={wrist_max_mm:.3f}mm (目标=0.000mm)")

    # 验证精确骨骼头部局部系（Blender 坐标：(x_sam, z_sam, -y_sam)）
    def to_bl(p_sam):
        return np.array([p_sam[0], p_sam[2], -p_sam[1]], dtype=np.float64)

    p_head = to_bl(j0[jn.index("c_head")])
    p_top = to_bl(j0[jn.index("c_head_null")])
    p_reye = to_bl(j0[jn.index("r_eye")])
    p_leye = to_bl(j0[jn.index("l_eye")])

    e_up = p_top - p_head
    e_up /= np.linalg.norm(e_up)
    e_right = p_leye - p_reye  # 角色右眼(-X)指向左眼(+X)，即 Blender +X 方向
    e_right = e_right - e_up * np.dot(e_right, e_up)
    e_right /= np.linalg.norm(e_right)
    e_fwd = np.cross(e_up, e_right)  # +Z × +X = +Y? 注意右手系：right(+X), fwd(-Y), up(+Z) => right × fwd = up => fwd = up × right 指向 +Y，故取 cross(right, up) 指向人物前方(-Y)!
    e_fwd = np.cross(e_right, e_up)
    e_fwd /= np.linalg.norm(e_fwd)
    # 校验朝向相机（相机在 Blender -Y 方向，故 e_fwd.y 应 < 0）
    if e_fwd[1] > 0:
        e_fwd = -e_fwd
    head_center = 0.52 * p_head + 0.48 * (0.5 * (p_reye + p_leye))

    tilt_deg = float(np.degrees(np.arccos(np.clip(e_up[2], -1.0, 1.0))))
    ortho_err = float(abs(np.dot(e_up, e_right)) + abs(np.dot(e_up, e_fwd)) + abs(np.dot(e_right, e_fwd)))
    print(f"[head_rig] center={head_center.round(4).tolist()}  tilt_from_Z={tilt_deg:.2f}°  ortho_err={ortho_err:.2e}")
    print(f"           right={e_right.round(4).tolist()}  fwd={e_fwd.round(4).tolist()}  up={e_up.round(4).tolist()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        v=v0,
        faces=faces,
        tex=tex,
        tfaces=tfaces,
        rest_v=rest,
        pred_cam_t=np.asarray(d["pred_cam_t"]).reshape(-1)[:3].astype(np.float32),
        focal_length=np.asarray(d["focal_length"]).reshape(-1)[:1].astype(np.float32),
        orig_img_size=np.asarray(d.get("orig_img_size", [1402, 1122]), dtype=np.int32),
        pred_joint_coords=j0,
        pred_global_rots=np.asarray(d["pred_global_rots"], dtype=np.float32),
        joint_names=np.array(jn),
        cloth_region=cloth_region,
        is_face_core=is_face_core.astype(np.uint8),
        is_oral_eye=is_oral_eye.astype(np.uint8),
        vertex_inflate=vertex_inflate,
        head_center=head_center.astype(np.float32),
        head_right=e_right.astype(np.float32),
        head_fwd=e_fwd.astype(np.float32),
        head_up=e_up.astype(np.float32),
    )

    meta = {
        "verts": int(n_verts),
        "faces": int(len(faces)),
        "head_tilt_deg": round(tilt_deg, 2),
        "head_ortho_err": ortho_err,
        "max_inflate_mm": round(float(vertex_inflate.max()) * 1000.0, 2),
        "wrist_inflate_mm": round(wrist_max_mm, 4),
        "region_ratios": {
            "skin": round(float((cloth_region == 0).mean()), 4),
            "shirt": round(float((cloth_region == 1).mean()), 4),
            "shorts": round(float((cloth_region == 2).mean()), 4),
            "sock_shoe": round(float((cloth_region == 3).mean()), 4),
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] 导出 {out_path}")


if __name__ == "__main__":
    main()
