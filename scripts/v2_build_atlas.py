"""M2 (envs/mhr)：屏幕空间 Z-Buffer 反投影 + 3D 连续解剖语义引导的角色图集合成。

解决核心缺陷（对应 OPTIMIZATION_SPEC.md M2 / RC-1, RC-2）：
1. 矢量化 UV 重心坐标反投影 + 逐纹素屏幕空间 Z-Buffer 遮挡剔除（耗时从 3.4 分钟降至 ~4 秒），
   彻底消除双臂/比耶手指穿透印到胸肩、小腿穿透到大腿的伪影。
2. 3D 纯皮肤保护区（头面部含额头、颈、前臂、双手全指、大腿中下段、小腿）：
   额头刘海黑发由色度闸门剔除并用真实肤色场平滑填充；面部五官核心区（is_face_core）保留瞳孔/睫毛/唇齿高清细节；
   四肢正面与侧面采用法向入射角余弦软权重 + 40px 距离羽化过渡，消除掠射黑边与小腿明暗补丁分界。
3. 连续重心坐标 T-pose 衣物边界 + 四向镜像无缝平铺布料纹理（印花短袖 + 水洗牛仔短裤），
   消除三角面锯齿衣摆、V 领误切与背部 Voronoi 深棕拼布。
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


def build_screen_zbuffer(v_cam: np.ndarray, faces: np.ndarray, is_oral_eye: np.ndarray,
                         f: float, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """光栅化屏幕空间浮点深度缓冲 zbuf(H, W) 及深度跳变边缘掩码。"""
    cx, cy = w / 2.0, h / 2.0
    z = np.clip(v_cam[:, 2], 1e-3, None)
    uv2d = np.stack([f * v_cam[:, 0] / z + cx, f * v_cam[:, 1] / z + cy], axis=1)

    fn = np.cross(v_cam[faces[:, 1]] - v_cam[faces[:, 0]], v_cam[faces[:, 2]] - v_cam[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    cen = v_cam[faces].mean(axis=1)
    view = -cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    facing = (fn * view).sum(axis=1) > -0.05
    oral_face = is_oral_eye[faces].all(axis=1)
    valid_f = np.where(facing & (~oral_face))[0]

    z_tri = v_cam[faces[valid_f], 2].min(axis=1)
    order = valid_f[np.argsort(-z_tri)]

    zbuf = np.full((h, w), 100.0, dtype=np.float32)
    pts_all = np.round(uv2d).astype(np.int32)
    for fi in order:
        pts = pts_all[faces[fi]]
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= w or pts[:, 1].max() < 0 or pts[:, 1].min() >= h:
            continue
        cv2.fillConvexPoly(zbuf, pts, float(v_cam[faces[fi], 2].mean()))

    k5 = np.ones((5, 5), np.uint8)
    z_near = cv2.erode(zbuf, k5)
    z_far = cv2.dilate(zbuf, k5)
    depth_jump = ((z_far - z_near) > 0.032) & (z_far < 50.0)
    return z_near, depth_jump


def make_seamless_fabric(img_bgr: np.ndarray, mask_2d: np.ndarray, tile_px: int = 280,
                         fallback_bgr=(145, 148, 158)) -> np.ndarray:
    """从原图纯净衣物掩码提取最大内接正方形块（0.66*R < R/sqrt(2) 严防四角越界混入肤色），构造四向镜像无缝布料。"""
    ys, xs = np.where(mask_2d > 0)
    if len(ys) < 300:
        med = np.array(fallback_bgr, dtype=np.uint8)
        return np.full((tile_px * 2, tile_px * 2, 3), med, dtype=np.uint8)

    dt = cv2.distanceTransform(mask_2d.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    # 严格受限于真实内接圆半径 dt*0.64（不设过高下限，严防窄条牛仔区越界采到大腿肤色）
    r = max(6, min(int(dt[cy, cx] * 0.64), tile_px // 2))
    y0, y1 = max(0, cy - r), min(img_bgr.shape[0], cy + r)
    x0, x1 = max(0, cx - r), min(img_bgr.shape[1], cx + r)
    patch = img_bgr[y0:y1, x0:x1].astype(np.float32)

    lum = 0.114 * patch[..., 0] + 0.587 * patch[..., 1] + 0.299 * patch[..., 2]
    low = np.maximum(cv2.GaussianBlur(lum, (0, 0), max(r / 2.0, 10.0)), 1.0)
    patch = np.clip(patch * (float(np.median(lum)) / low)[..., None], 0, 255).astype(np.uint8)
    patch = cv2.resize(patch, (tile_px, tile_px), interpolation=cv2.INTER_CUBIC)

    top = np.concatenate([patch, patch[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    return full


def tile_to_canvas(tile: np.ndarray, s: int) -> np.ndarray:
    th, tw = tile.shape[:2]
    ny = (s + th - 1) // th
    nx = (s + tw - 1) // tw
    return np.tile(tile, (ny, nx, 1))[:s, :s].copy()


def fast_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """大核高斯金字塔加速：sigma>14 时先按比例降采样模糊再双线性上采样。"""
    if sigma <= 14.0:
        return cv2.GaussianBlur(arr, (0, 0), sigma)
    h, w = arr.shape[:2]
    factor = max(2, int(sigma // 10.0))
    sh, sw = max(8, h // factor), max(8, w // factor)
    small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    small_b = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small_b, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=r"work\q004-base\v2\mesh_lod0_v2.npz")
    ap.add_argument("--image", default=r"work\q004-base\scene\gpen_2x.png")
    ap.add_argument("--person", default=r"work\q004-base\v2\person_2x.png")
    ap.add_argument("--out", default=r"work\q004-base\v2\tex")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--facing-thr", type=float, default=0.30)
    ap.add_argument("--z-tol", type=float, default=0.015)
    ap.add_argument("--delight", type=float, default=0.68)
    ap.add_argument("--face-delight", type=float, default=0.12)
    args = ap.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = args.size

    dm = np.load(args.mesh, allow_pickle=True)
    v = dm["v"].astype(np.float32)
    faces = dm["faces"].astype(np.int32)
    tex = dm["tex"].astype(np.float32)
    tfaces = dm["tfaces"].astype(np.int32)
    rest_v = dm["rest_v"].astype(np.float32)
    cam_t = dm["pred_cam_t"].astype(np.float32)
    f_base = float(dm["focal_length"][0])
    w_orig, h_orig = int(dm["orig_img_size"][0]), int(dm["orig_img_size"][1])
    cloth_region = dm["cloth_region"].astype(np.uint8)
    is_face_core = dm["is_face_core"].astype(bool)
    is_oral_eye = dm["is_oral_eye"].astype(bool)

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    scale = w / float(w_orig)
    f = f_base * scale
    cx, cy = w / 2.0, h / 2.0

    person = (cv2.imread(args.person, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    # 身体与四肢轮廓向内收缩 19px（2x 分辨率下约 9.5px），彻底剥离小腿/手臂外侧地毯与阴影掠射黑边
    person_eroded_body = cv2.erode(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    person_eroded_fine = cv2.erode(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # 1. 计算顶点法线与屏幕空间 Z-Buffer
    v_cam = v + cam_t[None, :]
    vn = np.zeros_like(v_cam)
    fn_raw = np.cross(v_cam[faces[:, 1]] - v_cam[faces[:, 0]], v_cam[faces[:, 2]] - v_cam[faces[:, 0]])
    np.add.at(vn, faces[:, 0], fn_raw)
    np.add.at(vn, faces[:, 1], fn_raw)
    np.add.at(vn, faces[:, 2], fn_raw)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9

    z_near, depth_jump = build_screen_zbuffer(v_cam, faces, is_oral_eye, f, w, h)
    print(f"[zbuf] 屏幕空间深度缓冲完成 ({w}x{h}), 耗时 {time.time() - t_start:.2f}s")

    # 2. 在 UV 空间光栅化全部 14.7 万个三角面索引 fid_map
    uv_px = np.stack([tex[:, 0] * (S - 1), (1.0 - tex[:, 1]) * (S - 1)], axis=1).astype(np.float32)
    fid_map = np.zeros((S, S), dtype=np.float64)
    tri_uv_int = np.round(uv_px[tfaces]).astype(np.int32)
    for fi in range(len(faces)):
        cv2.fillConvexPoly(fid_map, tri_uv_int[fi], float(fi + 1))

    ys, xs = np.where(fid_map > 0.5)
    fids = fid_map[ys, xs].astype(np.int32) - 1
    uv_covered = (fid_map > 0.5)

    # 3. 矢量化计算全部有效纹素的重心坐标 -> 连续 3D 坐标、法线与 T-pose 解剖坐标
    tf_sel = tfaces[fids]
    A = uv_px[tf_sel[:, 0]]
    B = uv_px[tf_sel[:, 1]]
    C = uv_px[tf_sel[:, 2]]
    P = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    v0_2d = B - A
    v1_2d = C - A
    v2_2d = P - A
    den = v0_2d[:, 0] * v1_2d[:, 1] - v1_2d[:, 0] * v0_2d[:, 1]
    inv_den = np.where(np.abs(den) > 1e-8, 1.0 / den, 0.0)
    b1 = np.clip((v2_2d[:, 0] * v1_2d[:, 1] - v1_2d[:, 0] * v2_2d[:, 1]) * inv_den, 0.0, 1.0)
    b2 = np.clip((v0_2d[:, 0] * v2_2d[:, 1] - v0_2d[:, 1] * v2_2d[:, 0]) * inv_den, 0.0, 1.0)
    b0 = np.clip(1.0 - b1 - b2, 0.0, 1.0)
    bsum = np.maximum(b0 + b1 + b2, 1e-6)
    b0, b1, b2 = b0 / bsum, b1 / bsum, b2 / bsum

    vf_sel = faces[fids]
    p3d = (b0[:, None] * v_cam[vf_sel[:, 0]] +
           b1[:, None] * v_cam[vf_sel[:, 1]] +
           b2[:, None] * v_cam[vf_sel[:, 2]])
    n3d = (b0[:, None] * vn[vf_sel[:, 0]] +
           b1[:, None] * vn[vf_sel[:, 1]] +
           b2[:, None] * vn[vf_sel[:, 2]])
    n3d /= np.linalg.norm(n3d, axis=1, keepdims=True) + 1e-9
    rest_3d = (b0[:, None] * rest_v[vf_sel[:, 0]] +
               b1[:, None] * rest_v[vf_sel[:, 1]] +
               b2[:, None] * rest_v[vf_sel[:, 2]])

    view_dir = -p3d / (np.linalg.norm(p3d, axis=1, keepdims=True) + 1e-9)
    cos_theta = (n3d * view_dir).sum(axis=1)

    proj_u = f * p3d[:, 0] / np.maximum(p3d[:, 2], 1e-3) + cx
    proj_v = f * p3d[:, 1] / np.maximum(p3d[:, 2], 1e-3) + cy

    in_img = (proj_u >= 2) & (proj_u < w - 3) & (proj_v >= 2) & (proj_v < h - 3)
    ui = np.clip(np.round(proj_u).astype(np.int32), 0, w - 1)
    vi = np.clip(np.round(proj_v).astype(np.int32), 0, h - 1)

    oral_tex = (is_oral_eye[vf_sel[:, 0]] | is_oral_eye[vf_sel[:, 1]] | is_oral_eye[vf_sel[:, 2]])
    face_core_tex = (is_face_core[vf_sel[:, 0]] | is_face_core[vf_sel[:, 1]] | is_face_core[vf_sel[:, 2]])
    is_hand_tex = np.abs(rest_3d[:, 0]) > 56.0
    fine_sil_tex = face_core_tex | is_hand_tex

    z_buf_val = z_near[vi, ui]
    on_jump = depth_jump[vi, ui]
    in_fg = np.where(fine_sil_tex, person_eroded_fine[vi, ui] > 0, person_eroded_body[vi, ui] > 0)

    unoccluded = (p3d[:, 2] - z_buf_val) < np.where(oral_tex, 0.035, args.z_tol)
    unoccluded &= (~on_jump) | face_core_tex | (np.abs(p3d[:, 2] - z_buf_val) < 0.003)
    facing_ok = cos_theta > np.where(face_core_tex, 0.05, np.where(is_hand_tex, 0.16, args.facing_thr))
    valid_obs_1d = in_img & in_fg & facing_ok & unoccluded

    # 双线性重采样原图高清颜色到 UV 画布
    map_x = np.zeros((S, S), dtype=np.float32)
    map_y = np.zeros((S, S), dtype=np.float32)
    map_x[ys, xs] = proj_u
    map_y[ys, xs] = proj_v
    warped_atlas = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    valid_obs = np.zeros((S, S), dtype=bool)
    valid_obs[ys, xs] = valid_obs_1d

    uv_cos = np.zeros((S, S), dtype=np.float32)
    uv_cos[ys, xs] = np.clip(cos_theta, 0.0, 1.0)

    uv_face_core = np.zeros((S, S), dtype=bool)
    uv_face_core[ys, xs] = face_core_tex

    # 连续亚像素级 3D T-pose 解剖区域（彻底消除逐三角面光栅化的锯齿衣摆/裤边与 V 领缺口）
    uv_rx = np.zeros((S, S), dtype=np.float32)
    uv_ry = np.zeros((S, S), dtype=np.float32)
    uv_rz = np.zeros((S, S), dtype=np.float32)
    uv_rx[ys, xs] = rest_3d[:, 0]
    uv_ry[ys, xs] = rest_3d[:, 1]
    uv_rz[ys, xs] = rest_3d[:, 2]
    abs_rx = np.abs(uv_rx)

    # 精确连续 3D 衣物掩码（与 v2_prepare_mesh.py 解剖标定一致，但全连续无锯齿）
    # 锁骨正中窄 V 领开口（仅保留颈下正中裸露胸口皮肤，绝不误切两侧粉白翻领）
    vneck_open = (uv_rz > 5.0) & (uv_ry > 132.5) & (abs_rx < np.maximum(0.0, (uv_ry - 132.5) * 0.36))
    neck_open = (uv_ry > 142.2) & (abs_rx < 6.5)
    shirt_3d = uv_covered & (uv_ry >= 117.8) & (uv_ry <= 143.8) & (abs_rx <= 29.2) & (~vneck_open) & (~neck_open)

    shorts_hem_y = 79.5 + 0.42 * np.clip(abs_rx - 4.0, 0.0, 14.0)
    shorts_3d = uv_covered & (uv_ry >= shorts_hem_y) & (uv_ry < 95.8) & (abs_rx < 19.5)

    sock_3d = uv_covered & (uv_ry < 23.8) & (uv_ry > 0.01)

    # 同时保留顶点蒙皮权重的平滑兜底（防止腋下极个别点越界到前臂）
    reg_shirt_w = ((cloth_region[vf_sel[:, 0]] == 1).astype(np.float32) * b0 +
                   (cloth_region[vf_sel[:, 1]] == 1).astype(np.float32) * b1 +
                   (cloth_region[vf_sel[:, 2]] == 1).astype(np.float32) * b2)
    uv_shirt_ok = np.zeros((S, S), dtype=bool)
    uv_shirt_ok[ys, xs] = (reg_shirt_w > 0.05) | (np.abs(rest_3d[:, 0]) < 24.0)
    shirt_3d &= uv_shirt_ok

    print(f"[uv] 覆盖率={uv_covered.mean():.2%}  无遮挡有效观测={valid_obs.mean():.2%} "
          f"(占总UV {valid_obs.sum()/max(uv_covered.sum(),1):.1%})")

    # ------------------------------------------------------------------
    # 4. 构建纯净全身皮肤图集 skin_out（无刘海黑斑、无掠射黑边、无小腿色块断层）
    # ------------------------------------------------------------------
    b = warped_atlas[..., 0].astype(np.float32)
    g = warped_atlas[..., 1].astype(np.float32)
    r = warped_atlas[..., 2].astype(np.float32)
    lum_raw = (b + g + r) / 3.0
    warm_skin = ((r - b) > 20.0) & (r > g + 4.0) & (lum_raw > 86.0) & (lum_raw < 222.0)

    pure_skin_3d = uv_covered & (~shirt_3d) & (~shorts_3d) & (~sock_3d)
    seed_body = valid_obs & pure_skin_3d & (~uv_face_core) & warm_skin
    seed_body = cv2.morphologyEx(seed_body.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(seed_body.astype(np.uint8), connectivity=8)
    keep_cc = np.zeros(nlab, dtype=bool)
    for idx in range(1, nlab):
        if stats[idx, cv2.CC_STAT_AREA] >= int(S * S * 0.0003):
            keep_cc[idx] = True
    seed_body = keep_cc[lab] if nlab > 1 else seed_body

    # 仅在真实眼眶/眉毛内侧 (|rx|<5.0) 与口唇 (|rx|<3.8) 允许保留深色瞳孔睫毛，太阳穴与脸颊外侧严格剔除碎黑发
    inner_features = uv_face_core & (
        ((abs_rx < 5.0) & (uv_ry >= 159.2) & (uv_ry <= 163.5)) |
        ((abs_rx < 3.9) & (uv_ry >= 150.5) & (uv_ry <= 156.8))
    )
    cheek_nose_skin = uv_face_core & (~inner_features) & ((r - b) > 14.0) & (lum_raw > 78.0)
    seed_face = valid_obs & ((inner_features & (lum_raw > 18.0)) | cheek_nose_skin)
    seed_skin = seed_body | seed_face
    print(f"[skin] 有效皮肤种子={seed_skin.mean():.2%} (五官核心={seed_face.mean():.2%})")

    med_skin = np.median(warped_atlas[seed_body].reshape(-1, 3), axis=0).astype(np.float32)
    med_lum = max(float(0.114 * med_skin[0] + 0.587 * med_skin[1] + 0.299 * med_skin[2]), 1.0)

    # 先对观测皮肤本身做柔和匀光，压平正面机顶闪光灯在小腿迎光面造成的亮白条带
    obs_skin_src = warped_atlas.astype(np.float32).copy()
    y_obs = 0.114 * obs_skin_src[..., 0] + 0.587 * obs_skin_src[..., 1] + 0.299 * obs_skin_src[..., 2]
    # 仅用种子内部的加权均值估计局部低频光照，避免种子外黑像素拉低边缘
    sf_body = seed_body.astype(np.float32)
    y_num = fast_gaussian_blur(y_obs * sf_body, S / 38.0)
    y_den = fast_gaussian_blur(sf_body, S / 38.0)
    y_local_mean = np.where(y_den > 0.05, y_num / np.maximum(y_den, 1e-4), med_lum)
    delight_ratio = np.clip(med_lum / np.maximum(y_local_mean, 40.0), 0.72, 1.32)

    face_w = fast_gaussian_blur(uv_face_core.astype(np.float32), S / 170.0)
    dl_strength = args.delight * (1.0 - face_w) + args.face_delight * face_w
    obs_delit = obs_skin_src * (1.0 - dl_strength[..., None] + dl_strength[..., None] * delight_ratio[..., None])
    obs_delit = np.clip(obs_delit, 0.0, 255.0)

    # 构造平滑背景肤色场（额头刘海下方、四肢侧面与背面）
    field_src = obs_delit.copy()
    field_src[seed_face & (lum_raw < 88.0)] = med_skin

    sf = seed_skin.astype(np.float32)
    sig1 = S / 34.0
    sig2 = S / 10.0
    w1_g = fast_gaussian_blur(sf, sig1)
    w2_g = fast_gaussian_blur(sf, sig2)
    c1_g = np.stack([fast_gaussian_blur(field_src[..., i] * sf, sig1) for i in range(3)], axis=-1)
    c2_g = np.stack([fast_gaussian_blur(field_src[..., i] * sf, sig2) for i in range(3)], axis=-1)
    multi_c = (c1_g + 0.40 * c2_g + med_skin * 0.025) / (w1_g[..., None] + 0.40 * w2_g[..., None] + 0.025)

    y_multi = 0.114 * multi_c[..., 0] + 0.587 * multi_c[..., 1] + 0.299 * multi_c[..., 2]
    lo_p, hi_p = np.percentile(y_multi[seed_body], [15, 90])
    skin_bg_field = (med_skin / med_lum)[None, None, :] * np.clip(y_multi, lo_p, hi_p)[..., None]

    # 连续软融合权重：法向入射角余弦软斜坡 × 种子内部 42px 距离羽化（五官核心区全权重 1.0 保真）
    dist_in = ndimage.distance_transform_edt(seed_skin).astype(np.float32)
    w_dist = np.clip(dist_in / (S / 96.0), 0.0, 1.0)
    w_cos = np.clip((uv_cos - 0.28) / 0.28, 0.0, 1.0)
    w_obs_skin = np.where(uv_face_core & seed_face, 1.0, w_dist * w_cos)
    w_obs_skin = cv2.GaussianBlur(w_obs_skin.astype(np.float32), (0, 0), 5.0)[..., None]

    skin_out = obs_delit * w_obs_skin + skin_bg_field * (1.0 - w_obs_skin)

    # 五官微锐化还原睫毛与唇齿清晰度
    blur_face = cv2.GaussianBlur(skin_out, (0, 0), 1.8)
    sharp_face = np.clip(skin_out + 0.48 * (skin_out - blur_face), 0, 255)
    skin_out = np.clip(skin_out * (1.0 - face_w[..., None]) + sharp_face * face_w[..., None], 0, 255)

    # ------------------------------------------------------------------
    # 5. 构建衬衫与水洗牛仔短裤图集（纯净采样 + 正面保真 + 背面无缝镜像布料纹理）
    # ------------------------------------------------------------------
    img_b = img[..., 0].astype(np.float32)
    img_g = img[..., 1].astype(np.float32)
    img_r = img[..., 2].astype(np.float32)
    img_lum = (img_b + img_g + img_r) / 3.0

    # 2D 纯净衣物掩码：严加色度门控（水洗牛仔 R-B < 20，彻底杜绝混入大腿暖肤 R-B ≈ 62！）
    shirt_2d = np.zeros((h, w), dtype=np.uint8)
    shorts_2d = np.zeros((h, w), dtype=np.uint8)
    shirt_3d_1d = shirt_3d[ys, xs]
    shorts_3d_1d = shorts_3d[ys, xs]

    obs_shirt_pts = valid_obs_1d & shirt_3d_1d
    obs_shorts_pts = valid_obs_1d & shorts_3d_1d
    shirt_2d[vi[obs_shirt_pts], ui[obs_shirt_pts]] = 1
    shorts_2d[vi[obs_shorts_pts], ui[obs_shorts_pts]] = 1

    shirt_2d = cv2.morphologyEx(shirt_2d, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    shorts_2d = cv2.morphologyEx(shorts_2d, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    pure_shirt_2d = (shirt_2d > 0) & ((img_r - img_b) < 34.0) & (img_lum > 72.0) & (img_lum < 218.0)
    # 拍立得暖黄偏色下水洗浅蓝牛仔布 R-B≈10..35 (B>112)，而裸露大腿暖肤 R-B≈62
    pure_shorts_2d = (shorts_2d > 0) & ((img_r - img_b) < 38.0) & (img_b > 110.0) & (img_lum > 100.0) & (img_lum < 205.0)
    pure_shirt_2d = cv2.erode(pure_shirt_2d.astype(np.uint8), np.ones((3, 3), np.uint8))
    pure_shorts_2d = cv2.erode(pure_shorts_2d.astype(np.uint8), np.ones((3, 3), np.uint8))

    shirt_tile = make_seamless_fabric(img, pure_shirt_2d, tile_px=260, fallback_bgr=(152, 154, 172))
    shorts_tile = make_seamless_fabric(img, pure_shorts_2d, tile_px=200, fallback_bgr=(148, 144, 148))
    shirt_bg = tile_to_canvas(shirt_tile, S).astype(np.float32)
    shorts_bg = tile_to_canvas(shorts_tile, S).astype(np.float32)

    # 正面候选衣物区：允许前襟衣摆小幅延伸贴合原图，同时保证 3D 衣物网格内部 100% 为衣物纹理（绝不漏出皮肤斑块）
    in_shirt_cand = (uv_ry >= 112.5) & (uv_ry <= 144.2) & (abs_rx <= 30.2) & (~uv_face_core)
    in_shorts_cand = (uv_ry >= 77.5) & (uv_ry <= 97.5) & (abs_rx <= 19.8) & (~uv_face_core)

    not_bare_skin_sh = ~(((r - b) > 36.0) & ((g - b) > 15.0) & (lum_raw < 178.0))
    is_shirt_pix = (((r - b) < 28.0) & (lum_raw > 65.0)) | (shirt_3d & not_bare_skin_sh & (lum_raw > 62.0))
    is_denim_pix = ((r - b) < 38.0) & (b > 105.0) & (lum_raw > 85.0) & (lum_raw < 210.0)

    obs_shirt = valid_obs & in_shirt_cand & is_shirt_pix
    obs_shorts = valid_obs & in_shorts_cand & is_denim_pix

    cloth_photo = warped_atlas.astype(np.float32)
    yc = 0.114 * cloth_photo[..., 0] + 0.587 * cloth_photo[..., 1] + 0.299 * cloth_photo[..., 2]
    yc_low = np.maximum(fast_gaussian_blur(yc, S / 20.0), 1.0)

    if obs_shirt.any():
        med_sh_bgr = np.median(cloth_photo[obs_shirt].reshape(-1, 3), axis=0)
        bg_med = np.maximum(np.median(shirt_bg.reshape(-1, 3), axis=0), 1.0)
        ratio_sh = np.clip(med_sh_bgr / bg_med, 0.80, 1.25)
        shirt_bg = np.clip(shirt_bg * ratio_sh[None, None, :], 0, 255)

        med_yc = float(np.median(yc[obs_shirt]))
        shirt_flat = np.clip(cloth_photo * np.clip(med_yc / yc_low, 0.78, 1.25)[..., None], 0, 255)
        shirt_obs_col = 0.80 * cloth_photo + 0.20 * shirt_flat
    else:
        shirt_obs_col = shirt_bg

    if obs_shorts.any():
        med_dn_bgr = np.median(cloth_photo[obs_shorts].reshape(-1, 3), axis=0)
        bg_med2 = np.maximum(np.median(shorts_bg.reshape(-1, 3), axis=0), 1.0)
        ratio_dn = np.clip(med_dn_bgr / bg_med2, 0.85, 1.18)
        shorts_bg = np.clip(shorts_bg * ratio_dn[None, None, :], 0, 255)

    w_obs_sh = cv2.GaussianBlur(obs_shirt.astype(np.float32), (0, 0), S / 280.0)[..., None]
    shirt_layer = shirt_obs_col * w_obs_sh + shirt_bg * (1.0 - w_obs_sh)

    # 短裤正面高保真保留纽扣、门襟与口袋缝线细节，仅在边缘 8px 柔化过渡到背面无缝牛仔纹
    w_obs_dn = cv2.GaussianBlur(obs_shorts.astype(np.float32), (0, 0), 8.0)[..., None]
    shorts_layer = cloth_photo * w_obs_dn + shorts_bg * (1.0 - w_obs_dn)
    # 轻微校正拍立得暖黄偏色（还原清透水洗浅蓝牛仔质感）
    denim_tint = np.array([1.08, 1.01, 0.91], dtype=np.float32)[None, None, :]
    shorts_layer = np.clip(shorts_layer * denim_tint, 0, 255)

    # ------------------------------------------------------------------
    # 6. 按连续 3D 解剖语义合成最终图集（衣物网格 100% 饱满覆盖、衣摆平滑抗锯齿）
    # ------------------------------------------------------------------
    final_shirt_mask = shirt_3d | obs_shirt
    final_shorts_mask = shorts_3d | obs_shorts
    final_sock_mask = sock_3d

    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m_shirt_bin = cv2.morphologyEx(final_shirt_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_clean)
    m_shorts_bin = cv2.morphologyEx(final_shorts_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_clean)

    m_shirt = cv2.GaussianBlur(m_shirt_bin.astype(np.float32), (0, 0), 3.2)[..., None]
    m_shorts = cv2.GaussianBlur(m_shorts_bin.astype(np.float32), (0, 0), 3.2)[..., None]
    m_sock = cv2.GaussianBlur(final_sock_mask.astype(np.float32), (0, 0), 2.5)[..., None]

    sock_col = np.full_like(skin_out, [206.0, 212.0, 220.0])

    comp = skin_out.copy()
    comp = comp * (1.0 - m_shirt) + shirt_layer * m_shirt
    comp = comp * (1.0 - m_shorts) + shorts_layer * m_shorts
    comp = comp * (1.0 - m_sock) + sock_col * m_sock
    comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)

    # UV 岛外扩填充（Gutter Dilation），彻底消除 Blender Mipmap / Smart 插值黑边
    _, nearest_idx = ndimage.distance_transform_edt(~uv_covered, return_distances=True, return_indices=True)
    comp_padded = comp_u8[tuple(nearest_idx)]

    atlas_path = out_dir / "char_atlas_v2.png"
    cv2.imwrite(str(atlas_path), comp_padded)
    cv2.imwrite(str(out_dir / "skin_only_v2.png"), np.clip(skin_out, 0, 255).astype(np.uint8)[tuple(nearest_idx)])
    cv2.imwrite(str(out_dir / "valid_obs_mask.png"), valid_obs.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "shirt_tile_preview.png"), shirt_tile)
    cv2.imwrite(str(out_dir / "shorts_tile_preview.png"), shorts_tile)

    stats = {
        "atlas_size": S,
        "uv_coverage": round(float(uv_covered.mean()), 4),
        "valid_obs_ratio": round(float(valid_obs.mean()), 4),
        "skin_seed_ratio": round(float(seed_skin.mean()), 4),
        "median_skin_bgr": [round(float(x), 1) for x in med_skin],
        "median_shorts_tile_bgr": [round(float(x), 1) for x in np.median(shorts_tile.reshape(-1, 3), axis=0)],
        "median_shirt_tile_bgr": [round(float(x), 1) for x in np.median(shirt_tile.reshape(-1, 3), axis=0)],
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    (out_dir / "atlas_stats_v2.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[done] 合成完成 ({stats['elapsed_sec']}s) -> {atlas_path}")


if __name__ == "__main__":
    main()
