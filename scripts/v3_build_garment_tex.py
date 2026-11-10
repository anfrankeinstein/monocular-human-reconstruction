"""v3 独立壳体解耦贴图合成器：纯肤色身体图集 + 物理衬衫壳体图集 + 重工水洗牛仔短裤图集。

核心特性：
1. 身体与服装纹理 100% 物理隔离：
   - skin_body_v3.png：纯裸露皮肤纹理（清除锁骨领尖残留粉斑，完整保留串珠项链、五官与肚脐），杜绝任何布料渗色；
   - shirt_atlas_v3.png：直接向物理解算后的三维衬衫壳体（含外翻古巴领与宽松短袖）反投影原图印花，
     由 (身体 + 衬衫壳 + 短裤壳) 联合屏幕空间 Z-Buffer 精确剔除举手手臂遮挡，无缝衔接四向镜像印花布料；
   - shorts_atlas_v3.png：向物理解算后的高腰牛仔短裤壳体反投影正面水洗细节，叠加 45° 斜纹牛仔纱线肌理、
     腰头双明线、后臀育克线 (Back Yoke) 与对称贴袋车线、裤脚毛边白棉线须细节。
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


def fast_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 14.0:
        return cv2.GaussianBlur(arr, (0, 0), sigma)
    h, w = arr.shape[:2]
    factor = max(2, int(sigma // 10.0))
    sh, sw = max(8, h // factor), max(8, w // factor)
    small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    small_b = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small_b, (w, h), interpolation=cv2.INTER_LINEAR)


def compute_vn_cam(v_cam: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vn = np.zeros_like(v_cam)
    fn = np.cross(v_cam[faces[:, 1]] - v_cam[faces[:, 0]], v_cam[faces[:, 2]] - v_cam[faces[:, 0]])
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
    return vn


def build_combined_zbuffer(meshes_cam: list[tuple[np.ndarray, np.ndarray]], f: float, w: int, h: int):
    """将多个网格联合光栅化到同一屏幕空间深度缓冲，确保举起的前臂/手指准确遮挡后方衬衫壳体。"""
    cx, cy = w / 2.0, h / 2.0
    all_tris_2d = []
    all_z_mean = []
    all_z_min = []

    for v_cam, faces in meshes_cam:
        z = np.clip(v_cam[:, 2], 1e-3, None)
        uv2d = np.stack([f * v_cam[:, 0] / z + cx, f * v_cam[:, 1] / z + cy], axis=1)
        fn = np.cross(v_cam[faces[:, 1]] - v_cam[faces[:, 0]], v_cam[faces[:, 2]] - v_cam[faces[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
        cen = v_cam[faces].mean(axis=1)
        view = -cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
        valid = np.where((fn * view).sum(axis=1) > -0.08)[0]

        pts_int = np.round(uv2d).astype(np.int32)
        for fi in valid:
            tri_pts = pts_int[faces[fi]]
            if tri_pts[:, 0].max() < 0 or tri_pts[:, 0].min() >= w or tri_pts[:, 1].max() < 0 or tri_pts[:, 1].min() >= h:
                continue
            zs = v_cam[faces[fi], 2]
            all_tris_2d.append(tri_pts)
            all_z_mean.append(float(zs.mean()))
            all_z_min.append(float(zs.min()))

    order = np.argsort(-np.asarray(all_z_min))
    zbuf = np.full((h, w), 100.0, dtype=np.float32)
    for idx in order:
        cv2.fillConvexPoly(zbuf, all_tris_2d[idx], all_z_mean[idx])

    k5 = np.ones((5, 5), np.uint8)
    z_near = cv2.erode(zbuf, k5)
    z_far = cv2.dilate(zbuf, k5)
    depth_jump = ((z_far - z_near) > 0.032) & (z_far < 50.0)
    return z_near, depth_jump


def rasterize_uv_barycentric(v_cam: np.ndarray, rest_v: np.ndarray, uv: np.ndarray,
                             faces: np.ndarray, tfaces: np.ndarray, s: int):
    """矢量化光栅化任意壳体网格到 SxS UV 空间，输出逐纹素 3D 相机坐标、法线与解剖坐标。"""
    vn = compute_vn_cam(v_cam, faces)
    uv_px = np.stack([uv[:, 0] * (s - 1), (1.0 - uv[:, 1]) * (s - 1)], axis=1).astype(np.float32)
    fid_map = np.zeros((s, s), dtype=np.float64)
    tri_uv_int = np.round(uv_px[tfaces]).astype(np.int32)
    for fi in range(len(faces)):
        cv2.fillConvexPoly(fid_map, tri_uv_int[fi], float(fi + 1))

    ys, xs = np.where(fid_map > 0.5)
    fids = fid_map[ys, xs].astype(np.int32) - 1
    uv_cov = fid_map > 0.5

    tf_sel = tfaces[fids]
    A = uv_px[tf_sel[:, 0]]
    B = uv_px[tf_sel[:, 1]]
    C = uv_px[tf_sel[:, 2]]
    P = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    v0 = B - A
    v1 = C - A
    v2 = P - A
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    inv_den = np.where(np.abs(den) > 1e-8, 1.0 / den, 0.0)
    b1 = np.clip((v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) * inv_den, 0.0, 1.0)
    b2 = np.clip((v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) * inv_den, 0.0, 1.0)
    b0 = np.clip(1.0 - b1 - b2, 0.0, 1.0)
    bsum = np.maximum(b0 + b1 + b2, 1e-6)
    b0, b1, b2 = b0 / bsum, b1 / bsum, b2 / bsum

    vf_sel = faces[fids]
    p3d = b0[:, None] * v_cam[vf_sel[:, 0]] + b1[:, None] * v_cam[vf_sel[:, 1]] + b2[:, None] * v_cam[vf_sel[:, 2]]
    n3d = b0[:, None] * vn[vf_sel[:, 0]] + b1[:, None] * vn[vf_sel[:, 1]] + b2[:, None] * vn[vf_sel[:, 2]]
    n3d /= np.linalg.norm(n3d, axis=1, keepdims=True) + 1e-9
    r3d = b0[:, None] * rest_v[vf_sel[:, 0]] + b1[:, None] * rest_v[vf_sel[:, 1]] + b2[:, None] * rest_v[vf_sel[:, 2]]
    return ys, xs, vf_sel, p3d, n3d, r3d, uv_cov


def make_seamless_fabric(img_bgr: np.ndarray, mask_2d: np.ndarray, tile_px: int = 640,
                         fallback_bgr=(156, 158, 175)) -> np.ndarray:
    """非镜像环面余弦窗拼贴合成（Toroidal Hann Quilting）：100% 无缝平铺且彻底消除镜像十字对称伪影。"""
    dt = cv2.distanceTransform(mask_2d.astype(np.uint8), cv2.DIST_L2, 5)
    cand_y, cand_x = np.where(dt >= 18.0)
    if len(cand_y) < 40:
        return np.full((tile_px, tile_px, 3), np.array(fallback_bgr, dtype=np.uint8), dtype=np.uint8)

    # 先对整图做局部去阴影匀光，使采样的每块碎花布片底色一致
    img_f = img_bgr.astype(np.float32)
    lum = 0.114 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.299 * img_f[..., 2]
    m_f = (mask_2d > 0).astype(np.float32)
    lum_blur = cv2.GaussianBlur(lum * m_f, (0, 0), 28.0) / np.maximum(cv2.GaussianBlur(m_f, (0, 0), 28.0), 1e-3)
    med_l = float(np.median(lum[mask_2d > 0]))
    norm_img = np.clip(img_f * np.clip(med_l / np.maximum(lum_blur, 40.0), 0.78, 1.25)[..., None], 0, 255)
    med_bgr = np.median(norm_img[mask_2d > 0].reshape(-1, 3), axis=0).astype(np.float32)

    rng = np.random.default_rng(202608)
    patch_sz = 200
    win1d = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(patch_sz, dtype=np.float32) / float(patch_sz))
    win2d = (win1d[:, None] * win1d[None, :])[..., None]

    acc_col = np.ones((tile_px, tile_px, 3), dtype=np.float32) * (med_bgr[None, None, :] * 1e-2)
    acc_w = np.full((tile_px, tile_px, 1), 1e-2, dtype=np.float32)

    # 在 6x6 抖动网格上环面投递非镜像原图碎花图块（0 镜像翻转，彻底告别罗夏十字伪影）
    grid_n = 6
    step = tile_px // grid_n
    for gy in range(grid_n):
        for gx in range(grid_n):
            pick = int(rng.integers(0, len(cand_y)))
            cy, cx = int(cand_y[pick]), int(cand_x[pick])
            r = max(16, min(int(dt[cy, cx] * 0.82), 85))
            raw_p = norm_img[max(0, cy - r):min(norm_img.shape[0], cy + r),
                             max(0, cx - r):min(norm_img.shape[1], cx + r)]
            p_res = cv2.resize(raw_p, (patch_sz, patch_sz), interpolation=cv2.INTER_CUBIC)
            # 仅允许 180° 旋转（中心对称不产生轴对称十字纹）或原向
            if rng.random() < 0.35:
                p_res = np.rot90(p_res, 2)

            y0 = (gy * step + int(rng.integers(-step // 4, step // 4 + 1))) % tile_px
            x0 = (gx * step + int(rng.integers(-step // 4, step // 4 + 1))) % tile_px
            ys_idx = (y0 + np.arange(patch_sz)) % tile_px
            xs_idx = (x0 + np.arange(patch_sz)) % tile_px
            ix = np.ix_(ys_idx, xs_idx)
            acc_col[ix] += p_res * win2d
            acc_w[ix] += win2d

    out_tile = np.clip(acc_col / acc_w, 0, 255).astype(np.uint8)
    return out_tile


def tile_to_canvas(tile: np.ndarray, s: int) -> np.ndarray:
    th, tw = tile.shape[:2]
    return np.tile(tile, ((s + th - 1) // th, (s + tw - 1) // tw, 1))[:s, :s].copy()


def synthesize_denim_twill_canvas(s: int, base_bgr: np.ndarray) -> np.ndarray:
    """合成具有真实 45° 斜纹经纬竹节纱肌理的浅蓝水洗牛仔布底图。"""
    rng = np.random.default_rng(2026)
    tile_s = 512
    yy, xx = np.mgrid[0:tile_s, 0:tile_s].astype(np.float32)

    # 45° 经典右斜纹 (2/1 Twill) 周期纹理 + 经纱纵向竹节起伏 (Slub Yarn)
    diag = np.sin((xx + yy) * (2.0 * np.pi / 5.0)) * 3.8
    warp = cv2.GaussianBlur(rng.normal(0.0, 5.5, (tile_s // 4, tile_s)).astype(np.float32), (1, 9), 0)
    warp = cv2.resize(warp, (tile_s, tile_s), interpolation=cv2.INTER_LINEAR)
    weft = cv2.GaussianBlur(rng.normal(0.0, 3.2, (tile_s, tile_s // 4)).astype(np.float32), (9, 1), 0)
    weft = cv2.resize(weft, (tile_s, tile_s), interpolation=cv2.INTER_LINEAR)

    # 水洗微云斑
    wash = cv2.GaussianBlur(rng.normal(0.0, 14.0, (64, 64)).astype(np.float32), (0, 0), 3.0)
    wash = cv2.resize(wash, (tile_s, tile_s), interpolation=cv2.INTER_CUBIC)

    grain = (diag + warp + weft + wash)[..., None]
    tile = np.clip(base_bgr[None, None, :] + grain, 0, 255).astype(np.uint8)
    top = np.concatenate([tile, tile[:, ::-1]], axis=1)
    sym = np.concatenate([top, top[::-1, :]], axis=0)
    return tile_to_canvas(sym, s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=r"work\q004-base\v2\mesh_lod0_v2.npz")
    ap.add_argument("--sim", default=r"work\q004-base\v3\garments_sim_v3.npz")
    ap.add_argument("--image", default=r"work\q004-base\scene\gpen_2x.png")
    ap.add_argument("--person", default=r"work\q004-base\v2\person_2x.png")
    ap.add_argument("--out", default=r"work\q004-base\v3\tex")
    ap.add_argument("--size", type=int, default=4096)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = args.size

    dm = np.load(args.mesh, allow_pickle=True)
    ds = np.load(args.sim, allow_pickle=True)

    cam_t = dm["pred_cam_t"].astype(np.float32)
    f_base = float(dm["focal_length"][0])
    w_orig, h_orig = int(dm["orig_img_size"][0]), int(dm["orig_img_size"][1])

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    f = f_base * (w / float(w_orig))
    cx, cy = w / 2.0, h / 2.0

    person = (cv2.imread(args.person, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    person_body = cv2.erode(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    person_fine = cv2.erode(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    person_cloth = cv2.erode(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # 1. 构建联合屏幕空间 Z-Buffer（身体非眼口面 + 物理衬衫壳 + 物理短裤壳）
    v_body_cam = dm["v"].astype(np.float32) + cam_t[None, :]
    f_body = dm["faces"].astype(np.int32)
    is_oral_eye = dm["is_oral_eye"].astype(bool)
    f_body_ext = f_body[~is_oral_eye[f_body].all(axis=1)]

    v_sh_cam = ds["shirt_v_sam"].astype(np.float32) + cam_t[None, :]
    f_sh = ds["shirt_f"].astype(np.int32)
    v_dn_cam = ds["shorts_v_sam"].astype(np.float32) + cam_t[None, :]
    f_dn = ds["shorts_f"].astype(np.int32)

    # 先构建仅身体的 Z-buffer 供纯皮肤采样，再构建含服装壳体的联合 Z-buffer 供壳体采样
    z_body_only, jump_body = build_combined_zbuffer([(v_body_cam, f_body_ext)], f, w, h)
    z_all, jump_all = build_combined_zbuffer([(v_body_cam, f_body_ext), (v_sh_cam, f_sh), (v_dn_cam, f_dn)], f, w, h)
    print(f"[zbuf] 独立与联合深度缓冲完成 ({time.time() - t0:.2f}s)")

    # ==================================================================
    # A. 合成 100% 纯净无衣物污染的身体皮肤图集 skin_body_v3.png
    # ==================================================================
    ys_b, xs_b, vf_b, p3d_b, n3d_b, r3d_b, cov_b = rasterize_uv_barycentric(
        v_body_cam, dm["rest_v"].astype(np.float32), dm["tex"].astype(np.float32),
        f_body, dm["tfaces"].astype(np.int32), S
    )
    view_b = -p3d_b / (np.linalg.norm(p3d_b, axis=1, keepdims=True) + 1e-9)
    cos_b = (n3d_b * view_b).sum(axis=1)
    pu_b = f * p3d_b[:, 0] / np.maximum(p3d_b[:, 2], 1e-3) + cx
    pv_b = f * p3d_b[:, 1] / np.maximum(p3d_b[:, 2], 1e-3) + cy
    ui_b = np.clip(np.round(pu_b).astype(np.int32), 0, w - 1)
    vi_b = np.clip(np.round(pv_b).astype(np.int32), 0, h - 1)

    is_face_core = dm["is_face_core"].astype(bool)
    face_core_b = is_face_core[vf_b[:, 0]] | is_face_core[vf_b[:, 1]] | is_face_core[vf_b[:, 2]]
    oral_b = is_oral_eye[vf_b[:, 0]] | is_oral_eye[vf_b[:, 1]] | is_oral_eye[vf_b[:, 2]]
    hand_b = np.abs(r3d_b[:, 0]) > 56.0

    in_fg_b = np.where(face_core_b | hand_b, person_fine[vi_b, ui_b] > 0, person_body[vi_b, ui_b] > 0)
    unocc_b = ((p3d_b[:, 2] - z_body_only[vi_b, ui_b]) < np.where(oral_b, 0.035, 0.015)) & (
        (~jump_body[vi_b, ui_b]) | face_core_b | (np.abs(p3d_b[:, 2] - z_body_only[vi_b, ui_b]) < 0.003)
    )
    facing_b = cos_b > np.where(face_core_b, 0.05, np.where(hand_b, 0.16, 0.30))
    in_img_b = (pu_b >= 2) & (pu_b < w - 3) & (pv_b >= 2) & (pv_b < h - 3)
    valid_b_1d = in_img_b & in_fg_b & facing_b & unocc_b

    mx_b = np.zeros((S, S), dtype=np.float32); my_b = np.zeros((S, S), dtype=np.float32)
    mx_b[ys_b, xs_b] = pu_b; my_b[ys_b, xs_b] = pv_b
    warp_b = cv2.remap(img, mx_b, my_b, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    valid_b = np.zeros((S, S), dtype=bool); valid_b[ys_b, xs_b] = valid_b_1d
    uv_cos_b = np.zeros((S, S), dtype=np.float32); uv_cos_b[ys_b, xs_b] = np.clip(cos_b, 0.0, 1.0)
    uv_fc_b = np.zeros((S, S), dtype=bool); uv_fc_b[ys_b, xs_b] = face_core_b
    uv_sx_b = np.zeros((S, S), dtype=np.float32); uv_sx_b[ys_b, xs_b] = r3d_b[:, 0]
    uv_rx_b = np.abs(uv_sx_b)
    uv_ry_b = np.zeros((S, S), dtype=np.float32); uv_ry_b[ys_b, xs_b] = r3d_b[:, 1]
    uv_rz_b = np.zeros((S, S), dtype=np.float32); uv_rz_b[ys_b, xs_b] = r3d_b[:, 2]
    # 实拍露出的左眉 (ry∈[163.3, 165.0], sx∈[0.6, 5.1]) 在 MHR 默认 face_core 掩码之外，显式纳入面部保真区以保留清晰真眉！
    exposed_left_brow_3d = (uv_sx_b >= 0.6) & (uv_sx_b <= 5.1) & (uv_ry_b >= 163.3) & (uv_ry_b <= 165.0) & (uv_rz_b > 8.8)
    uv_fc_b = uv_fc_b | exposed_left_brow_3d

    # 严格排除服装内部覆盖区（与高腰牛仔裤前腰线 95.8cm 精确对齐，消除肚脐下方原图裤头阴影横线！）
    in_vneck_bare = (uv_rz_b > 4.5) & (uv_ry_b > 125.0) & (uv_rx_b < np.maximum(0.0, (uv_ry_b - 125.0) * 0.28))
    under_shirt = (uv_ry_b >= 116.5) & (uv_ry_b <= 145.5) & (uv_rx_b <= 30.5) & (~in_vneck_bare)
    under_shorts = (uv_ry_b >= 73.5) & (uv_ry_b <= 95.8) & (uv_rx_b <= 20.0)
    under_sock = uv_ry_b < 24.0

    bb, gg, rr = warp_b[..., 0].astype(np.float32), warp_b[..., 1].astype(np.float32), warp_b[..., 2].astype(np.float32)
    lum_b = (bb + gg + rr) / 3.0
    # 锁骨颈下保留彩色串珠项链，滤除下颌/颈侧重阴影斑块
    is_necklace_zone = in_vneck_bare & (uv_ry_b > 134.0) & (uv_ry_b < 143.5) & (uv_rx_b < 4.5) & (lum_b > 65.0) & (lum_b < 195.0)
    not_chin_shadow = ~((uv_ry_b > 138.0) & (uv_ry_b < 150.2) & (~is_necklace_zone) & (lum_b < 132.0))
    warm_skin = ((((rr - bb) > 22.0) & (rr > gg + 4.0) & (lum_b > 86.0) & (lum_b < 220.0)) | is_necklace_zone) & not_chin_shadow

    pure_bare_3d = cov_b & (~under_shirt) & (~under_shorts) & (~under_sock)
    seed_body = valid_b & pure_bare_3d & (~uv_fc_b) & warm_skin
    seed_body = cv2.morphologyEx(seed_body.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0

    # 完整保留双眼、口唇以及实拍露出的左眉，同时滤除上额 (ry>165.1) 与右额刘海暗影！
    exposed_left_brow = exposed_left_brow_3d & (lum_b > 38.0)
    inner_feat = (uv_fc_b & (((uv_rx_b < 5.0) & (uv_ry_b >= 159.2) & (uv_ry_b <= 163.5)) |
                             ((uv_rx_b < 3.9) & (uv_ry_b >= 150.5) & (uv_ry_b <= 156.8)))) | exposed_left_brow
    is_bang_shadow_zone = (uv_ry_b > 165.1) | ((uv_sx_b < 0.6) & (uv_ry_b > 163.2))
    min_fc_lum = np.where((uv_ry_b < 150.2) | is_bang_shadow_zone, 132.0, 84.0)
    min_fc_rb = np.where(is_bang_shadow_zone, 22.0, 14.0)
    cheek_skin = uv_fc_b & (~inner_feat) & ((rr - bb) > min_fc_rb) & (lum_b > min_fc_lum)
    seed_face = valid_b & ((inner_feat & (lum_b > 18.0)) | cheek_skin)
    seed_skin = seed_body | seed_face

    med_skin = np.median(warp_b[seed_body].reshape(-1, 3), axis=0).astype(np.float32)
    med_lum = max(float(0.114 * med_skin[0] + 0.587 * med_skin[1] + 0.299 * med_skin[2]), 1.0)

    obs_sk = warp_b.astype(np.float32).copy()
    y_obs = 0.114 * obs_sk[..., 0] + 0.587 * obs_sk[..., 1] + 0.299 * obs_sk[..., 2]
    sf_b = seed_body.astype(np.float32)
    y_mean = np.where(fast_gaussian_blur(sf_b, S / 38.0) > 0.05,
                      fast_gaussian_blur(y_obs * sf_b, S / 38.0) / np.maximum(fast_gaussian_blur(sf_b, S / 38.0), 1e-4),
                      med_lum)
    dl_ratio = np.clip(med_lum / np.maximum(y_mean, 40.0), 0.72, 1.32)
    face_w = fast_gaussian_blur(uv_fc_b.astype(np.float32), S / 170.0)
    dl_str = 0.68 * (1.0 - face_w) + 0.12 * face_w
    obs_delit = np.clip(obs_sk * (1.0 - dl_str[..., None] + dl_str[..., None] * dl_ratio[..., None]), 0, 255)

    field_src = obs_delit.copy()
    field_src[seed_face & (lum_b < 88.0)] = med_skin
    sf_all = seed_skin.astype(np.float32)
    w1 = fast_gaussian_blur(sf_all, S / 34.0); w2 = fast_gaussian_blur(sf_all, S / 10.0)
    c1 = np.stack([fast_gaussian_blur(field_src[..., i] * sf_all, S / 34.0) for i in range(3)], axis=-1)
    c2 = np.stack([fast_gaussian_blur(field_src[..., i] * sf_all, S / 10.0) for i in range(3)], axis=-1)
    multi = (c1 + 0.40 * c2 + med_skin * 0.025) / (w1[..., None] + 0.40 * w2[..., None] + 0.025)
    y_m = 0.114 * multi[..., 0] + 0.587 * multi[..., 1] + 0.299 * multi[..., 2]
    lo_p, hi_p = np.percentile(y_m[seed_body], [15, 90])
    skin_bg = (med_skin / med_lum)[None, None, :] * np.clip(y_m, lo_p, hi_p)[..., None]

    dist_in = ndimage.distance_transform_edt(seed_skin).astype(np.float32)
    w_obs_sk = np.where(inner_feat & seed_face, 1.0,
                        np.clip(dist_in / (S / 48.0), 0.0, 1.0) * np.clip((uv_cos_b - 0.26) / 0.28, 0.0, 1.0))
    w_obs_sk = cv2.GaussianBlur(w_obs_sk.astype(np.float32), (0, 0), 14.0)[..., None]
    skin_out = obs_delit * w_obs_sk + skin_bg * (1.0 - w_obs_sk)

    blur_f = cv2.GaussianBlur(skin_out, (0, 0), 1.8)
    skin_out = np.clip(skin_out * (1.0 - face_w[..., None]) + np.clip(skin_out + 0.48 * (skin_out - blur_f), 0, 255) * face_w[..., None], 0, 255)
    # 脚踝以下袜底区涂成干净浅米白以免鞋口内窥见深色
    sock_w = cv2.GaussianBlur((cov_b & under_sock).astype(np.float32), (0, 0), 3.0)[..., None]
    skin_out = skin_out * (1.0 - sock_w) + np.full_like(skin_out, [212.0, 215.0, 220.0]) * sock_w

    _, near_b = ndimage.distance_transform_edt(~cov_b, return_distances=True, return_indices=True)
    skin_u8 = np.clip(skin_out, 0, 255).astype(np.uint8)[tuple(near_b)]
    cv2.imwrite(str(out_dir / "skin_body_v3.png"), skin_u8)
    print(f"[skin_v3] 纯皮肤身体图集已输出 -> {out_dir / 'skin_body_v3.png'}")

    # ==================================================================
    # B. 合成物理衬衫壳体专属 4K 印花图集 shirt_atlas_v3.png
    # ==================================================================
    ys_s, xs_s, _, p3d_s, n3d_s, r3d_s, cov_s = rasterize_uv_barycentric(
        v_sh_cam, ds["shirt_rest"].astype(np.float32), ds["shirt_uv"].astype(np.float32),
        f_sh, ds["shirt_tf"].astype(np.int32), S
    )
    view_s = -p3d_s / (np.linalg.norm(p3d_s, axis=1, keepdims=True) + 1e-9)
    cos_s = (n3d_s * view_s).sum(axis=1)
    pu_s = f * p3d_s[:, 0] / np.maximum(p3d_s[:, 2], 1e-3) + cx
    pv_s = f * p3d_s[:, 1] / np.maximum(p3d_s[:, 2], 1e-3) + cy
    ui_s = np.clip(np.round(pu_s).astype(np.int32), 0, w - 1)
    vi_s = np.clip(np.round(pv_s).astype(np.int32), 0, h - 1)

    unocc_s = ((p3d_s[:, 2] - z_all[vi_s, ui_s]) < 0.018) & (
        (~jump_all[vi_s, ui_s]) | (np.abs(p3d_s[:, 2] - z_all[vi_s, ui_s]) < 0.005)
    )
    in_img_s = (pu_s >= 2) & (pu_s < w - 3) & (pv_s >= 2) & (pv_s < h - 3)
    valid_s_1d = in_img_s & (person_cloth[vi_s, ui_s] > 0) & (cos_s > 0.20) & unocc_s

    mx_s = np.zeros((S, S), dtype=np.float32); my_s = np.zeros((S, S), dtype=np.float32)
    mx_s[ys_s, xs_s] = pu_s; my_s[ys_s, xs_s] = pv_s
    warp_s = cv2.remap(img, mx_s, my_s, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    valid_s = np.zeros((S, S), dtype=bool); valid_s[ys_s, xs_s] = valid_s_1d
    bs, gs, rs = warp_s[..., 0].astype(np.float32), warp_s[..., 1].astype(np.float32), warp_s[..., 2].astype(np.float32)
    lum_s = (bs + gs + rs) / 3.0

    # 滤除宽松衣摆悬空边缘投向裸露腰腹皮肤或黑发的像素，确保衬衫壳体 100% 只有印花布料！
    not_skin_or_hair = (~(((rs - bs) > 35.0) & ((gs - bs) > 14.0) & (lum_s < 180.0))) & (lum_s > 58.0)
    obs_shirt = valid_s & cov_s & not_skin_or_hair
    obs_shirt = cv2.morphologyEx(obs_shirt.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0

    img_b, img_g, img_r = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    img_lum = (img_b + img_g + img_r) / 3.0
    sh_2d = np.zeros((h, w), dtype=np.uint8)
    sh_2d[vi_s[valid_s_1d], ui_s[valid_s_1d]] = 1
    sh_2d = cv2.morphologyEx(sh_2d, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    pure_sh_2d = (sh_2d > 0) & ((img_r - img_b) < 32.0) & (img_lum > 75.0) & (img_lum < 218.0)
    pure_sh_2d = cv2.erode(pure_sh_2d.astype(np.uint8), np.ones((3, 3), np.uint8))

    shirt_tile = make_seamless_fabric(img, pure_sh_2d, tile_px=640, fallback_bgr=(156, 158, 175))
    # 使用 3D 躯干环向柱面坐标采样周期碎花环面图（分支割线置于前胸正中，后背脊柱跨 UV 接缝 100.0% 连续无拼缝！）
    shirt_bg = tile_to_canvas(shirt_tile, S).astype(np.float32)
    ts_h, ts_w = shirt_tile.shape[:2]
    theta_s = np.arctan2(r3d_s[:, 0], -r3d_s[:, 2])  # 后背正中 theta=0 完全连续，割线位于前胸正中
    u_cyl = np.mod((theta_s / (2.0 * np.pi) + 0.5) * 2.2 * ts_w, float(ts_w)).astype(np.float32)
    v_cyl = np.mod((152.0 - r3d_s[:, 1]) * (ts_h / 32.0), float(ts_h)).astype(np.float32)
    cyl_map_x = np.zeros((S, S), dtype=np.float32); cyl_map_x[ys_s, xs_s] = u_cyl
    cyl_map_y = np.zeros((S, S), dtype=np.float32); cyl_map_y[ys_s, xs_s] = v_cyl
    shirt_bg_cyl = cv2.remap(shirt_tile, cyl_map_x, cyl_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32)
    shirt_bg[cov_s] = shirt_bg_cyl[cov_s]

    sh_photo = warp_s.astype(np.float32)
    if obs_shirt.any():
        med_sh = np.median(sh_photo[obs_shirt].reshape(-1, 3), axis=0)
        bg_m = np.maximum(np.median(shirt_bg.reshape(-1, 3), axis=0), 1.0)
        shirt_bg = np.clip(shirt_bg * np.clip(med_sh / bg_m, 0.82, 1.22)[None, None, :], 0, 255)

        yc = 0.114 * sh_photo[..., 0] + 0.587 * sh_photo[..., 1] + 0.299 * sh_photo[..., 2]
        yc_low = np.maximum(fast_gaussian_blur(yc, S / 20.0), 1.0)
        sh_flat = np.clip(sh_photo * np.clip(float(np.median(yc[obs_shirt])) / yc_low, 0.80, 1.22)[..., None], 0, 255)
        sh_obs_col = 0.82 * sh_photo + 0.18 * sh_flat
    else:
        sh_obs_col = shirt_bg

    dist_sh = ndimage.distance_transform_edt(obs_shirt).astype(np.float32)
    w_sh = cv2.GaussianBlur(np.clip(dist_sh / 28.0, 0.0, 1.0), (0, 0), 6.0)[..., None]
    shirt_comp = np.clip(sh_obs_col * w_sh + shirt_bg * (1.0 - w_sh), 0, 255).astype(np.uint8)
    # 全画布 Gutter 填充保证 SOLIDIFY 边缘厚度面同样拥有印花布色
    _, near_s = ndimage.distance_transform_edt(~cov_s, return_distances=True, return_indices=True)
    shirt_final = np.where(cov_s[..., None], shirt_comp, shirt_comp[tuple(near_s)])
    cv2.imwrite(str(out_dir / "shirt_atlas_v3.png"), shirt_final)
    print(f"[shirt_v3] 衬衫壳体图集已输出 -> {out_dir / 'shirt_atlas_v3.png'} (有效正面采样={obs_shirt[cov_s].mean():.1%})")

    # ==================================================================
    # C. 合成重工水洗浅蓝牛仔短裤壳体专属 4K 图集 shorts_atlas_v3.png
    # ==================================================================
    ys_d, xs_d, _, p3d_d, n3d_d, r3d_d, cov_d = rasterize_uv_barycentric(
        v_dn_cam, ds["shorts_rest"].astype(np.float32), ds["shorts_uv"].astype(np.float32),
        f_dn, ds["shorts_tf"].astype(np.int32), S
    )
    view_d = -p3d_d / (np.linalg.norm(p3d_d, axis=1, keepdims=True) + 1e-9)
    cos_d = (n3d_d * view_d).sum(axis=1)
    pu_d = f * p3d_d[:, 0] / np.maximum(p3d_d[:, 2], 1e-3) + cx
    pv_d = f * p3d_d[:, 1] / np.maximum(p3d_d[:, 2], 1e-3) + cy
    ui_d = np.clip(np.round(pu_d).astype(np.int32), 0, w - 1)
    vi_d = np.clip(np.round(pv_d).astype(np.int32), 0, h - 1)

    unocc_d = ((p3d_d[:, 2] - z_all[vi_d, ui_d]) < 0.018) & (
        (~jump_all[vi_d, ui_d]) | (np.abs(p3d_d[:, 2] - z_all[vi_d, ui_d]) < 0.006)
    )
    in_img_d = (pu_d >= 2) & (pu_d < w - 3) & (pv_d >= 2) & (pv_d < h - 3)
    valid_d_1d = in_img_d & (person_cloth[vi_d, ui_d] > 0) & (cos_d > 0.22) & unocc_d

    mx_d = np.zeros((S, S), dtype=np.float32); my_d = np.zeros((S, S), dtype=np.float32)
    mx_d[ys_d, xs_d] = pu_d; my_d[ys_d, xs_d] = pv_d
    warp_d = cv2.remap(img, mx_d, my_d, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    valid_d = np.zeros((S, S), dtype=bool); valid_d[ys_d, xs_d] = valid_d_1d
    uv_rx_d = np.zeros((S, S), dtype=np.float32); uv_rx_d[ys_d, xs_d] = r3d_d[:, 0]
    uv_ry_d = np.zeros((S, S), dtype=np.float32); uv_ry_d[ys_d, xs_d] = r3d_d[:, 1]
    uv_rz_d = np.zeros((S, S), dtype=np.float32); uv_rz_d[ys_d, xs_d] = r3d_d[:, 2]
    abs_rx_d = np.abs(uv_rx_d)

    bd, gd, rd = warp_d[..., 0].astype(np.float32), warp_d[..., 1].astype(np.float32), warp_d[..., 2].astype(np.float32)
    lum_d = (bd + gd + rd) / 3.0
    is_denim = ((rd - bd) < 34.0) & (bd > 90.0) & (lum_d > 74.0) & (lum_d < 214.0)
    obs_shorts = valid_d & cov_d & is_denim
    obs_shorts = cv2.morphologyEx(obs_shorts.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0

    # 清透水洗浅蓝牛仔基准色 + 45° 斜纹竹节纱肌理
    if obs_shorts.any():
        raw_med = np.median(warp_d[obs_shorts].reshape(-1, 3), axis=0).astype(np.float32)
        # 校正拍立得暖黄偏色，还原原图清透水洗浅蓝牛仔 (B 略高于 R)
        denim_base_bgr = np.array([raw_med[0] * 1.12, raw_med[1] * 1.02, raw_med[2] * 0.87], dtype=np.float32)
    else:
        denim_base_bgr = np.array([164.0, 148.0, 136.0], dtype=np.float32)

    denim_bg = synthesize_denim_twill_canvas(S, denim_base_bgr)

    # 在背面/侧面合成真实牛仔裤结构工艺细节（自适应前低后高人体工学高腰头、后育克线、后贴袋、毛边白线须）
    stitch_gold = np.array([108.0, 132.0, 158.0], dtype=np.float32)
    seam_shade = 0.84

    # 1) 高腰头下沿压线与金棕车线（同步匹配前低后高+胯侧提拉立体高腰曲线）
    hip_lift_uv = 2.1 * np.exp(-((abs_rx_d - 11.5) / 4.8) ** 2) * (0.5 + 0.5 * np.tanh(uv_rz_d / 4.0))
    wb_top_uv = 99.2 - 2.2 * np.tanh(uv_rz_d / 4.5) + hip_lift_uv
    wb_seam_y = wb_top_uv - 3.0
    d_wb = np.abs(uv_ry_d - wb_seam_y)
    wb_seam = cov_d & (d_wb < 0.18)
    wb_stitch = cov_d & ((np.abs(d_wb - 0.35) < 0.09) | (np.abs(uv_ry_d - (wb_top_uv - 0.45)) < 0.10))

    # 2) 后臀 V 字育克缝线 (Back Yoke) 与左右对称五角后贴袋 (rz < -2.0cm)
    yoke_y = 93.2 - 0.20 * abs_rx_d
    back_yoke = cov_d & (uv_rz_d < -2.0) & (np.abs(uv_ry_d - yoke_y) < 0.18)
    in_pkt_x = (abs_rx_d >= 3.6) & (abs_rx_d <= 10.6)
    pkt_bot = 82.0 + 0.32 * np.abs(abs_rx_d - 7.1)
    pkt_edge = cov_d & (uv_rz_d < -3.5) & (
        (in_pkt_x & (np.abs(uv_ry_d - 90.2) < 0.16)) |
        (in_pkt_x & (np.abs(uv_ry_d - pkt_bot) < 0.18)) |
        ((uv_ry_d >= pkt_bot) & (uv_ry_d <= 90.2) & ((np.abs(abs_rx_d - 3.6) < 0.16) | (np.abs(abs_rx_d - 10.6) < 0.16)))
    )
    in_pkt_fill = cov_d & (uv_rz_d < -3.5) & in_pkt_x & (uv_ry_d >= pkt_bot) & (uv_ry_d <= 90.2)
    denim_bg[in_pkt_fill] *= 0.95

    # 3) 中缝门襟 J 形车线与臀中缝
    center_seam = cov_d & (abs_rx_d < 0.18) & (uv_ry_d < wb_seam_y)
    j_fly = cov_d & (uv_rz_d > 2.0) & (uv_rx_d > 0.0) & (uv_rx_d < 2.3) & (uv_ry_d > 83.5) & (uv_ry_d < wb_seam_y - 0.2) & (
        (np.abs(uv_rx_d - 2.15) < 0.14) | ((uv_ry_d < 85.2) & (np.abs((uv_ry_d - 83.5) / 1.7 - (2.15 - uv_rx_d) / 2.15) < 0.15))
    )

    # 4) 正面弧形插袋口 (Front Scoop Pockets)
    scoop_r = np.sqrt(((abs_rx_d - 14.5) / 6.2) ** 2 + ((uv_ry_d - wb_seam_y) / 4.6) ** 2)
    front_scoop = cov_d & (uv_rz_d > 2.0) & (abs_rx_d > 7.5) & (uv_ry_d < wb_seam_y) & (np.abs(scoop_r - 1.0) < 0.045)

    for seam_m in (wb_seam, back_yoke, pkt_edge, center_seam, front_scoop):
        denim_bg[seam_m] *= seam_shade
    for st_m in (wb_stitch, j_fly):
        denim_bg[st_m] = 0.55 * denim_bg[st_m] + 0.45 * stitch_gold

    # 正面真实观测与工艺底图融合
    dn_photo = np.clip(warp_d.astype(np.float32) * np.array([1.10, 1.01, 0.89], dtype=np.float32)[None, None, :], 0, 255)
    dist_dn = ndimage.distance_transform_edt(obs_shorts).astype(np.float32)
    w_dn = cv2.GaussianBlur(np.clip(dist_dn / 16.0, 0.0, 1.0), (0, 0), 4.0)[..., None]
    shorts_comp = dn_photo * w_dn + denim_bg * (1.0 - w_dn)

    # 5) 裤管下沿 0.95cm 经典水洗毛边白棉线须 (Raw Frayed Denim Fringe)
    leg_hem_y = 73.8 + 0.38 * np.clip(abs_rx_d - 5.0, 0.0, 14.5)
    d_cuff = uv_ry_d - leg_hem_y
    in_fringe = cov_d & (d_cuff >= 0.0) & (d_cuff < 0.95)
    # 沿环向角度生成密集竖向白线须纹理
    theta_leg = np.arctan2(uv_rz_d, np.abs(uv_rx_d) - 9.0)
    fray_stripes = 0.5 + 0.5 * np.sin(theta_leg * 190.0 + np.sin(theta_leg * 37.0) * 2.5)
    fray_w = np.clip((0.95 - d_cuff) / 0.95, 0.0, 1.0) * (0.45 + 0.55 * fray_stripes)
    fray_col = np.array([218.0, 219.0, 222.0], dtype=np.float32)
    shorts_comp = np.where(
        in_fringe[..., None],
        shorts_comp * (1.0 - 0.78 * fray_w[..., None]) + fray_col * (0.78 * fray_w[..., None]),
        shorts_comp,
    )

    shorts_u8 = np.clip(shorts_comp, 0, 255).astype(np.uint8)
    _, near_d = ndimage.distance_transform_edt(~cov_d, return_distances=True, return_indices=True)
    shorts_final = np.where(cov_d[..., None], shorts_u8, shorts_u8[tuple(near_d)])
    cv2.imwrite(str(out_dir / "shorts_atlas_v3.png"), shorts_final)
    print(f"[shorts_v3] 牛仔短裤壳体图集已输出 -> {out_dir / 'shorts_atlas_v3.png'} (有效正面采样={obs_shorts[cov_d].mean():.1%})")

    stats = {
        "atlas_size": S,
        "shirt_front_obs_ratio": round(float(obs_shirt[cov_s].mean()), 4),
        "shorts_front_obs_ratio": round(float(obs_shorts[cov_d].mean()), 4),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_dir / "tex_stats_v3.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[done] v3 三通道解耦图集全部构建完毕 ({stats['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
