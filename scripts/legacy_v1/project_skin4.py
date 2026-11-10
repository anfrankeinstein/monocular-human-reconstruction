"""S6 皮肤投影 v4：在 v3 全表面覆盖的基础上，修掉 Voronoi 色块放大问题。

v3 的缺陷：非皮肤区域用「最近皮肤 texel」填充，等于给每个种子分配一个 Voronoi 单元。
边界上少量被背景/衣物污染的种子会被放大成大色块（实测手臂上出现蓝灰斑）。

v4 做法：
1. 种子在 4096 图集空间做颜色闸门（冷色剔除）+ 形态学开运算，去掉边界碎屑
2. 非种子区域用「种子加权高斯场」填充 —— 每个 texel 取邻近上千个种子的加权均值，
   个别坏种子被稀释，不会再形成色块
3. 种子区内保留照片原始细节（面部五官、手部）
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

MHR_PT = r"Q:\stb-assets\models\sam-3d-body-dinov3\assets\mhr_model.pt"


def build_atlas(params, image, skin, S, pad=2, facing_thr=-0.02, scale=1.0, face=None, mesh=None):
    """mesh 参数给定时从 npz 读网格（v/faces/tex/tfaces），否则退回 TorchScript 的 LOD1。"""
    d = np.load(params, allow_pickle=True)
    if mesh:
        dm = np.load(mesh)
        tex = np.asarray(dm["tex"]).astype(np.float64)
        tfaces = np.asarray(dm["tfaces"])
        faces = np.asarray(dm["faces"])
        v = np.asarray(dm["v"]).reshape(-1, 3).astype(np.float64)
    else:
        m = torch.jit.load(MHR_PT, map_location="cpu")
        me = m.character_torch.mesh
        tex = me.texcoords.numpy().astype(np.float64)
        tfaces = me.texcoord_faces.numpy()
        faces = me.faces.numpy()
        v = np.asarray(d["pred_vertices"]).reshape(-1, 3).astype(np.float64)
    t = np.asarray(d["pred_cam_t"]).reshape(-1)[:3].astype(np.float64)
    f = float(np.asarray(d["focal_length"]).reshape(-1)[0]) if "focal_length" in d.files else 1795.69
    f *= scale

    img = cv2.imread(image)
    h, w = img.shape[:2]
    sk = cv2.imread(skin, cv2.IMREAD_GRAYSCALE)
    fm = cv2.imread(face, cv2.IMREAD_GRAYSCALE) if face else None
    cx, cy = w / 2.0, h / 2.0

    p = v + t
    z = np.clip(p[:, 2], 1e-3, None)
    uv2d = np.stack([f * p[:, 0] / z + cx, f * p[:, 1] / z + cy], axis=1)

    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    cen = v[faces].mean(axis=1) + t
    view = -cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    facing = (fn * view).sum(axis=1) > facing_thr
    order = np.argsort(-cen[:, 2])

    atlas = np.zeros((S, S, 3), np.uint8)
    painted = np.zeros((S, S), np.uint8)
    flag = np.zeros((S, S), np.uint8)
    fflag = np.zeros((S, S), np.uint8)
    n = 0
    for fi in order:
        if not facing[fi]:
            continue
        src = uv2d[faces[fi]].astype(np.float32)
        if not np.isfinite(src).all():
            continue
        if src[:, 0].max() < -w or src[:, 0].min() > 2 * w or \
           src[:, 1].max() < -h or src[:, 1].min() > 2 * h:
            continue
        tri_t = tfaces[fi]
        dst = np.stack([tex[tri_t, 0] * (S - 1), (1.0 - tex[tri_t, 1]) * (S - 1)], axis=1).astype(np.float32)

        M = cv2.getAffineTransform(src, dst)
        x, y, bw, bh = cv2.boundingRect(dst.astype(np.int32))
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, S), min(y + bh + pad, S)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        M2 = M.copy()
        M2[0, 2] -= x0
        M2[1, 2] -= y0
        warped = cv2.warpAffine(img, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        wskin = cv2.warpAffine(sk, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        local = dst - np.array([x0, y0], dtype=np.float32)
        cv2.fillConvexPoly(tri_mask, np.round(local).astype(np.int32), 255)
        sel = tri_mask > 0

        roi = atlas[y0:y1, x0:x1]
        roi[sel] = warped[sel]
        atlas[y0:y1, x0:x1] = roi
        rp = painted[y0:y1, x0:x1]
        rp[sel] = 1
        painted[y0:y1, x0:x1] = rp
        rf = flag[y0:y1, x0:x1]
        rf[sel] = (wskin[sel] > 127).astype(np.uint8)
        flag[y0:y1, x0:x1] = rf
        if fm is not None:
            wface = cv2.warpAffine(fm, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rr = fflag[y0:y1, x0:x1]
            rr[sel] = (wface[sel] > 127).astype(np.uint8)
            fflag[y0:y1, x0:x1] = rr
        n += 1
    return atlas, painted, flag, n, len(faces), fflag


def skin_field(col, seed, S, sig_frac=18.0, gate=10.0, min_island=0.0004, mode="med"):
    """种子加权高斯场：稳健的非种子区填充（不会放大单个坏种子）。"""
    b, g, r = col[..., 0].astype(np.float32), col[..., 1].astype(np.float32), col[..., 2].astype(np.float32)
    warm = (r - b) > gate
    # 亮度闸门：照片里的白袜、鞋帮、浅色手链会被几何 mask 误当皮肤
    v = (b + g + r) / 3.0
    s = np.logical_and(np.logical_and(seed, warm), v < 200.0)
    s = cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    # 去掉孤立小岛（指缝间背景、袖口碎屑）
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(s.astype(np.uint8), connectivity=8)
    keep = np.zeros(nlab, bool)
    thr = S * S * min_island
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            keep[i] = True
    s = keep[lab] if nlab > 1 else s
    print(f"[gate] 冷色剔除后种子 {s.mean():.2%}，保留岛屿 {int(keep.sum())}/{max(nlab-1,0)}")
    if s.sum() < 1000:
        raise SystemExit("有效种子不足")

    sig = S / sig_frac
    sf = s.astype(np.float32)
    wsum = cv2.GaussianBlur(sf, (0, 0), sig)
    cs = np.stack([cv2.GaussianBlur(col[..., i].astype(np.float32) * sf, (0, 0), sig) for i in range(3)], -1)
    med = np.median(col[s].reshape(-1, 3), axis=0)
    # 关键：远离种子时 cs 与 wsum 同时趋零，直接相除会数值爆炸（→ clip 成纯白）。
    # 用 eps 把无支撑区域平滑回退到全局中位色。
    eps = 0.01
    field = (cs + med * eps) / (wsum[..., None] + eps)
    if mode == "med":
        field = np.broadcast_to(med, col.shape).astype(np.float32).copy()
    elif mode == "local":
        # 亮度取「邻域种子加权均值」、色度取皮肤中位色：
        # 既跟着邻近皮肤的明暗走（不会有浅色补丁），又不会被别的 UV 岛的色偏污染。
        c_l = (cs + med * eps) / (wsum[..., None] + eps)
        y_l = 0.114 * c_l[..., 0] + 0.587 * c_l[..., 1] + 0.299 * c_l[..., 2]
        lo_p, hi_p = np.percentile(y_l[s], [8, 88])
        med_lum = max(float(0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]), 1.0)
        field = (med / med_lum)[None, None, :] * np.clip(y_l, lo_p, hi_p)[..., None]
    elif mode == "luma":
        # 亮度取照片、色度取皮肤中位色：跨过衣物边界时明暗连续，不会出现浅色补丁。
        # 亮度先夹到皮肤种子自身的 [p3, p97] 区间，避免白袜/白袖口被当成高光带进来。
        y_ph = (0.114 * col[..., 0] + 0.587 * col[..., 1] + 0.299 * col[..., 2]).astype(np.float32)
        paint = (col.sum(axis=2) > 0).astype(np.float32)
        k = S / 60.0
        wp = cv2.GaussianBlur(paint, (0, 0), k)
        y_lo = cv2.GaussianBlur(y_ph * paint, (0, 0), k) / np.maximum(wp, 1e-3)
        lo_p, hi_p = np.percentile(y_lo[s], [10, 85])  # 收窄到皮肤中段，白袜/白袖口会被压回中灰
        med_lum = max(float(0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]), 1.0)
        y_lo = np.where(wp < 1e-3, med_lum, y_lo)
        y_cl = np.clip(y_lo, lo_p, hi_p)
        field = (med / med_lum)[None, None, :] * y_cl[..., None]

    filled = np.where(s[..., None], col.astype(np.float32), field)

    # 局部过亮抑制：照片里的白袜、浅色袖口、强反光会残留成白斑。
    # 以「局部低频亮度」为基准，明显过亮的种子 texel 直接压回填充色。
    lum = 0.114 * filled[..., 0] + 0.587 * filled[..., 1] + 0.299 * filled[..., 2]
    ref = np.maximum(cv2.GaussianBlur(lum, (0, 0), S / 120.0), 1.0)
    bad = np.logical_and(s, lum > ref * 1.12 + 4.0)
    if bad.any():
        filled[bad] = field[bad]
        print(f"[suppress] 压掉过亮 texel {bad.mean():.3%}")

    # 只在新填充区做 40px 羽化，种子区（照片细节）保持原样
    from scipy import ndimage
    soft = cv2.GaussianBlur(filled, (0, 0), 8.0)
    dd = ndimage.distance_transform_edt(~s).astype(np.float32)
    alpha = np.clip(dd / max(S / 100.0, 12.0), 0, 1)[..., None]
    return filled * (1 - alpha) + soft * alpha, s, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--delight", type=float, default=0.45)
    ap.add_argument("--sig-frac", type=float, default=34.0)
    ap.add_argument("--gate", type=float, default=10.0)
    ap.add_argument("--face-delight", type=float, default=None)
    ap.add_argument("--fill", default="local", choices=["local", "luma", "med", "field"])
    ap.add_argument("--scale", type=float, default=1.0, help="输入图相对原图的分辨率倍数（焦点同步放大）")
    ap.add_argument("--face-mask", default=None, help="2D 人脸区域 mask，用于单独降去光照强度")
    ap.add_argument("--mesh", default=None, help="mesh_uv npz（含 v/faces/tex/tfaces）；缺省用 TorchScript LOD1")
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--sharpen-radius", type=float, default=2.0)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    S = args.size

    atlas, painted, flag, n, nf, fflag = build_atlas(args.params, args.image, args.skin, S,
                                                     scale=args.scale, face=args.face_mask,
                                                     mesh=args.mesh)
    seed = np.logical_and(flag > 0, painted > 0)
    print(f"[warp] 朝向面 {n}/{nf}  覆盖 {painted.mean():.2%}  皮肤 texel {seed.mean():.2%}")

    filled, s, med = skin_field(atlas, seed, S, args.sig_frac, args.gate, mode=args.fill)
    print(f"[field] 采用种子 {s.mean():.2%}  种子中位色 BGR {med.round(1)}")

    # 去光照必须只按亮度做：逐通道归一化会把平滑区的颜色一起中性化成灰
    y = (0.114 * filled[..., 0] + 0.587 * filled[..., 1] + 0.299 * filled[..., 2])
    y_low = np.maximum(cv2.GaussianBlur(y, (0, 0), S / 24.0), 1.0)
    gain_map = float(y_low.mean()) / y_low
    flat = filled * gain_map[..., None]
    # 脸部单独降去光照强度：保住眼窝/鼻影/唇线，脸才不会糊成一片
    dl = np.full((S, S), args.delight, np.float32)
    if args.face_delight is not None and fflag is not None and fflag.any():
        fsoft = cv2.GaussianBlur((fflag > 0).astype(np.float32), (0, 0), S / 300.0)
        dl = dl * (1 - fsoft) + args.face_delight * fsoft
        print(f"[delight] 脸部权重 {float((fflag>0).mean()):.2%}，强度 {args.delight} → {args.face_delight}")
    out = filled * (1 - dl[..., None]) + flat * dl[..., None]
    out = np.clip(out, 0, 255)

    img = cv2.imread(args.image)
    sk = cv2.imread(args.skin, cv2.IMREAD_GRAYSCALE) > 0
    tgt = img[sk].reshape(-1, 3).mean(axis=0)
    cur = out[s].reshape(-1, 3).mean(axis=0)
    gain = np.clip(tgt / np.maximum(cur, 1.0), 0.75, 1.35)
    out = np.clip(out * gain, 0, 255).astype(np.uint8)
    print(f"[tone] 目标 {tgt.round(1)} 当前 {cur.round(1)} 增益 {gain.round(3)}")

    # 去光照压平了面部起伏，最后补一道 unsharp 把五官边缘找回来
    if args.sharpen > 0:
        f32 = out.astype(np.float32)
        blur = cv2.GaussianBlur(f32, (0, 0), args.sharpen_radius)
        out = np.clip(f32 + args.sharpen * (f32 - blur), 0, 255).astype(np.uint8)
        print(f"[sharpen] amount={args.sharpen} radius={args.sharpen_radius}")

    cv2.imwrite(str(outdir / "skin_atlas_v4.png"), out)
    cv2.imwrite(str(outdir / "seed_mask.png"), s.astype(np.uint8) * 255)
    cv2.imwrite(str(outdir / "uv_covered.png"), painted * 255)

    stats = {"atlas": S, "facing": n, "uv_coverage": float(painted.mean()),
             "seed_frac": float(s.mean()), "seed_median_bgr": [float(x) for x in med],
             "tone_target": [float(x) for x in tgt], "gain": [float(x) for x in gain],
             "delight": args.delight}
    (outdir / "skin_stats_v4.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("[out]", json.dumps(stats))


if __name__ == "__main__":
    main()
