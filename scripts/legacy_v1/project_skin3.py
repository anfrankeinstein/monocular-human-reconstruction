"""S6 皮肤投影 v3：全表面覆盖。

与 v2 的差别：
- 所有朝向面都写进图集（不再只写皮肤面），因此 UV 岛 100% 有值
- 每 texel 记录「是否属于裸露皮肤」（warp 皮肤 mask，不是判面中心点）
- 非皮肤 texel（被衣服遮挡的区域）用「最近皮肤 texel」填充 → 全身连续肤色
- 去光照（Retinex 型）后把整体均值对齐到照片皮肤均值
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

MHR_PT = r"Q:\stb-assets\models\sam-3d-body-dinov3\assets\mhr_model.pt"


def nearest_fill(img, known):
    """用最近的已知像素填满 unknown。"""
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return img[tuple(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--focal", type=float, default=1795.69)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--delight", type=float, default=0.6, help="0=保留原光照 1=完全压平")
    ap.add_argument("--facing", type=float, default=-0.02)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    m = torch.jit.load(MHR_PT, map_location="cpu")
    me = m.character_torch.mesh
    tex = me.texcoords.numpy().astype(np.float64)
    tfaces = me.texcoord_faces.numpy()
    faces = me.faces.numpy()

    d = np.load(args.params, allow_pickle=True)
    v = np.asarray(d["pred_vertices"]).reshape(-1, 3).astype(np.float64)
    t = np.asarray(d["pred_cam_t"]).reshape(-1)[:3].astype(np.float64)
    f = float(np.asarray(d["focal_length"]).reshape(-1)[0]) if "focal_length" in d.files else args.focal

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    skin = cv2.imread(args.skin, cv2.IMREAD_GRAYSCALE)
    cx, cy = w / 2.0, h / 2.0

    p = v + t
    z = np.clip(p[:, 2], 1e-3, None)
    uv2d = np.stack([f * p[:, 0] / z + cx, f * p[:, 1] / z + cy], axis=1)

    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    cen = v[faces].mean(axis=1) + t
    view = -cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    facing = (fn * view).sum(axis=1) > args.facing
    order = np.argsort(-cen[:, 2])

    S = args.size
    atlas = np.zeros((S, S, 3), np.uint8)
    painted = np.zeros((S, S), np.uint8)
    sk = np.zeros((S, S), np.uint8)
    n_face = 0

    for fi in order:
        if not facing[fi]:
            continue
        tri_v = faces[fi]
        tri_t = tfaces[fi]
        src = uv2d[tri_v].astype(np.float32)
        if not np.isfinite(src).all():
            continue
        if src[:, 0].max() < -w or src[:, 0].min() > 2 * w or \
           src[:, 1].max() < -h or src[:, 1].min() > 2 * h:
            continue
        dst = np.stack([tex[tri_t, 0] * (S - 1), (1.0 - tex[tri_t, 1]) * (S - 1)], axis=1).astype(np.float32)

        M = cv2.getAffineTransform(src, dst)
        x, y, bw, bh = cv2.boundingRect(dst.astype(np.int32))
        x0, y0 = max(x - args.pad, 0), max(y - args.pad, 0)
        x1, y1 = min(x + bw + args.pad, S), min(y + bh + args.pad, S)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        M2 = M.copy()
        M2[0, 2] -= x0
        M2[1, 2] -= y0
        warped = cv2.warpAffine(img, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        wskin = cv2.warpAffine(skin, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        local = dst - np.array([x0, y0], dtype=np.float32)
        cv2.fillConvexPoly(tri_mask, np.round(local).astype(np.int32), 255)

        sel = tri_mask > 0
        roi = atlas[y0:y1, x0:x1]
        roi[sel] = warped[sel]
        atlas[y0:y1, x0:x1] = roi
        rm = painted[y0:y1, x0:x1]
        rm[sel] = 1
        painted[y0:y1, x0:x1] = rm
        rs = sk[y0:y1, x0:x1]
        rs[sel] = (wskin[sel] > 127).astype(np.uint8)
        sk[y0:y1, x0:x1] = rs
        n_face += 1

    cov = painted.mean()
    # 注意：numpy 2.x 里 bool & uint8 会升成 uint8，必须用 logical_and 才是布尔掩码
    skin_tex = sk > 0
    print(f"[warp] 朝向面 {n_face}/{len(faces)}  图集覆盖 {cov:.2%}  其中皮肤 texel {skin_tex.mean():.2%}")

    # ---- 全表面填充：以皮肤 texel 为种子，最近邻 + 羽化 ----
    seed = np.logical_and(skin_tex, painted > 0)
    if seed.sum() < 100:
        raise SystemExit("皮肤种子 texel 太少，检查 mask / 投影")
    base = nearest_fill(atlas, seed).astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), S / 90.0)
    dist = ndimage.distance_transform_edt(~seed).astype(np.float32)
    wgt = np.clip(dist / (S / 64.0), 0, 1)[..., None]
    filled = base * (1 - wgt) + blur * wgt

    # ---- 去光照 ----
    low = cv2.GaussianBlur(filled, (0, 0), S / 24.0)
    low = np.maximum(low, 1.0)
    flat = filled * (low.mean() / low)
    out = filled * (1 - args.delight) + flat * args.delight
    out = np.clip(out, 0, 255).astype(np.uint8)

    # ---- 均值对齐到照片皮肤 ----
    tgt = img[skin > 0].reshape(-1, 3).mean(axis=0)
    cur = out[seed].reshape(-1, 3).mean(axis=0)
    gain = np.clip(tgt / np.maximum(cur, 1.0), 0.75, 1.35)
    out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    print(f"[tone] 目标 BGR {tgt.round(1)}  当前 {cur.round(1)}  增益 {gain.round(3)}")

    cv2.imwrite(str(outdir / "skin_atlas_v3.png"), out)
    cv2.imwrite(str(outdir / "skin_atlas_v3_prefelt.png"), filled.astype(np.uint8))
    cv2.imwrite(str(outdir / "skin_uv_painted.png"), sk * 255)
    cv2.imwrite(str(outdir / "uv_covered.png"), painted * 255)

    stats = {"faces": int(len(faces)), "facing": int(n_face), "uv_coverage": float(cov),
             "skin_texels": float(skin_tex.mean()), "atlas": S,
             "tone_target": [float(x) for x in tgt], "gain": [float(x) for x in gain]}
    (outdir / "skin_stats_v3.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("[out]", json.dumps(stats))


if __name__ == "__main__":
    main()
