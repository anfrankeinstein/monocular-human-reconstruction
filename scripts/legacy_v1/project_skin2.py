"""S6 皮肤投影 v2：逐三角形仿射 warp + 深度排序，生成连续 UV 贴图。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

MHR_PT = r"Q:\stb-assets\models\sam-3d-body-dinov3\assets\mhr_model.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--focal", type=float, default=1795.69)
    ap.add_argument("--pad", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    m = torch.jit.load(MHR_PT, map_location="cpu")
    me = m.character_torch.mesh
    tex = me.texcoords.numpy().astype(np.float64)
    tfaces = me.texcoord_faces.numpy()
    faces = me.faces.numpy()
    print(f"[uv] texcoords={tex.shape} faces={faces.shape}")

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
    facing = (fn * view).sum(axis=1) > 0.05

    fz = cen[:, 2]
    order = np.argsort(-fz)  # 远到近，近的覆盖远的

    S = args.size
    atlas = np.full((S, S, 3), (196, 205, 214), dtype=np.uint8)
    painted = np.zeros((S, S), np.uint8)
    n_face, n_skinface = 0, 0

    for fi in order:
        if not facing[fi]:
            continue
        n_face += 1
        tri_v = faces[fi]
        tri_t = tfaces[fi]
        src = uv2d[tri_v].astype(np.float32)
        dst = np.stack([tex[tri_t, 0] * (S - 1), (1.0 - tex[tri_t, 1]) * (S - 1)], axis=1).astype(np.float32)

        cxs = int(np.mean(np.clip(src[:, 0], 0, w - 1)))
        cys = int(np.mean(np.clip(src[:, 1], 0, h - 1)))
        if skin[cys, cxs] == 0:
            continue
        n_skinface += 1

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

        tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        local = dst - np.array([x0, y0], dtype=np.float32)
        cv2.fillConvexPoly(tri_mask, np.round(local).astype(np.int32), 255)

        roi = atlas[y0:y1, x0:x1]
        sel = tri_mask > 0
        roi[sel] = warped[sel]
        atlas[y0:y1, x0:x1] = roi
        painted[y0:y1, x0:x1][sel] = 1

    print(f"[warp] 朝向面 {n_face}，皮肤面 {n_skinface}，UV 覆盖 {painted.mean():.2%}")

    holes = (painted == 0).astype(np.uint8)
    near = cv2.dilate(painted, np.ones((31, 31), np.uint8))
    fill = holes & near
    atlas = cv2.inpaint(atlas, fill, 7, cv2.INPAINT_TELEA)

    f32 = atlas.astype(np.float32) + 1.0
    low = cv2.GaussianBlur(f32, (0, 0), S / 26)
    flat = np.clip(f32 / np.maximum(low / low.mean(), 1e-3), 0, 255)
    blend = cv2.addWeighted(atlas, 0.3, flat.astype(np.uint8), 0.7, 0)

    cv2.imwrite(str(outdir / "skin_atlas_v2_raw.png"), atlas)
    cv2.imwrite(str(outdir / "skin_atlas_v2_flat.png"), blend)
    cv2.imwrite(str(outdir / "skin_uv_painted.png"), painted * 255)

    stats = {"faces_total": int(len(faces)), "facing": n_face, "skin_faces": n_skinface,
             "uv_coverage": float(painted.mean()), "atlas_size": S}
    (outdir / "skin_stats_v2.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[out] {stats}")


if __name__ == "__main__":
    main()
