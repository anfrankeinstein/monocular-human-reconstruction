"""S6 皮肤投影：按相机把照片的裸露皮肤投到 MHR 的 UV 上，并做去光照。"""
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
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    m = torch.jit.load(MHR_PT, map_location="cpu")
    me = m.character_torch.mesh
    tex = me.texcoords.numpy()
    tfaces = me.texcoord_faces.numpy()
    faces = me.faces.numpy()
    print(f"[uv] texcoords={tex.shape} faces={tfaces.shape}")

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
    u = f * p[:, 0] / z + cx
    ww = f * p[:, 1] / z + cy

    tc2v = {}
    for fi in range(len(tfaces)):
        for k in range(3):
            tc2v[int(tfaces[fi, k])] = int(faces[fi, k])
    print(f"[map] texcoord->vertex {len(tc2v)}")

    S = args.size
    atlas = np.full((S, S, 3), 205, dtype=np.uint8)
    filled = np.zeros((S, S), bool)

    # 面法线判断朝向（背面不采样）
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    center = v[faces].mean(axis=1) + t
    view = -center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-9)
    facing = (fn * view).sum(axis=1) > 0.05

    n_skin, n_total = 0, 0
    for tc, vi in tc2v.items():
        n_total += 1
        x, y = int(round(u[vi])), int(round(ww[vi]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        if skin[y, x] == 0:
            continue
        uu, vv = tex[tc]
        px, py = int(uu * (S - 1)), int((1.0 - vv) * (S - 1))
        if not (0 <= px < S and 0 <= py < S):
            continue
        b, g, r = img[y, x]
        atlas[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2] = (int(b), int(g), int(r))
        filled[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2] = True
        n_skin += 1

    print(f"[sample] 皮肤区采样 {n_skin}/{n_total} 个 texcoord，覆盖 {filled.mean():.2%} 的 UV 面积")

    # 空洞填充（只填已覆盖区域附近，避免污染未使用 UV）
    holes = (~filled).astype(np.uint8)
    near = cv2.dilate(filled.astype(np.uint8), np.ones((25, 25), np.uint8))
    fill_mask = (holes & near)
    atlas = cv2.inpaint(atlas, fill_mask, 5, cv2.INPAINT_TELEA)

    # 去光照：除以大尺度模糊，压掉闪光灯造成的明暗
    f32 = atlas.astype(np.float32) + 1.0
    low = cv2.GaussianBlur(f32, (0, 0), S / 24)
    flat = np.clip(f32 / (low / low.mean()), 0, 255).astype(np.uint8)
    blend = cv2.addWeighted(atlas, 0.35, flat, 0.65, 0)

    cv2.imwrite(str(outdir / "skin_atlas_raw.png"), atlas)
    cv2.imwrite(str(outdir / "skin_atlas_flat.png"), blend)
    cv2.imwrite(str(outdir / "skin_uv_mask.png"), (filled * 255).astype(np.uint8))

    stats = {
        "texcoords": int(len(tex)),
        "sampled": n_skin,
        "uv_coverage": float(filled.mean()),
        "atlas_size": S,
    }
    (outdir / "skin_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[out] {stats}")


if __name__ == "__main__":
    main()
