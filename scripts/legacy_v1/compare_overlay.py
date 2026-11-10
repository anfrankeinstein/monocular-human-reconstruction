"""把 baseline / 方案1 / 方案3 的网格投影叠加到照片，做视觉对比。"""
import argparse
from pathlib import Path

import cv2
import numpy as np

COLORS = {"base": (120, 200, 120), "s1": (60, 60, 230), "s3": (230, 150, 40)}


def project(v, cam_t, f, cx, cy):
    p = v + cam_t
    z = np.clip(p[:, 2], 1e-3, None)
    return np.stack([f * p[:, 0] / z + cx, f * p[:, 1] / z + cy], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    sources = [("base", args.base),
               ("s1", str(Path(args.fit_dir) / "params_s1.npz")),
               ("s3", str(Path(args.fit_dir) / "params_s3.npz"))]

    canvas = img.copy()
    canvas_sep = {}
    stats = {}
    for name, path in sources:
        if not Path(path).exists():
            print(f"[skip] {name} missing")
            continue
        d = np.load(path, allow_pickle=True)
        v = np.asarray(d["pred_vertices"]).reshape(-1, 3)
        f = float(np.asarray(d["focal_length"]).reshape(-1)[0]) if "focal_length" in d.files else 1795.69
        t = np.asarray(d["pred_cam_t"]).reshape(-1)[:3]
        uv = project(v, t, f, cx, cy)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pts = uv[inside]
        stats[name] = {
            "u": [float(pts[:, 0].min()), float(pts[:, 0].max())],
            "v": [float(pts[:, 1].min()), float(pts[:, 1].max())],
            "width_px": float(pts[:, 0].max() - pts[:, 0].min()),
            "height_px": float(pts[:, 1].max() - pts[:, 1].min()),
        }
        sep = img.copy()
        for x, y in pts[::2]:
            cv2.circle(canvas, (int(x), int(y)), 1, COLORS[name], -1)
            cv2.circle(sep, (int(x), int(y)), 1, COLORS[name], -1)
        canvas_sep[name] = sep

    cv2.imwrite(str(Path(args.out) / "overlay_all.png"), canvas)
    for name, im in canvas_sep.items():
        cv2.imwrite(str(Path(args.out) / f"overlay_{name}.png"), im)

    print("[bbox]")
    for k, v in stats.items():
        print(f"  {k}: u {v['u'][0]:.0f}..{v['u'][1]:.0f} (w={v['width_px']:.0f})  v {v['v'][0]:.0f}..{v['v'][1]:.0f} (h={v['height_px']:.0f})")

    import json
    (Path(args.out) / "bbox_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
