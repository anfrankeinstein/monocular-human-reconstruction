"""把 MHR 顶点按估计相机投影回照片，检查拟合是否对齐。"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focal", type=float, default=None, help="像素焦距；缺省时从 npz 读取")
    args = ap.parse_args()

    d = np.load(args.params, allow_pickle=True)
    v = np.asarray(d["pred_vertices"]).reshape(-1, 3)
    if args.focal is not None:
        f = args.focal
    else:
        f = float(np.asarray(d["focal_length"]).reshape(-1)[0])
    t = np.asarray(d["pred_cam_t"]).reshape(-1)[:3]

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    print(f"[in] verts={v.shape} focal={f:.1f} cam_t={t.round(3).tolist()} img={w}x{h}")
    print(f"[range] x {v[:,0].min():.3f}..{v[:,0].max():.3f}  y {v[:,1].min():.3f}..{v[:,1].max():.3f}  z {v[:,2].min():.3f}..{v[:,2].max():.3f}")

    for name, offset in (("world", np.zeros(3)), ("plus_t", t)):
        p = v + offset
        z = p[:, 2]
        if np.any(z <= 1e-6):
            print(f"[{name}] 存在非正深度，跳过")
            continue
        u = f * p[:, 0] / z + cx
        ww = f * p[:, 1] / z + cy
        vis = img.copy()
        inside = (u >= 0) & (u < w) & (ww >= 0) & (ww < h)
        pts = np.stack([u[inside], ww[inside]], axis=1).astype(np.int32)
        for x, y in pts[::3]:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
        cov = inside.mean()
        print(f"[{name}] 投影像素内比例 {cov:.2%}, u {u.min():.0f}..{u.max():.0f}, v {ww.min():.0f}..{ww.max():.0f}")
        out = Path(args.out).with_name(Path(args.out).stem + f"_{name}.png")
        cv2.imwrite(str(out), vis)


if __name__ == "__main__":
    main()
