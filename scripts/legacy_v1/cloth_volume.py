"""衣物体积：给身体顶点算一个「沿法线外推量」场。

为什么不另做一层壳：身体网格面约 1–2cm，按面分类切壳必然锯齿。
改成把**身体本身**在衣物区域按法线外推，外推量在衣物边界处平滑衰减到 0，
于是衬衫/短裤的轮廓自带体积，而接缝处没有硬边。

输出 `vertex_inflate.npy`（每个顶点的位移，单位米）。
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", default="1402,1122")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--torso", type=float, default=0.024)
    ap.add_argument("--hips", type=float, default=0.018)
    ap.add_argument("--thigh", type=float, default=0.013)
    ap.add_argument("--lower", type=float, default=0.008)
    args = ap.parse_args()

    W, H = [int(v) for v in args.image_size.split(",")]
    d = np.load(args.mesh)
    V = d["v"].astype(np.float64)
    F = d["faces"].astype(np.int32)
    t = np.asarray(np.load(args.params, allow_pickle=True)["pred_cam_t"]).reshape(-1)[:3].astype(np.float64)
    f = 1795.69

    p = V + t
    z = np.clip(p[:, 2], 1e-3, None)
    uv = np.stack([f * p[:, 0] / z + W / 2.0, f * p[:, 1] / z + H / 2.0], axis=1)

    sk = cv2.imread(args.skin, cv2.IMREAD_GRAYSCALE)
    xi = np.clip(uv[:, 0].astype(int), 0, sk.shape[1] - 1)
    yi = np.clip(uv[:, 1].astype(int), 0, sk.shape[0] - 1)
    is_skin = (sk[yi, xi] > 127)

    w = (~is_skin).astype(np.float32)

    # 顶点邻接平滑：把二值边界摊成 2–3cm 的过渡带
    adj = defaultdict(set)
    for a, b, c in F:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    nb = [np.fromiter(adj[i], dtype=np.int64) for i in range(len(V))]
    for _ in range(args.iters):
        w2 = w.copy()
        for i in range(len(V)):
            if len(nb[i]):
                w2[i] = 0.35 * w[i] + 0.65 * w[nb[i]].mean()
        w = w2
    w = np.clip((w - 0.06) / 0.88, 0, 1)

    # Blender 坐标下的高度决定厚度
    zb = -V[:, 1]
    amp = np.full(len(V), args.lower, np.float32)
    amp[zb > 0.86] = args.thigh
    amp[zb > 1.02] = args.hips
    amp[zb > 1.16] = args.torso
    # 头部不要外推
    amp[zb > 1.45] *= 0.0

    disp = (w * amp).astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, disp)
    print(f"[volume] 权重>0.5 的顶点 {float((w>0.5).mean()):.1%}  "
          f"位移 中位 {np.median(disp[disp>0]):.4f} m  最大 {disp.max():.3f} m")


if __name__ == "__main__":
    main()
