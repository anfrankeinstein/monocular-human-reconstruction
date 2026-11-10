"""求头部的局部坐标系（origin + 三轴），供头发在"头的坐标系"里生成。

为什么要这一步：`build_hair_cards` 的 az/el 原先用世界轴（-Y 前、+Z 上），
但照片里她的头是歪的 —— 发际线在世界 z 上是水平线，在头的局部系里就是斜的，
于是刘海一侧高一侧低、丸子也不在中轴两侧。

输出 head_frame.npz：center(3) / right(3) / fwd(3) / up(3) / radius(3)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--face-mask", required=True, help="2D 人脸区域 mask（与 --scale 匹配）")
    ap.add_argument("--image-size", default="1402,1122")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    W = int(args.image_size.split(",")[0]) * args.scale
    H = int(args.image_size.split(",")[1]) * args.scale

    dm = np.load(args.mesh)
    V = np.asarray(dm["v"], dtype=np.float64).reshape(-1, 3)
    B = np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1)      # → Blender 系

    t = np.asarray(np.load(args.params, allow_pickle=True)["pred_cam_t"]).reshape(-1)[:3]
    f = 1795.69 * args.scale

    crown = B[int(np.argmax(B[:, 2]))]
    # 用"绕头轴的竖直圆柱"取头簇 —— 球形取法会把举在脸侧的手也圈进来
    rxy = np.linalg.norm(B[:, :2] - crown[:2], axis=1)
    sel = (rxy < 0.086) & (B[:, 2] > crown[2] - 0.175)
    head_v = B[sel]
    print(f"[frame] 头簇 {len(head_v)} 顶点（圆柱半径 86mm）")

    # up：头簇 PCA 的第一主轴（椭球头的长轴≈竖直），再用世界 +Z 定符号
    c0 = head_v.mean(axis=0)
    X = head_v - c0
    w, vec = np.linalg.eigh(X.T @ X)
    up = vec[:, np.argmax(w)]
    if up[2] < 0:
        up = -up
    up /= np.linalg.norm(up)
    head_c = crown - 0.085 * up

    # 面部质心：投影到 2D 后采样人脸 mask
    mask = cv2.imread(args.face_mask, cv2.IMREAD_GRAYSCALE)
    p = V + t
    z = np.clip(p[:, 2], 1e-3, None)
    u = np.clip((f * p[:, 0] / z + W / 2).astype(int), 0, mask.shape[1] - 1)
    vv = np.clip((f * p[:, 1] / z + H / 2).astype(int), 0, mask.shape[0] - 1)
    hit = mask[vv, u] > 127
    hit &= np.linalg.norm(B - crown, axis=1) < 0.16
    face_c = B[hit].mean(axis=0)
    print(f"[frame] 人脸顶点 {int(hit.sum())}  质心 {face_c.round(4)}")

    fwd = face_c - head_c
    fwd -= up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    fwd = np.cross(right, up)      # 正交化收尾

    hv = head_v - head_c
    loc = np.stack([hv @ right, hv @ fwd, hv @ up], axis=1)
    radius = np.array([np.abs(loc[:, 0]).max(), np.abs(loc[:, 1]).max(), np.abs(loc[:, 2]).max()])

    print(f"[frame] center {head_c.round(4)}")
    print(f"[frame] right  {right.round(4)}")
    print(f"[frame] fwd    {fwd.round(4)}")
    print(f"[frame] up     {up.round(4)}")
    print(f"[frame] 半径   {radius.round(4)}")
    print(f"[frame] 与世界的夹角：up 偏离 +Z {np.degrees(np.arccos(np.clip(up[2], -1, 1))):.1f}°，"
          f"fwd 偏离 -Y {np.degrees(np.arccos(np.clip(-fwd[1], -1, 1))):.1f}°")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, center=head_c, right=right, fwd=fwd, up=up, radius=radius)
    print("[out]", args.out)


if __name__ == "__main__":
    main()
