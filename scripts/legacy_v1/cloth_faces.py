"""预计算每个面的「朝向 / 是否皮肤」，供 Blender 脚本读取（Blender 里没有 cv2）。"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--image-size", default="1402,1122")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    W, H = [int(v) for v in args.image_size.split(",")]
    d = np.load(args.mesh)
    V = d["v"].astype(np.float64)
    F = d["faces"].astype(np.int32)

    t = np.asarray(np.load(args.params, allow_pickle=True)["pred_cam_t"]).reshape(-1)[:3].astype(np.float64)
    f = 1795.69
    p = V + t
    z = np.clip(p[:, 2], 1e-3, None)
    uv2d = np.stack([f * p[:, 0] / z + W / 2.0, f * p[:, 1] / z + H / 2.0], axis=1)
    cen2d = uv2d[F].mean(axis=1)

    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    cen3d = V[F].mean(axis=1)
    view = -cen3d / (np.linalg.norm(cen3d, axis=1, keepdims=True) + 1e-9)
    facing = ((n * view).sum(axis=1) > 0.05).astype(np.uint8)

    sk = cv2.imread(args.skin, cv2.IMREAD_GRAYSCALE)
    xi = np.clip(cen2d[:, 0].astype(int), 0, sk.shape[1] - 1)
    yi = np.clip(cen2d[:, 1].astype(int), 0, sk.shape[0] - 1)
    is_skin = (sk[yi, xi] > 127).astype(np.uint8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, facing=facing, is_skin=is_skin)
    np.save(str(Path(args.out).with_name("skin_mask.npy")), (sk > 127).astype(np.uint8))
    print(f"[faces] facing {facing.sum()}  skin {int((facing & is_skin).sum())}  "
          f"clothing {int((facing & (1 - is_skin)).sum())}")


if __name__ == "__main__":
    main()
