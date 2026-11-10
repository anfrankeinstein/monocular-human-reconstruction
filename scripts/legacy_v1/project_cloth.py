"""S8.5 服装图集：把照片反投影到身体 UV，但**不做皮肤填充**。

关键思路：身体上被衣物覆盖的那些面，照片里对应的像素就是衣物本身。
直接把非皮肤面的投影结果保留下来，服装片就用这份图集，
印花/牛仔纹会自然跟着身体表面走，不需要手工裁贴图。

未被朝向面覆盖的 UV 区域（背面）用最近已绘 texel 填补，避免黑洞。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).parent))
from project_skin4 import build_atlas


def nearest_fill(img, known):
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return img[tuple(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--delight", type=float, default=0.0)
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--sharpen-radius", type=float, default=2.0)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    S = args.size

    atlas, painted, flag, n, nf, _ = build_atlas(args.params, args.image, args.skin, S,
                                                 scale=args.scale, mesh=args.mesh)
    print(f"[warp] 朝向面 {n}/{nf}  覆盖 {painted.mean():.2%}  皮肤 texel {(flag>0).mean():.2%}")

    known = painted > 0
    filled = nearest_fill(atlas, known).astype(np.float32)

    # 轻微平滑未覆盖区，去掉最近邻的硬 Voronoi 边界
    blur = cv2.GaussianBlur(filled, (0, 0), S / 200.0)
    dd = ndimage.distance_transform_edt(known).astype(np.float32)
    w = np.clip(dd / (S / 300.0), 0, 1)[..., None]
    filled = filled * (1 - w) + blur * w
    out = np.clip(filled, 0, 255).astype(np.uint8)

    if args.delight > 0:
        f32 = out.astype(np.float32)
        y = 0.114 * f32[..., 0] + 0.587 * f32[..., 1] + 0.299 * f32[..., 2]
        y_low = np.maximum(cv2.GaussianBlur(y, (0, 0), S / 22.0), 1.0)
        gm = float(y_low.mean()) / y_low
        flat = np.clip(f32 * gm[..., None], 0, 255)
        out = np.clip(f32 * (1 - args.delight) + flat * args.delight, 0, 255).astype(np.uint8)
        print(f"[delight] 衣物去光照 {args.delight}")
    if args.sharpen > 0:
        f32 = out.astype(np.float32)
        blur = cv2.GaussianBlur(f32, (0, 0), args.sharpen_radius)
        out = np.clip(f32 + args.sharpen * (f32 - blur), 0, 255).astype(np.uint8)
        print(f"[sharpen] amount={args.sharpen}")

    cv2.imwrite(str(outdir / "cloth_atlas.png"), out)
    cv2.imwrite(str(outdir / "cloth_uv_painted.png"), painted * 255)
    cv2.imwrite(str(outdir / "cloth_skin_flag.png"), (flag > 0).astype(np.uint8) * 255)

    stats = {"faces": int(nf), "facing": int(n), "uv_coverage": float(painted.mean()),
             "skin_texels": float((flag > 0).mean()), "atlas": S}
    (outdir / "cloth_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("[out]", json.dumps(stats))


if __name__ == "__main__":
    main()
