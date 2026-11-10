"""合成角色图集：皮肤 texel 用皮肤图集，其余（衣物）用服装图集。

这样一份图集同时承载皮肤和衣服，身体网格直接就是"穿好衣服的人"，
不需要靠面级分类去切衣服壳（面级分类在 1–2cm 的面尺度上必然锯齿）。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from project_skin4 import build_atlas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--skin-atlas", required=True)
    ap.add_argument("--cloth-atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--feather", type=int, default=9)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--mesh", default=None)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    _, _, flag, n, nf, _ = build_atlas(args.params, args.image, args.skin, args.size,
                                    scale=args.scale, mesh=args.mesh)
    skin_atlas = cv2.imread(args.skin_atlas)
    cloth_atlas = cv2.imread(args.cloth_atlas)
    if skin_atlas.shape != cloth_atlas.shape:
        raise SystemExit(f"图集尺寸不一致 {skin_atlas.shape} vs {cloth_atlas.shape}")

    m = (flag > 0).astype(np.uint8)
    # 交界处羽化，避免皮肤与衣物之间出现硬边
    k = args.feather * 2 + 1
    soft = cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)[..., None]
    out = (skin_atlas.astype(np.float32) * soft + cloth_atlas.astype(np.float32) * (1 - soft))
    out = np.clip(out, 0, 255).astype(np.uint8)

    cv2.imwrite(str(outdir / "char_atlas.png"), out)
    print(f"[compose] 皮肤 texel {m.mean():.2%}  羽化 {args.feather}px  → {outdir / 'char_atlas.png'}")


if __name__ == "__main__":
    main()
