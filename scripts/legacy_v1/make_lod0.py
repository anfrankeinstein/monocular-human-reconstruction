"""把 LOD1 的拟合结果上采样到 LOD0 拓扑，导出带 UV 的网格。

原理：MHR 仓库自带 `tools/mhr_LOD_conversion/lod1_to_lod0_mapping.npz`，
里面是"LOD0 每个顶点 = LOD1 某个三角形的重心坐标组合"，所以
**S1 推理与 S3.5 体态校正都不用重跑**，直接对已拟合的 LOD1 顶点做插值即可。

用法（在 envs/mhr0 里、并把 Library/bin 加进 PATH）：
  python scripts/make_lod0.py --assets models/mhr-assets/assets \
      --params work/q004-base/fitA/paramsA_s3.npz --out work/q004-base/tex_v3/mesh_uv_lod0.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--mapping", default=r"Q:/stb-assets/repos/MHR/tools/mhr_LOD_conversion/lod1_to_lod0_mapping.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-lod", type=int, default=1)
    ap.add_argument("--dst-lod", type=int, default=0)
    args = ap.parse_args()

    import torch
    from mhr.mhr import MHR

    dev = torch.device("cpu")
    m1 = MHR.from_files(folder=Path(args.assets), device=dev, lod=args.src_lod,
                        wants_pose_correctives=False)
    m0 = MHR.from_files(folder=Path(args.assets), device=dev, lod=args.dst_lod,
                        wants_pose_correctives=False)
    mesh1, mesh0 = m1.character.mesh, m0.character.mesh
    print(f"[lod] src L{args.src_lod}: {mesh1.vertices.shape} / {mesh1.faces.shape}")
    print(f"[lod] dst L{args.dst_lod}: {mesh0.vertices.shape} / {mesh0.faces.shape}")
    print("[lod] mesh0 属性:", [a for a in dir(mesh0) if not a.startswith("_")][:20])

    def to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    src_faces = to_np(mesh1.faces).astype(np.int64)
    mapping = dict(np.load(args.mapping))
    tri_ids = mapping["triangle_ids"].astype(np.int64)
    bary = mapping["baryc_coords"].astype(np.float64)
    print(f"[lod] mapping: triangle_ids {tri_ids.shape}, baryc {bary.shape}")

    d = np.load(args.params, allow_pickle=True)
    verts1 = np.asarray(d["pred_vertices"], dtype=np.float64).reshape(-1, 3)
    assert len(verts1) == len(mesh1.vertices), f"{len(verts1)} != {len(mesh1.vertices)}"

    tri = verts1[src_faces[tri_ids]]
    verts0 = np.einsum("ijk,ij->ik", tri, bary)
    print(f"[lod] 插值得到 {verts0.shape}")

    out = {"v": verts0.astype(np.float32),
           "faces": to_np(mesh0.faces).astype(np.int32)}
    tex = getattr(mesh0, "texcoords", None)
    tfaces = getattr(mesh0, "texcoord_faces", None)
    if tex is not None and tfaces is not None:
        out["tex"] = to_np(tex).astype(np.float32)
        out["tfaces"] = to_np(tfaces).astype(np.int32)
        print(f"[lod] UV texcoords {out['tex'].shape} texcoord_faces {out['tfaces'].shape}")
    else:
        print("[lod] ⚠ LOD0 mesh 没有 texcoords / texcoord_faces")

    for k in ("pred_cam_t", "focal_length", "shape_params"):
        if k in d.files:
            out[k] = np.asarray(d[k])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[out] {args.out}  keys={list(out)}")
    print(json.dumps({"verts": int(len(verts0)), "faces": int(len(out["faces"]))}))


if __name__ == "__main__":
    main()
