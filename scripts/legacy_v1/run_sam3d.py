"""SAM 3D Body 单图推理：直传 bbox，绕过检测器与 detectron2。

用法（在 Q:\\stb-assets 下）：
  Q:\\stb-assets\\envs\\mhr\\Scripts\\python.exe scripts\\run_sam3d.py ^
      --image <png> ^
      --ckpt Q:\\stb-assets\\models\\sam-3d-body-dinov3\\model.ckpt ^
      --mhr  Q:\\stb-assets\\models\\sam-3d-body-dinov3\\assets\\mhr_model.pt ^
      --bbox 200,60,1100,1050 ^
      --out  work\\q004-base\\fit
"""
import argparse
import sys
from pathlib import Path

REPO = Path(r"Q:\stb-assets\repos\sam-3d-body")
sys.path.insert(0, str(REPO))

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--ckpt", required=True, help="model.ckpt")
    ap.add_argument("--mhr", required=True, help="assets/mhr_model.pt")
    ap.add_argument("--bbox", default="200,60,1100,1050", help="x1,y1,x2,y2 像素")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--mode", default="full", choices=["full", "body", "hand"],
                    help="full=身体+手部解码器")
    return ap.parse_args()


def main():
    args = parse_args()

    import torch
    from sam_3d_body.build_models import load_sam_3d_body
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"[env] gpu {torch.cuda.get_device_name(0)}")

    model, cfg = load_sam_3d_body(args.ckpt, device=device, mhr_path=args.mhr)
    est = SAM3DBodyEstimator(model, cfg)

    bbox = np.array([[float(x) for x in args.bbox.split(",")]], dtype=np.float32)
    print(f"[infer] bbox={bbox.tolist()} mode={args.mode}")
    out = est.process_one_image(args.image, bboxes=bbox, inference_type=args.mode)[0]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    np.savez(outdir / "params.npz", **arrays)

    verts = np.asarray(out["pred_vertices"]).reshape(-1, 3)
    faces = np.asarray(est.faces)
    with open(outdir / "body_lod0.obj", "w", encoding="utf-8") as fh:
        for p in verts:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for tri in faces:
            fh.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

    print(f"[out] vertices={verts.shape} faces={faces.shape}")
    for key in ("focal_length", "pred_cam_t", "scale_params", "lhand_bbox", "rhand_bbox"):
        if key in out:
            print(f"[out] {key} = {np.asarray(out[key]).tolist()}")
    print(f"[out] hand_pose shape = {np.shape(out.get('hand_pose_params'))}")
    print(f"[done] {outdir}")


if __name__ == "__main__":
    main()
