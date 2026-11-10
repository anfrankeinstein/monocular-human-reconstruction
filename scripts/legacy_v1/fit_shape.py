"""体态贴合比对：baseline / 方案1（裸露区宽度约束） / 方案3（躯干收窄正则）。

优化变量：MHR 的 shape 系数（只用前 N 维 PCA 分量）。
目标：让四肢在照片上的投影宽度贴合裸露皮肤的实际宽度。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

MHR_PT = r"Q:\stb-assets\models\sam-3d-body-dinov3\assets\mhr_model.pt"

LIMB_SEGS = [
    ("L_upper_arm", 11, 13),
    ("L_forearm", 13, 15),
    ("R_upper_arm", 12, 14),
    ("R_forearm", 14, 16),
    ("L_shank", 25, 27),
    ("R_shank", 26, 28),
]


def load_model():
    m = torch.jit.load(MHR_PT, map_location="cpu")
    m.eval()
    return m


def forward_verts(m, shape, model_params, expr):
    with torch.no_grad():
        v, _ = m(shape, model_params, expr)
    v = v / 100.0
    v = v.clone()
    v[:, :, 1] *= -1
    v[:, :, 2] *= -1
    return v


def project(v, cam_t, f, cx, cy):
    p = v + torch.as_tensor(cam_t, dtype=v.dtype)
    z = p[:, 2].clamp(min=1e-3)
    return torch.stack([f * p[:, 0] / z + cx, f * p[:, 1] / z + cy], dim=1)


def band_mask(uv, mid, dirv, half_len, half_w):
    d = uv - mid
    along = d @ dirv
    perp = d[:, 0] * (-dirv[1]) + d[:, 1] * dirv[0]
    return (np.abs(along) < half_len) & (np.abs(perp) < half_w), perp


def skin_width_2d(skin, mid, dirv, half_len, half_w):
    """沿肢体横向扫描皮肤 mask，返回中段平均宽度（像素）。"""
    n = np.array([-dirv[1], dirv[0]], dtype=np.float32)
    widths = []
    for t in np.linspace(-half_len, half_len, 9):
        c = np.asarray(mid) + dirv * t
        vals = []
        for s in np.arange(-half_w, half_w, 1.0):
            q = c + n * s
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= y < skin.shape[0] and 0 <= x < skin.shape[1] and skin[y, x] > 0:
                vals.append(s)
        if vals:
            widths.append(max(vals) - min(vals))
    return float(np.median(widths)) if widths else None


def build_targets(kpts, skin):
    pts = {p["i"]: np.array([p["x"], p["y"]], dtype=np.float32) for p in kpts["pose"] if p["v"] > 0.3}
    targets = {}
    for name, a, b in LIMB_SEGS:
        if a not in pts or b not in pts:
            continue
        p0, p1 = pts[a], pts[b]
        mid = (p0 + p1) / 2
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-3:
            continue
        d = d / L
        w = skin_width_2d(skin, mid, d, L * 0.25, 90.0)
        if w:
            targets[name] = {"mid": mid.tolist(), "dir": d.tolist(), "skin_w": w, "seg_len": L}
    return targets


def cloud_width(uv, mid, dirv, half_len, half_w):
    m, perp = band_mask(uv, mid, dirv, half_len, half_w)
    if int(m.sum()) < 20:
        return None
    sel = perp[m]
    return float(sel.max() - sel.min())


def make_loss(m, shape0, model_params, expr, cam_t, f, cx, cy, targets, mode, factors, torso_penalty):
    params = torch.as_tensor(model_params, dtype=torch.float32)
    ex = torch.as_tensor(expr, dtype=torch.float32)
    base = shape0.clone().detach()

    def loss_fn(delta):
        d = torch.as_tensor(delta, dtype=torch.float32)
        shape = (base + 0.0)
        shape = torch.cat([shape[:factors] + d, shape[factors:]])
        v = forward_verts(m, shape.view(1, -1), params.view(1, -1), ex.view(1, -1))[0]
        uv = project(v, cam_t, f, cx, cy).detach().numpy()

        terms = {}
        for name, t in targets.items():
            mid = np.array(t["mid"], dtype=np.float32)
            dirv = np.array(t["dir"], dtype=np.float32)
            hl = t["seg_len"] * 0.25
            cw = cloud_width(uv, mid, dirv, hl, 110.0)
            if cw is None:
                continue
            terms[name] = (cw - t["skin_w"] * factors_scale) ** 2
        l = sum(terms.values()) / max(len(terms), 1) / 400.0

        if torso_penalty > 0:
            l = l + torso_penalty * float((d * d).sum())

        return l

    return loss_fn


factors_scale = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--kpts", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focal", type=float, default=1795.69)
    ap.add_argument("--factors", type=int, default=8, help="优化的 PCA 分量数")
    ap.add_argument("--mode", choices=["s1", "s3"], default="s1")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    d = np.load(args.params, allow_pickle=True)
    shape0 = torch.as_tensor(d["shape_params"], dtype=torch.float32)
    model_params = d["mhr_model_params"]
    expr = d["expr_params"]
    cam_t = d["pred_cam_t"].astype(np.float64)

    kpts = json.loads(Path(args.kpts).read_text(encoding="utf-8"))
    skin = cv2.imread(args.skin, cv2.IMREAD_GRAYSCALE)
    targets = build_targets(kpts, skin)
    print("[targets]")
    for k, v in targets.items():
        print(f"  {k}: skin_width={v['skin_w']:.1f}px seg={v['seg_len']:.0f}px")

    h, w = skin.shape
    f, cx, cy = args.focal, w / 2.0, h / 2.0

    m = load_model()
    params = torch.as_tensor(model_params, dtype=torch.float32)
    ex = torch.as_tensor(expr, dtype=torch.float32)

    def measure(shape_vec):
        v = forward_verts(m, shape_vec.view(1, -1), params.view(1, -1), ex.view(1, -1))[0]
        uv = project(v, cam_t, f, cx, cy).detach().numpy()
        out = {}
        for name, t in targets.items():
            mid = np.array(t["mid"], dtype=np.float32)
            dirv = np.array(t["dir"], dtype=np.float32)
            cw = cloud_width(uv, mid, dirv, t["seg_len"] * 0.25, 110.0)
            out[name] = cw
        return out

    base_measure = measure(shape0)
    print("[baseline] 点云宽度 vs 皮肤宽度")
    for k in targets:
        print(f"  {k}: cloud={base_measure[k]:.1f} skin={targets[k]['skin_w']:.1f} delta={base_measure[k]-targets[k]['skin_w']:+.1f}")

    from scipy.optimize import minimize

    factors = args.factors
    torso_penalty = 0.03 if args.mode == "s3" else 0.0

    def obj(delta):
        shape = torch.cat([shape0[:factors] + torch.as_tensor(delta, dtype=torch.float32), shape0[factors:]])
        cur = measure(shape)
        err = 0.0
        n = 0
        for k, t in targets.items():
            sw = t["skin_w"] * (0.94 if args.mode == "s3" else 1.0)
            if cur[k] is None:
                continue
            err += (cur[k] - sw) ** 2
            n += 1
        err = err / max(n, 1) / 400.0
        reg = 0.6 * float((delta ** 2).sum()) / max(factors, 1)
        return err + reg + torso_penalty * float((delta ** 2).sum())

    x0 = np.zeros(factors, dtype=np.float64)
    print(f"[opt] mode={args.mode} factors={factors} iters={args.iters} (bounded ±0.8, reg 0.6)")
    res = minimize(obj, x0, method="Powell",
                   bounds=[(-0.8, 0.8)] * factors,
                   options={"maxiter": args.iters, "xtol": 1e-3, "ftol": 1e-4})
    print(f"[opt] loss {res.fun:.4f} nfev={res.nfev}")

    new_shape = torch.cat([shape0[:factors] + torch.as_tensor(res.x, dtype=torch.float32), shape0[factors:]])
    new_measure = measure(new_shape)
    print("[after] 点云宽度 vs 目标")
    for k in targets:
        tgt = targets[k]["skin_w"] * (0.88 if args.mode == "s3" else 1.0)
        print(f"  {k}: cloud={new_measure[k]:.1f} target={tgt:.1f} delta={new_measure[k]-tgt:+.1f}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    v_new = forward_verts(m, new_shape.view(1, -1), params.view(1, -1), ex.view(1, -1))[0].detach().numpy()
    faces = d["faces"] if "faces" in d.files else None
    np.savez(outdir / f"params_{args.mode}.npz",
             shape_params=new_shape.numpy(),
             pred_vertices=v_new,
             pred_cam_t=cam_t,
             focal_length=np.array([f]),
             delta=res.x)
    with open(outdir / f"widths_{args.mode}.json", "w", encoding="utf-8") as fh:
        json.dump({"baseline": {k: base_measure[k] for k in targets},
                   "after": {k: new_measure[k] for k in targets},
                   "skin": {k: targets[k]["skin_w"] for k in targets},
                   "delta": res.x.tolist()}, fh, indent=1)
    print(f"[done] {outdir}")


if __name__ == "__main__":
    main()
