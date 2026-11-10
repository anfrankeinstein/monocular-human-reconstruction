"""方案 A：用蒙皮权重做顶点级部位归属，再按裸露皮肤宽度收窄体型。

与 fit_shape.py 的区别：不做 2D 矩形取点，而是
  1) 投影 127 关节，与 mediapipe 的肘/腕/膝/踝匹配，拿到关节索引
  2) 用 skin_indices/weights 把每个顶点归给主导关节
  3) 只统计属于该肢体的顶点的横向跨度
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

MHR_PT = r"Q:\stb-assets\models\sam-3d-body-dinov3\assets\mhr_model.pt"

MP_JOINTS = {"L_elbow": 13, "L_wrist": 15, "R_elbow": 14, "R_wrist": 16,
             "L_knee": 25, "L_ankle": 27, "R_knee": 26, "R_ankle": 28}

LIMBS = [("L_upper_arm", "L_elbow", "L_wrist"),
         ("R_upper_arm", "R_elbow", "R_wrist"),
         ("L_shank", "L_knee", "L_ankle"),
         ("R_shank", "R_knee", "R_ankle")]


def load_model():
    m = torch.jit.load(MHR_PT, map_location="cpu")
    m.eval()
    return m


def forward_all(m, shape, model_params, expr):
    with torch.no_grad():
        v, skel = m(shape, model_params, expr)
    v = v / 100.0
    v[:, :, 1] *= -1
    v[:, :, 2] *= -1
    return v


def project(p, cam_t, f, cx, cy):
    q = p + torch.as_tensor(cam_t, dtype=p.dtype)
    z = q[..., 2].clamp(min=1e-3)
    return torch.stack([f * q[..., 0] / z + cx, f * q[..., 1] / z + cy], dim=-1)


def match_joints(juv, mp_pts, w, valid_joints=None):
    """把 mediapipe 的关节位置匹配到 MHR 的 127 个关节（同侧最近，仅限有蒙皮的主关节）。"""
    out = {}
    for name, idx in MP_JOINTS.items():
        if idx not in mp_pts:
            continue
        target = mp_pts[idx]
        left_side = name.startswith("L_")
        d = np.linalg.norm(juv - target[None, :], axis=1)
        mask = np.ones(len(d), dtype=bool)
        if left_side:
            mask &= juv[:, 0] >= w * 0.5
        else:
            mask &= juv[:, 0] < w * 0.5
        if valid_joints is not None:
            vmask = np.zeros(len(d), dtype=bool)
            vmask[list(valid_joints)] = True
            mask &= vmask
        if not mask.any():
            continue
        dd = np.where(mask, d, np.inf)
        j = int(np.argmin(dd))
        out[name] = {"joint": j, "err_px": float(dd[j])}
    return out


def limb_vertex_sets(lb, n_verts, joint_ids):
    si = lb.skin_indices_flattened.numpy()
    sw = lb.skin_weights_flattened.numpy()
    vi = lb.vert_indices_flattened.numpy()
    best_j = np.full(n_verts, -1, dtype=np.int64)
    best_w = np.zeros(n_verts, dtype=np.float32)
    for v, j, wt in zip(vi, si, sw):
        if wt > best_w[v]:
            best_w[v] = wt
            best_j[v] = j
    return best_j, best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--kpts", required=True)
    ap.add_argument("--skin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focal", type=float, default=1795.69)
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--mode", choices=["s1", "s3"], default="s1")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reg", type=float, default=0.12)
    ap.add_argument("--shrink", type=float, default=1.0, help="目标=皮肤宽度*shrink")
    args = ap.parse_args()

    d = np.load(args.params, allow_pickle=True)
    shape0 = torch.as_tensor(d["shape_params"], dtype=torch.float32)
    model_params = torch.as_tensor(d["mhr_model_params"], dtype=torch.float32)
    expr = torch.as_tensor(d["expr_params"], dtype=torch.float32)
    cam_t = torch.as_tensor(d["pred_cam_t"], dtype=torch.float32)
    j3 = np.asarray(d["pred_joint_coords"], dtype=np.float32)

    kpts = json.loads(Path(args.kpts).read_text(encoding="utf-8"))
    mp_pts = {p["i"]: np.array([p["x"], p["y"]], dtype=np.float32)
              for p in kpts["pose"] if p["v"] > 0.3}
    skin = cv2.imread(args.skin, cv2.IMREAD_GRAYSCALE)
    h, w = skin.shape
    f, cx, cy = args.focal, w / 2.0, h / 2.0

    m = load_model()
    lb = m.character_torch.linear_blend_skinning
    n_verts = int(m.character_torch.mesh.rest_vertices.shape[0])

    juv = project(torch.as_tensor(j3), cam_t, f, cx, cy).numpy()
    valid_joints = set(lb.skin_indices_flattened.numpy().tolist())
    print(f"[skinning] 有蒙皮的关节 {len(valid_joints)} / 127")
    matched = match_joints(juv, mp_pts, w, valid_joints)
    print("[joint match]")
    for k, v in matched.items():
        print(f"  {k}: joint#{v['joint']} err={v['err_px']:.1f}px")

    best_j, best_w = limb_vertex_sets(lb, n_verts, matched)

    def measure(shape_vec):
        v = forward_all(m, shape_vec.view(1, -1), model_params.view(1, -1), expr.view(1, -1))[0]
        uv = project(v, cam_t, f, cx, cy).numpy()
        vn = v.numpy()
        res = {}
        for limb, a, b in LIMBS:
            if a not in matched or b not in matched:
                continue
            ja, jb = matched[a]["joint"], matched[b]["joint"]
            p0, p1 = j3[ja], j3[jb]
            axis = p1 - p0
            L3 = float(np.linalg.norm(axis))
            if L3 < 1e-3:
                res[limb] = None
                continue
            axis = axis / L3
            rel = vn - p0
            t = rel @ axis
            perp3 = rel - np.outer(t, axis)
            d3 = np.linalg.norm(perp3, axis=1)
            sel = (d3 < max(0.09, L3 * 0.22)) & (t > L3 * 0.2) & (t < L3 * 0.8)
            if sel.sum() < 25:
                res[limb] = None
                continue
            pa, pb = juv[ja], juv[jb]
            dirv = pb - pa
            L2 = float(np.linalg.norm(dirv))
            if L2 < 1e-3:
                res[limb] = None
                continue
            dirv = dirv / L2
            n = np.array([-dirv[1], dirv[0]], dtype=np.float32)
            pts = uv[sel]
            perp2 = (pts - (pa + pb) / 2) @ n
            res[limb] = float(np.percentile(perp2, 97) - np.percentile(perp2, 3))
        return res

    base = measure(shape0)
    skin_w = {}
    for limb, a, b in LIMBS:
        if a not in matched or b not in matched:
            continue
        pa, pb = mp_pts[MP_JOINTS[a]], mp_pts[MP_JOINTS[b]]
        dirv = pb - pa
        L = float(np.linalg.norm(dirv))
        dirv = dirv / L
        n = np.array([-dirv[1], dirv[0]], dtype=np.float32)
        mid = (pa + pb) / 2
        vals = []
        for t in np.linspace(-L * 0.25, L * 0.25, 9):
            c = mid + dirv * t
            hits = []
            for s in np.arange(-70, 70, 1.0):
                q = c + n * s
                x, y = int(round(q[0])), int(round(q[1]))
                if 0 <= y < h and 0 <= x < w and skin[y, x] > 0:
                    hits.append(s)
            if hits:
                vals.append(max(hits) - min(hits))
        skin_w[limb] = float(np.median(vals)) if vals else None

    print("[baseline] 归属顶点跨度 vs 皮肤宽度")
    for limb in skin_w:
        bw = base.get(limb)
        sw_ = skin_w[limb]
        if bw is None or sw_ is None:
            print(f"  {limb}: n/a")
            continue
        print(f"  {limb}: cloud={bw:.1f} skin={sw_:.1f} ratio={bw/sw_:.2f}")

    from scipy.optimize import minimize

    factors = args.factors
    targets = {k: v for k, v in skin_w.items() if v and base.get(k)}

    def obj(delta):
        shape = torch.cat([shape0[:factors] + torch.as_tensor(delta, dtype=torch.float32), shape0[factors:]])
        cur = measure(shape)
        err, n = 0.0, 0
        for k, sw_ in targets.items():
            if cur.get(k) is None:
                continue
            err += (cur[k] - sw_ * args.shrink) ** 2
            n += 1
        err = err / max(n, 1) / 400.0
        reg = args.reg * float((np.asarray(delta) ** 2).sum()) / max(factors, 1)
        return err + reg

    print(f"[opt] mode={args.mode} factors={factors} shrink={args.shrink} reg={args.reg}")
    res = minimize(obj, np.zeros(factors), method="Powell",
                   bounds=[(-1.2, 1.2)] * factors,
                   options={"maxiter": args.iters, "xtol": 1e-3, "ftol": 1e-4})
    print(f"[opt] loss {res.fun:.4f} nfev={res.nfev}")

    new_shape = torch.cat([shape0[:factors] + torch.as_tensor(res.x, dtype=torch.float32), shape0[factors:]])
    after = measure(new_shape)
    print("[after]")
    for limb in targets:
        print(f"  {limb}: cloud={after.get(limb)} target={targets[limb]*args.shrink:.1f}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    v_new = forward_all(m, new_shape.view(1, -1), model_params.view(1, -1), expr.view(1, -1))[0].numpy()
    np.savez(outdir / f"paramsA_{args.mode}.npz",
             shape_params=new_shape.numpy(), pred_vertices=v_new,
             pred_cam_t=cam_t.numpy(), focal_length=np.array([f]), delta=res.x)
    (outdir / f"widthsA_{args.mode}.json").write_text(json.dumps({
        "baseline": base, "after": after, "skin": skin_w,
        "matched": {k: v["joint"] for k, v in matched.items()},
        "delta": res.x.tolist()}, indent=1), encoding="utf-8")
    print(f"[done] {outdir}")


if __name__ == "__main__":
    main()
