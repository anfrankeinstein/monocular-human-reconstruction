"""M1-A (envs/mhr)：干净前景分割 + 基于真实蒙皮归属的可微体态精修。

改进点（对应 OPTIMIZATION_SPEC.md M1 / RC-5）：
1. 用 models/selfie_segmenter.tflite 替代 GrabCut，彻底消除床单与地毯前景粘连。
2. 真正启用 MHR 骨骼蒙皮权重归属过滤肢体顶点，并用 PyTorch Adam 可微优化 shape_params，
   修复旧版 fit_shape2.py 中 limb_vertex_sets 未生效、前臂目标宽度虚高至 104px 的问题。
3. 完整透传 fit/params.npz 的全部 17 个键值（含 mhr_model_params / expr_params / 骨骼矩阵）。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def run_selfie_segmenter(img_bgr: np.ndarray, model_path: str, thr: float = 0.45) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    opts = vision.ImageSegmenterOptions(
        base_options=mp_py.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        conf = np.asarray(res.confidence_masks[0].numpy_view()).squeeze().copy()

    mask = (conf > thr).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nlab > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == best).astype(np.uint8)
    return mask, conf


def measure_fg_limb_width(fg_mask: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                          t0: float = 0.30, t1: float = 0.70, max_half: int = 65) -> float | None:
    """在干净前景 mask 上沿肢体骨骼中段法向扫描真实宽度（从中心向外找连续前景段，防背景粘连）。"""
    h, w = fg_mask.shape
    vec = p_b - p_a
    L = float(np.linalg.norm(vec))
    if L < 10.0:
        return None
    d = vec / L
    n = np.array([-d[1], d[0]], dtype=np.float32)

    widths = []
    for t in np.linspace(t0, t1, 11):
        c = p_a + vec * float(t)
        # 从中心向正负两侧走，遇到前景=0即停（只量当前肢体截面，不跨空白连到别的部位）
        s_pos = 0.0
        for s in range(0, max_half):
            q = c + n * s
            x, y = int(round(float(q[0]))), int(round(float(q[1])))
            if 0 <= x < w and 0 <= y < h and fg_mask[y, x] > 0:
                s_pos = float(s)
            elif s > 6:
                break
        s_neg = 0.0
        for s in range(0, max_half):
            q = c - n * s
            x, y = int(round(float(q[0]))), int(round(float(q[1])))
            if 0 <= x < w and 0 <= y < h and fg_mask[y, x] > 0:
                s_neg = float(s)
            elif s > 6:
                break
        span = s_pos + s_neg
        if span >= 14.0:
            widths.append(span)
    return float(np.median(widths)) if widths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-2x", default=r"work\q004-base\scene\gpen_2x.png")
    ap.add_argument("--params", default=r"work\q004-base\fit\params.npz")
    ap.add_argument("--mhr-pt", default=r"models\sam-3d-body-dinov3\assets\mhr_model.pt")
    ap.add_argument("--seg-model", default=r"models\selfie_segmenter.tflite")
    ap.add_argument("--out-dir", default=r"work\q004-base\v2")
    ap.add_argument("--factors", type=int, default=10)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--reg", type=float, default=0.18)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img2x = cv2.imread(args.image_2x)
    if img2x is None:
        raise SystemExit(f"无法读取输入图: {args.image_2x}")
    h2, w2 = img2x.shape[:2]
    w1, h1 = w2 // 2, h2 // 2
    img1x = cv2.resize(img2x, (w1, h1), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "ref_1x.png"), img1x)

    fg2x, conf2x = run_selfie_segmenter(img2x, args.seg_model, thr=0.42)
    fg1x = cv2.resize(fg2x, (w1, h1), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "person_2x.png"), fg2x * 255)
    cv2.imwrite(str(out_dir / "person_1x.png"), fg1x * 255)

    # 在干净前景内提取真实裸露暖肤色掩码，防止前臂与后方衬衫袖口粘连导致宽度虚高
    b1, g1, r1 = img1x[..., 0].astype(np.float32), img1x[..., 1].astype(np.float32), img1x[..., 2].astype(np.float32)
    v1_lum = (b1 + g1 + r1) / 3.0
    bare_skin_1x = ((fg1x > 0) & ((r1 - b1) > 22.0) & (r1 > g1 + 6.0) & (v1_lum > 65.0) & (v1_lum < 212.0)).astype(np.uint8)
    bare_skin_1x = cv2.morphologyEx(bare_skin_1x, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bare_skin_1x = cv2.morphologyEx(bare_skin_1x, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite(str(out_dir / "bare_skin_1x.png"), bare_skin_1x * 255)
    print(f"[seg] selfie_segmenter 前景占比: {fg2x.mean():.2%}  裸露肤色占比: {bare_skin_1x.mean():.2%}")

    raw = dict(np.load(args.params, allow_pickle=True))
    f = float(np.asarray(raw.get("focal_length", [1795.69])).reshape(-1)[0])
    cam_t = torch.as_tensor(np.asarray(raw["pred_cam_t"]).reshape(-1)[:3], dtype=torch.float32)
    shape0 = torch.as_tensor(np.asarray(raw["shape_params"]).reshape(-1), dtype=torch.float32)
    model_params = torch.as_tensor(np.asarray(raw["mhr_model_params"]).reshape(1, -1), dtype=torch.float32)
    expr_params = torch.as_tensor(np.asarray(raw["expr_params"]).reshape(1, -1), dtype=torch.float32)

    m = torch.jit.load(args.mhr_pt, map_location="cpu")
    m.eval()
    ct = m.character_torch
    jn = list(ct.skeleton.joint_names)
    lb = ct.linear_blend_skinning
    n_verts = int(ct.mesh.rest_vertices.shape[0])

    # 构建真实的顶点主导关节集合（修复旧版 best_j 未使用的 bug）
    vi = lb.vert_indices_flattened.numpy()
    si = lb.skin_indices_flattened.numpy()
    sw = lb.skin_weights_flattened.numpy()
    j_weights = {}
    for idx, name in enumerate(jn):
        j_weights[name] = np.zeros(n_verts, dtype=np.float32)
    for v_i, j_i, w_i in zip(vi, si, sw):
        j_weights[jn[int(j_i)]][int(v_i)] += float(w_i)

    def group_mask(substrs, thr=0.45):
        tot = np.zeros(n_verts, dtype=np.float32)
        for name, arr in j_weights.items():
            if any(s in name.lower() for s in substrs):
                tot += arr
        return tot >= thr

    # 只选裸露无宽松衣物遮挡的前臂中段与小腿中段做宽度约束
    rest = ct.mesh.rest_vertices.numpy()
    limb_specs = [
        # (名称, 起点关节名, 终点关节名, 顶点归属掩码, 2D采样区间)
        ("L_forearm", "l_lowarm", "l_wrist",
         group_mask(["l_lowarm"]) & (np.abs(rest[:, 0]) > 37.0) & (np.abs(rest[:, 0]) < 49.0),
         (0.28, 0.72)),
        ("R_forearm", "r_lowarm", "r_wrist",
         group_mask(["r_lowarm"]) & (np.abs(rest[:, 0]) > 37.0) & (np.abs(rest[:, 0]) < 49.0),
         (0.28, 0.72)),
        ("L_shank", "l_lowleg", "l_talocrural",
         group_mask(["l_lowleg"]) & (rest[:, 1] > 24.0) & (rest[:, 1] < 42.0),
         (0.22, 0.56)),
        ("R_shank", "r_lowleg", "r_talocrural",
         group_mask(["r_lowleg"]) & (rest[:, 1] > 24.0) & (rest[:, 1] < 42.0),
         (0.22, 0.56)),
    ]

    with torch.no_grad():
        v_init, sk_init = m(shape0.view(1, -1), model_params, expr_params)
    j_init = sk_init[0, :, :3].clone() / 100.0
    j_init[:, 1] *= -1
    j_init[:, 2] *= -1
    jp = j_init + cam_t
    juv = np.stack([
        (f * jp[:, 0] / jp[:, 2] + w1 / 2.0).numpy(),
        (f * jp[:, 1] / jp[:, 2] + h1 / 2.0).numpy(),
    ], axis=1)

    active_limbs = []
    for name, ja_name, jb_name, vmask, (t0, t1) in limb_specs:
        ja, jb = jn.index(ja_name), jn.index(jb_name)
        w_obs = measure_fg_limb_width(bare_skin_1x, juv[ja], juv[jb], t0=t0, t1=t1)
        if w_obs is not None and vmask.sum() >= 20:
            active_limbs.append((name, ja, jb, torch.as_tensor(np.where(vmask)[0], dtype=torch.long), w_obs))
            print(f"[target] {name:10s}: verts={int(vmask.sum()):4d}  obs_width={w_obs:5.1f}px")

    delta = torch.zeros(args.factors, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.05)

    def eval_widths(d_vec):
        sh = torch.cat([shape0[:args.factors] + d_vec, shape0[args.factors:]]).view(1, -1)
        v_raw, sk_raw = m(sh, model_params, expr_params)
        v = v_raw[0] / 100.0
        v = torch.stack([v[:, 0], -v[:, 1], -v[:, 2]], dim=1)
        j = sk_raw[0, :, :3] / 100.0
        j = torch.stack([j[:, 0], -j[:, 1], -j[:, 2]], dim=1)

        vp = v + cam_t
        uv = torch.stack([f * vp[:, 0] / vp[:, 2] + w1 / 2.0, f * vp[:, 1] / vp[:, 2] + h1 / 2.0], dim=1)
        jp2 = j + cam_t
        juv2 = torch.stack([f * jp2[:, 0] / jp2[:, 2] + w1 / 2.0, f * jp2[:, 1] / jp2[:, 2] + h1 / 2.0], dim=1)

        widths = {}
        for name, ja, jb, vidx, _ in active_limbs:
            pa, pb = juv2[ja], juv2[jb]
            dirv = pb - pa
            dirv = dirv / (torch.norm(dirv) + 1e-6)
            normal = torch.stack([-dirv[1], dirv[0]])
            mid = 0.5 * (pa + pb)
            perp = (uv[vidx] - mid) @ normal
            # 可微软分位跨度（96% - 4%）
            k_hi = max(1, int(round(0.04 * len(vidx))))
            hi = torch.topk(perp, k_hi, largest=True).values[-1]
            lo = torch.topk(perp, k_hi, largest=False).values[-1]
            widths[name] = hi - lo
        return widths, v, sk_raw[0], sh[0]

    with torch.no_grad():
        w_base, _, _, _ = eval_widths(delta)
    for name, _, _, _, w_obs in active_limbs:
        print(f"[before] {name:10s}: mesh={float(w_base[name]):5.1f}px  obs={w_obs:5.1f}px")

    for step in range(args.steps):
        opt.zero_grad()
        widths, _, _, _ = eval_widths(delta)
        loss_fit = sum(((widths[name] - w_obs) / 20.0) ** 2 for name, _, _, _, w_obs in active_limbs) / max(len(active_limbs), 1)
        loss_reg = args.reg * (delta ** 2).mean()
        loss = loss_fit + loss_reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-1.1, 1.1)

    with torch.no_grad():
        w_after, v_final, sk_final, sh_final = eval_widths(delta)

    stats = {"before": {}, "after": {}, "target": {}, "delta": delta.detach().numpy().round(4).tolist()}
    for name, _, _, _, w_obs in active_limbs:
        b_val, a_val = float(w_base[name]), float(w_after[name])
        stats["before"][name] = round(b_val, 2)
        stats["after"][name] = round(a_val, 2)
        stats["target"][name] = round(w_obs, 2)
        print(f"[after ] {name:10s}: {b_val:5.1f}px -> {a_val:5.1f}px (target {w_obs:5.1f}px)")

    # 完整透传原始 params 的所有键值，并更新 shape_params / pred_vertices / pred_joint_coords
    j_sam = sk_final[:, :3].numpy() / 100.0
    j_sam[:, 1] *= -1
    j_sam[:, 2] *= -1
    out_dict = dict(raw)
    out_dict["shape_params"] = sh_final.numpy().astype(np.float32)
    out_dict["pred_vertices"] = v_final.numpy().astype(np.float32)
    out_dict["pred_joint_coords"] = j_sam.astype(np.float32)
    out_dict["focal_length"] = np.array([f], dtype=np.float32)
    out_dict["orig_img_size"] = np.array([w1, h1], dtype=np.int32)
    np.savez(out_dir / "params_v2.npz", **out_dict)
    (out_dir / "fit_stats_v2.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[done] 已保存完整参数与掩码至 {out_dir}")


if __name__ == "__main__":
    main()
