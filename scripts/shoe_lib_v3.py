"""v3 帆布鞋与堆堆白袜解耦几何、独立非重叠 UV 与四通道 PBR 装配库（Converse Chuck 70 风格）。

核心架构特性：
1. 纯 NumPy 确定性参数化鞋楦/鞋底/鞋带/堆堆袜网格与独立非重叠 UV 生成器（供贴图烘焙器与 Blender 装配器零误差共享同一几何真源）：
   - 光滑一体式圆头帆布鞋楦曲面 + 防穿模包络推离，彻底消除 v2 基于裸足顶点膨胀导致的“鞋头印出五趾分趾沟槽”伪影；
   - 取消全部程序化几何贴花（如 v2 悬挑圆柱体 `patch_L/R`、独立黑边环 `foxing_L/R`、独立包头壳 `toecap_L/R` 及碎片鞋带块），
     将全套鞋袜从 24 个零散无 UV 图元收敛为 4 个干净流形 PBR 网格（`shoe_L`, `shoe_R`, `sock_L`, `sock_R`）；
   - 左右鞋帮、左右硫化围条、左右大底/内底、左右交叉扁棉鞋带及蝴蝶结均分配在严格不相交的独立 UV 岛（0.00% UV 重叠）；
   - 白袜拥有独立非重叠 UV 与专属针织棉质 PBR 材质（三维松弛堆堆褶皱环 + 双层卷边罗纹袜口）。
2. 四通道 PBR 材质图（`Albedo` + `Roughness` + `Metallic` + `Tangent Normal`）一次性驱动：
   12oz 重磅帆布织纹、内踝经典圆标五角星、鞋舌/鞋眼襟双排明线/侧翼弧线/后跟竖条、7×2 金属鞋眼孔 + 2 内侧足弓透气金属孔、
   亮面橡胶包头、前掌菱格防撞包边、上下双黑护条（下黑线于前包边接缝精准截断）以及后跟黑色立体 `"CONVERSE ALL ★ STAR"` 橡胶后标。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ==============================================================================
# 1. 独立非重叠 UV 图集分区常量（Boot 4K Atlas & Sock 2K Atlas，严格两两不相交）
# ==============================================================================
SHOE_UV_LAYOUT = {
    "upper_L": (0.020, 0.480, 0.520, 0.940),      # 外靴身与橡胶包头；卷边内衬延伸至 v=0.980
    "upper_L_rim": (0.020, 0.480, 0.940, 0.980),
    "upper_R": (0.520, 0.980, 0.520, 0.940),
    "upper_R_rim": (0.520, 0.980, 0.940, 0.980),
    "foxing_L": (0.020, 0.980, 0.405, 0.490),     # 全周长展开硫化橡胶围条（等比例无拉伸）
    "foxing_R": (0.020, 0.980, 0.305, 0.390),
    "sole_bot_L": (0.020, 0.235, 0.115, 0.285),   # 生胶华夫防滑大底
    "sole_top_L": (0.255, 0.470, 0.115, 0.285),   # 鞋底内沿封盖
    "sole_bot_R": (0.520, 0.735, 0.115, 0.285),
    "sole_top_R": (0.755, 0.970, 0.115, 0.285),
    "laces_L": (0.020, 0.480, 0.020, 0.095),      # 扁棉交叉鞋带 + 蝴蝶结环 + 鞋带头分片无重叠排布
    "laces_R": (0.520, 0.980, 0.020, 0.095),
}

SOCK_UV_LAYOUT = {
    "sock_L": (0.030, 0.475, 0.040, 0.860),
    "sock_L_rim": (0.030, 0.475, 0.860, 0.960),
    "sock_R": (0.525, 0.970, 0.040, 0.860),
    "sock_R_rim": (0.525, 0.970, 0.860, 0.960),
}

# 7 对鞋眼孔与鞋舌结构在靴身局部参数域 (theta, v_t) 上的精确几何常量
EYELET_V_LIST = np.linspace(0.280, 0.895, 7, dtype=np.float64)
EYELET_THETA_ABS = 0.295  # 左右鞋眼中心距正中线角度 (~16.9°)
TONGUE_HALF_THETA = 0.215  # 鞋舌可视槽半角 (~12.3°)
EYESTAY_INNER_THETA = 0.195  # 鞋眼襟内沿台阶角


# ==============================================================================
# 2. 纯 NumPy 几何滤波与多边形工具（无外部 scipy/bpy 强依赖，跨环境一致）
# ==============================================================================
def sam_to_bl_np(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v, dtype=np.float64)
    out[..., 0] = v[..., 0]
    out[..., 1] = v[..., 2]
    out[..., 2] = -v[..., 1]
    return out


def _gauss_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-5:
        return np.array([1.0], dtype=np.float64)
    rad = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_1d(arr: np.ndarray, sigma: float, periodic: bool = False) -> np.ndarray:
    k = _gauss_kernel_1d(sigma)
    pad = len(k) // 2
    if pad == 0:
        return arr.copy()
    mode = "wrap" if periodic else "edge"
    padded = np.pad(arr, pad, mode=mode)
    return np.convolve(padded, k, mode="valid")


def _smooth_2d_grid(grid: np.ndarray, sigma_v: float, sigma_u: float, periodic_u: bool = True) -> np.ndarray:
    """对 (N_v, N_u) 网格做可分离高斯平滑（u 轴支持周期边界，保证圆周接缝 C∞ 连续）。"""
    kv = _gauss_kernel_1d(sigma_v)
    ku = _gauss_kernel_1d(sigma_u)
    pv, pu = len(kv) // 2, len(ku) // 2
    out = grid.astype(np.float64, copy=True)
    if pv > 0:
        pad_v = np.pad(out, ((pv, pv), (0, 0)), mode="edge")
        out = np.apply_along_axis(lambda m: np.convolve(m, kv, mode="valid"), 0, pad_v)
    if pu > 0:
        mode_u = "wrap" if periodic_u else "edge"
        pad_u = np.pad(out, ((0, 0), (pu, pu)), mode=mode_u)
        out = np.apply_along_axis(lambda m: np.convolve(m, ku, mode="valid"), 1, pad_u)
    return out


def _dilate_2d_grid(grid: np.ndarray, rv: int, ru: int, periodic_u: bool = True) -> np.ndarray:
    out = grid.copy()
    nv, nu = out.shape
    for dv in range(-rv, rv + 1):
        v_idx = np.clip(np.arange(nv) + dv, 0, nv - 1)
        row_shifted = grid[v_idx, :]
        for du in range(-ru, ru + 1):
            if periodic_u:
                u_idx = (np.arange(nu) + du) % nu
            else:
                u_idx = np.clip(np.arange(nu) + du, 0, nu - 1)
            out = np.maximum(out, row_shifted[:, u_idx])
    return out


def _hull2d(pts: np.ndarray) -> np.ndarray:
    pts_s = sorted(set(map(tuple, np.round(pts, 5))))
    if len(pts_s) < 3:
        return np.array(pts_s, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_s:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_s):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _offset_poly(poly: np.ndarray, d: float) -> np.ndarray:
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1, p2 = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        e0, e1 = p1 - p0, p2 - p1
        n0 = np.array([e0[1], -e0[0]], dtype=np.float64)
        n1 = np.array([e1[1], -e1[0]], dtype=np.float64)
        n0 /= np.linalg.norm(n0) + 1e-9
        n1 /= np.linalg.norm(n1) + 1e-9
        # 确保法线朝外（相对多边形重心）
        c = poly.mean(axis=0)
        if np.dot(n0, p1 - c) < 0:
            n0 = -n0
        if np.dot(n1, p1 - c) < 0:
            n1 = -n1
        v = n0 + n1
        v /= np.linalg.norm(v) + 1e-9
        out.append(p1 + v * (d / max(float(v @ n0), 0.42)))
    return np.array(out, dtype=np.float64)


def _chaikin_smooth_closed(poly: np.ndarray, iters: int = 4) -> np.ndarray:
    pts = poly.astype(np.float64)
    for _ in range(iters):
        nxt = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * nxt
        r = 0.25 * pts + 0.75 * nxt
        pts = np.empty((len(pts) * 2, 2), dtype=np.float64)
        pts[0::2] = q
        pts[1::2] = r
    return pts


def _raycast_star_hull(origin_2d: np.ndarray, poly_closed: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """从凸包内部锚点 origin_2d 沿角度数组 thetas 发射射线，求与闭合凸多边形的精确交点半径。"""
    p0 = poly_closed - origin_2d[None, :]
    p1 = np.roll(p0, -1, axis=0)
    seg = p1 - p0  # (M, 2)
    radii = np.zeros(len(thetas), dtype=np.float64)
    for k, th in enumerate(thetas):
        dx, dy = math.cos(float(th)), math.sin(float(th))
        # r * d = p0 + t * seg  =>  det = dx * seg_y - dy * seg_x
        det = dx * seg[:, 1] - dy * seg[:, 0]
        valid = np.abs(det) > 1e-10
        t = np.full(len(seg), -1.0, dtype=np.float64)
        r = np.full(len(seg), -1.0, dtype=np.float64)
        t[valid] = (dx * (-p0[valid, 1]) - dy * (-p0[valid, 0])) / det[valid]
        # 修正：r * dx - t * seg_x = p0_x, r * dy - t * seg_y = p0_y
        # Cramer: D = dx*(-seg_y) - dy*(-seg_x) = -det
        # r = (p0_x*(-seg_y) - p0_y*(-seg_x)) / (-det) = (p0_x*seg_y - p0_y*seg_x) / det
        # t = (dx*p0_y - dy*p0_x) / (-det) = (dy*p0_x - dx*p0_y) / det
        r[valid] = (p0[valid, 0] * seg[valid, 1] - p0[valid, 1] * seg[valid, 0]) / det[valid]
        t[valid] = (dy * p0[valid, 0] - dx * p0[valid, 1]) / det[valid]
        hit = valid & (t >= -1e-6) & (t <= 1.0 + 1e-6) & (r > 0.0)
        if np.any(hit):
            radii[k] = float(np.max(r[hit]))
        else:
            radii[k] = 0.06
    return radii


def _smoothstep(e0: float, e1: float, x: np.ndarray | float) -> np.ndarray:
    denom = float(e1 - e0)
    arr = np.asarray(x, dtype=np.float64)
    if abs(denom) < 1e-12:
        return np.where(arr >= e0, 1.0, 0.0)
    t = np.clip((arr - e0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def toecap_v_curve(theta: np.ndarray) -> np.ndarray:
    """计算靴身前部亮面橡胶包头上沿在参数域 v_t 中的平滑弧线（|theta| < 0.76 rad）。"""
    th_abs = np.abs(theta)
    w = np.clip(1.0 - th_abs / 0.76, 0.0, 1.0)
    return np.where(th_abs < 0.76, 0.055 + 0.185 * (np.sin(0.5 * math.pi * w) ** 0.75), -1.0)


# ==============================================================================
# 3. 单脚解剖坐标系、光滑鞋楦靴身、一体化硫化大底、编织鞋带与堆堆袜生成
# ==============================================================================
def _compute_leg_profile(leg_pts: np.ndarray, p_ankle: np.ndarray, e_shin: np.ndarray,
                         e_fwd_shin: np.ndarray, e_side_shin: np.ndarray,
                         h_vals_query: np.ndarray, thetas_periodic: np.ndarray) -> np.ndarray:
    """提取小腿/踝部在各高度截面上的光滑解剖半径包络 (len(h_vals_query), len(thetas_periodic))。"""
    rel = leg_pts - p_ankle[None, :]
    h_p = rel @ e_shin
    x_p = rel @ e_fwd_shin
    y_p = rel @ e_side_shin
    r_p = np.hypot(x_p, y_p)
    th_p = np.arctan2(y_p, x_p)

    n_th = len(thetas_periodic)
    out = np.zeros((len(h_vals_query), n_th), dtype=np.float64)
    dth = 2.0 * math.pi / n_th

    for ih, hq in enumerate(h_vals_query):
        band = np.abs(h_p - hq) < 0.016
        if np.count_nonzero(band) < 12:
            band = np.abs(h_p - hq) < 0.032
        if np.count_nonzero(band) < 4:
            out[ih, :] = 0.044
            continue
        rb = r_p[band]
        tb = th_p[band]
        bins = np.zeros(n_th, dtype=np.float64)
        med_r = float(np.percentile(rb, 82))
        for k in range(n_th):
            ang_diff = np.abs((tb - thetas_periodic[k] + math.pi) % (2.0 * math.pi) - math.pi)
            in_sec = ang_diff <= (1.35 * dth)
            if np.any(in_sec):
                bins[k] = float(np.percentile(rb[in_sec], 95))
            else:
                bins[k] = med_r
        out[ih, :] = _smooth_1d(bins, sigma=2.6, periodic=True)
    return _smooth_2d_grid(out, sigma_v=1.6, sigma_u=2.2, periodic_u=True)


def _compute_surface_normals_grid(V_grid: np.ndarray) -> np.ndarray:
    """计算周期圆周网格 V_grid (N_v, N_u+1, 3) 的平滑朝外法线。"""
    core = V_grid[:, :-1, :]  # (N_v, N_u, 3)
    d_th = np.roll(core, -1, axis=1) - np.roll(core, 1, axis=1)
    d_v = np.zeros_like(core)
    d_v[1:-1] = core[2:] - core[:-2]
    d_v[0] = core[1] - core[0]
    d_v[-1] = core[-1] - core[-2]
    nrm = np.cross(d_th, d_v)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-9
    out = np.empty_like(V_grid)
    out[:, :-1, :] = nrm
    out[:, -1, :] = nrm[:, 0, :]
    return out


def _grid_to_mesh_arrays(V_grid: np.ndarray, u_vals: np.ndarray, v_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 (N_v, N_u_plus_1, 3) 网格展开为顶点、三角面与非重叠 UV（无跨缝翻转面）。"""
    nv, nu1, _ = V_grid.shape
    verts = V_grid.reshape(-1, 3).astype(np.float64)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uvs = np.column_stack([uu.reshape(-1), vv.reshape(-1)]).astype(np.float64)

    faces = []
    for j in range(nv - 1):
        r0 = j * nu1
        r1 = (j + 1) * nu1
        for i in range(nu1 - 1):
            v00 = r0 + i
            v01 = r0 + i + 1
            v10 = r1 + i
            v11 = r1 + i + 1
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))
    return verts, np.asarray(faces, dtype=np.int32), uvs


def _eval_bilinear_grid(V_grid: np.ndarray, u_norm: float, v_norm: float) -> np.ndarray:
    """在靴身外表面网格 (N_v_outer, N_u+1, 3) 上按归一化局部参数 (u_norm, v_norm) 双线性插值。"""
    nv, nu1, _ = V_grid.shape
    fx = float(np.clip(u_norm, 0.0, 1.0)) * (nu1 - 1)
    fy = float(np.clip(v_norm, 0.0, 1.0)) * (nv - 1)
    ix = min(int(math.floor(fx)), nu1 - 2)
    iy = min(int(math.floor(fy)), nv - 2)
    tx = fx - ix
    ty = fy - iy
    p00 = V_grid[iy, ix]
    p10 = V_grid[iy, ix + 1]
    p01 = V_grid[iy + 1, ix]
    p11 = V_grid[iy + 1, ix + 1]
    return (1.0 - ty) * ((1.0 - tx) * p00 + tx * p10) + ty * ((1.0 - tx) * p01 + tx * p11)


def _build_ribbon_tube(curve_pts: np.ndarray, up_hints: np.ndarray,
                       half_w: float, half_t: float,
                       uv_box: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿 3D 样条构建闭合扁椭圆棉质鞋带条，并映射至专属无重叠 UV 子块（两端微收口入孔，无重叠封盖）。"""
    n_s = len(curve_pts)
    n_c = 8  # 扁椭圆截面周向采样点 (+1 闭合列以消除环向 UV 折返)
    tangents = np.zeros_like(curve_pts)
    tangents[1:-1] = curve_pts[2:] - curve_pts[:-2]
    tangents[0] = curve_pts[1] - curve_pts[0]
    tangents[-1] = curve_pts[-1] - curve_pts[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9

    ang = np.linspace(0.0, 2.0 * math.pi, n_c + 1, dtype=np.float64)
    ca, sa = np.cos(ang), np.sin(ang)

    V_rib = np.zeros((n_s, n_c + 1, 3), dtype=np.float64)
    for k in range(n_s):
        t_vec = tangents[k]
        n_vec = up_hints[k] - t_vec * np.dot(up_hints[k], t_vec)
        n_norm = np.linalg.norm(n_vec)
        if n_norm < 1e-6:
            fallback = np.array([0.0, 0.0, 1.0]) if abs(t_vec[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n_vec = np.cross(t_vec, fallback)
            n_norm = np.linalg.norm(n_vec) + 1e-9
        n_vec /= n_norm
        b_vec = np.cross(n_vec, t_vec)  # 右手系：cross(da, ds) 朝外！
        b_vec /= np.linalg.norm(b_vec) + 1e-9

        # 两端微收束使其利落插入金属鞋眼中心或收成鞋带头塑料箍
        s_frac = k / max(n_s - 1, 1)
        end_taper = 0.55 + 0.45 * math.sin(math.pi * min(max(s_frac, 0.02), 0.98)) ** 0.35
        V_rib[k] = (
            curve_pts[k][None, :]
            + (half_w * end_taper * ca)[:, None] * b_vec[None, :]
            + (half_t * end_taper * sa)[:, None] * n_vec[None, :]
        )

    u0, u1, v0, v1 = uv_box
    u_cols = np.linspace(u0, u1, n_c + 1, dtype=np.float64)
    v_rows = np.linspace(v0, v1, n_s, dtype=np.float64)
    return _grid_to_mesh_arrays(V_rib, u_cols, v_rows)


def _merge_mesh_parts(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_all, f_all, uv_all = [], [], []
    v_off = 0
    for v, f, uv in parts:
        v_all.append(v)
        uv_all.append(uv)
        f_all.append(f + v_off)
        v_off += len(v)
    return np.vstack(v_all), np.vstack(f_all), np.vstack(uv_all)


def compute_footwear_data_v3(mesh_npz: str | Path) -> dict:
    """从 mesh_lod0_v2.npz 计算左右靴身、硫化鞋底、鞋带及堆堆白袜的精确三维网格、独立非重叠 UV 与特征锚点。"""
    d = np.load(str(Path(mesh_npz).resolve()), allow_pickle=True)
    B = sam_to_bl_np(d["v"])
    rest = d["rest_v"].astype(np.float64)
    j_bl = sam_to_bl_np(d["pred_joint_coords"])
    jn = [str(x) for x in d["joint_names"]]

    ry, rx, rz = rest[:, 1], rest[:, 0], rest[:, 2]
    out = {"feet": {}, "B": B}

    for side, pfx, rx_sign in (("L", "l_", +1.0), ("R", "r_", -1.0)):
        side_v = (rx * rx_sign) > 0.0
        p_knee = j_bl[jn.index(f"{pfx}lowleg")].copy()
        p_ankle = j_bl[jn.index(f"{pfx}talocrural")].copy()
        p_ball = j_bl[jn.index(f"{pfx}ball")].copy()

        e_shin = p_knee - p_ankle
        e_shin /= np.linalg.norm(e_shin) + 1e-9

        heel_vids = np.where(side_v & (ry < 2.2) & (rz < -2.0))[0]
        fore_vids = np.where(side_v & (ry < 2.2) & (rz > 6.0) & (rz < 13.5))[0]
        sole_vids = np.where(side_v & (ry < 2.8))[0]
        foot_vids = np.where(side_v & (ry < 16.5))[0]
        leg_vids = np.where(side_v & (ry > 5.0) & (ry < 36.0))[0]

        p_heel = B[heel_vids].mean(axis=0)
        p_fore = B[fore_vids].mean(axis=0)
        e_fwd = p_fore - p_heel
        e_fwd /= np.linalg.norm(e_fwd) + 1e-9

        x_med = float(np.median(np.abs(rx[sole_vids])))
        p_lat = B[sole_vids[np.abs(rx[sole_vids]) > x_med]].mean(axis=0)
        p_med = B[sole_vids[np.abs(rx[sole_vids]) <= x_med]].mean(axis=0)
        e_lat_raw = p_lat - p_med

        e_up_foot = np.cross(e_fwd, e_lat_raw)
        e_up_foot /= np.linalg.norm(e_up_foot) + 1e-9
        if np.dot(e_up_foot, p_ankle - p_heel) < 0:
            e_up_foot = -e_up_foot
        e_side = np.cross(e_up_foot, e_fwd)
        e_side /= np.linalg.norm(e_side) + 1e-9

        med_vec = p_med - p_lat
        med_sign = 1.0 if float(np.dot(e_side, med_vec)) >= 0.0 else -1.0
        med_dir_3d = med_vec - e_shin * np.dot(med_vec, e_shin)
        med_dir_3d /= np.linalg.norm(med_dir_3d) + 1e-9

        # 与小腿正交的前向/侧向基向量（保持与 (e_fwd, e_side, e_up_foot) 完全一致的右手系定向）
        e_fwd_shin = e_fwd - e_shin * np.dot(e_fwd, e_shin)
        e_fwd_shin /= np.linalg.norm(e_fwd_shin) + 1e-9
        e_side_shin = np.cross(e_shin, e_fwd_shin)
        e_side_shin /= np.linalg.norm(e_side_shin) + 1e-9

        # 足底平面原点与光滑圆润鞋底外轮廓（Chaikin 四阶平滑消除多边形折角）
        h_vals = (B[foot_vids] - p_heel) @ e_up_foot
        h_min = float(np.percentile(h_vals, 2))
        foot_origin = p_heel + e_up_foot * h_min

        rel_foot = B[foot_vids] - foot_origin
        u_f = rel_foot @ e_fwd
        v_f = rel_foot @ e_side
        w_f = rel_foot @ e_up_foot
        low_contact = w_f < 0.052
        hull_raw = _offset_poly(_hull2d(np.column_stack([u_f[low_contact], v_f[low_contact]])), 0.0105)
        hull_smooth = _chaikin_smooth_closed(hull_raw, iters=4)

        ankle_rel = p_ankle - foot_origin
        ankle_2d = np.array([float(ankle_rel @ e_fwd), float(ankle_rel @ e_side)], dtype=np.float64)

        # ------------------------------------------------------------------
        # A. 光滑匡威高帮靴身网格 + 卷边内衬 (shoe_upper)
        # ------------------------------------------------------------------
        n_u_up = 112
        n_v_up = 64
        thetas_per = np.linspace(-math.pi, math.pi, n_u_up, endpoint=False, dtype=np.float64)
        thetas_up = np.append(thetas_per, math.pi)
        v_t_vals = np.linspace(0.0, 1.0, n_v_up, dtype=np.float64)

        r_sole_per = _smooth_1d(_raycast_star_hull(ankle_2d, hull_smooth, thetas_per), sigma=1.8, periodic=True)
        r_sole_up = np.append(r_sole_per, r_sole_per[0])

        # 提取踝至小腿各高度真实腿围包络
        h_leg_samples = np.linspace(0.012, 0.160, 32, dtype=np.float64)
        r_leg_table = _compute_leg_profile(
            B[leg_vids], p_ankle, e_shin, e_fwd_shin, e_side_shin, h_leg_samples, thetas_per
        )

        def interp_leg_r(h_q: float) -> np.ndarray:
            idx_f = np.clip((h_q - h_leg_samples[0]) / (h_leg_samples[-1] - h_leg_samples[0]) * (len(h_leg_samples) - 1),
                            0.0, len(h_leg_samples) - 1.0001)
            i0 = int(math.floor(float(idx_f)))
            t_h = float(idx_f) - i0
            row = (1.0 - t_h) * r_leg_table[i0] + t_h * r_leg_table[i0 + 1]
            return np.append(row, row[0])

        # 靴筒领口高度曲线：前中圆弧鞋舌自然高出鞋领 +13mm，跟腱处微凹 -2.5mm
        tongue_lobe = np.clip(1.0 - (thetas_up / 0.34) ** 2, 0.0, 1.0) ** 1.35
        heel_dip = np.exp(-((np.abs(thetas_up) - math.pi) / 0.45) ** 2)
        h_collar = 0.0755 + 0.0135 * tongue_lobe - 0.0025 * heel_dip

        v_split = 0.54  # 脚背斜坡与高帮竖筒的过渡分界
        h_throat = 0.020
        r_throat = interp_leg_r(h_throat) + 0.0080

        cs_th = np.cos(thetas_up)
        sn_th = np.sin(thetas_up)
        dir_foot_3d = cs_th[:, None] * e_fwd[None, :] + sn_th[:, None] * e_side[None, :]
        dir_shin_3d = cs_th[:, None] * e_fwd_shin[None, :] + sn_th[:, None] * e_side_shin[None, :]

        # 三次 Bezier 脚背/圆头鞋楦控制点：确保鞋头饱满圆润、脚背过渡自然无折痕
        p_bot_rim = (
            foot_origin[None, :]
            + ankle_2d[0] * e_fwd[None, :]
            + ankle_2d[1] * e_side[None, :]
            + (0.948 * r_sole_up)[:, None] * dir_foot_3d
            + 0.0195 * e_up_foot[None, :]
        )
        p_throat_3d = p_ankle[None, :] + h_throat * e_shin[None, :] + r_throat[:, None] * dir_shin_3d

        fwd_pos = np.clip(cs_th, 0.0, 1.0)
        b0 = p_bot_rim
        b1 = (
            p_bot_rim
            + (0.024 + 0.011 * fwd_pos)[:, None] * e_up_foot[None, :]
            - (0.004 * fwd_pos)[:, None] * dir_foot_3d
        )
        b2 = (
            p_throat_3d
            - 0.024 * e_shin[None, :]
            + (0.28 * np.maximum(r_sole_up - r_throat, 0.0) * (0.35 + 0.65 * fwd_pos))[:, None] * dir_foot_3d
            + (0.006 * fwd_pos)[:, None] * e_up_foot[None, :]
        )
        b3 = p_throat_3d

        V_up_outer = np.zeros((n_v_up, n_u_up + 1, 3), dtype=np.float64)
        for j, vt in enumerate(v_t_vals):
            if vt <= v_split:
                s = vt / v_split
                om = 1.0 - s
                V_up_outer[j] = (
                    (om ** 3) * b0
                    + 3.0 * (om ** 2) * s * b1
                    + 3.0 * om * (s ** 2) * b2
                    + (s ** 3) * b3
                )
            else:
                s = (vt - v_split) / (1.0 - v_split)
                h_cur = h_throat + s * (h_collar - h_throat)
                r_base = interp_leg_r(float(np.mean(h_cur))) + (0.0080 - 0.0004 * s)
                V_up_outer[j] = (
                    p_ankle[None, :]
                    + h_cur[:, None] * e_shin[None, :]
                    + r_base[:, None] * dir_shin_3d
                )

        # 对喉部交界做轻微纵向平滑，确保脚背至靴筒达到二阶光滑连续
        V_up_outer[:, :-1, 0] = _smooth_2d_grid(V_up_outer[:, :-1, 0], sigma_v=1.3, sigma_u=0.8, periodic_u=True)
        V_up_outer[:, :-1, 1] = _smooth_2d_grid(V_up_outer[:, :-1, 1], sigma_v=1.3, sigma_u=0.8, periodic_u=True)
        V_up_outer[:, :-1, 2] = _smooth_2d_grid(V_up_outer[:, :-1, 2], sigma_v=1.3, sigma_u=0.8, periodic_u=True)
        V_up_outer[:, -1, :] = V_up_outer[:, 0, :]

        # 光滑包络防穿模推离：确保所有裸足/踝骨顶点均被严密包裹且绝不印出分趾沟槽
        nrm_base = _compute_surface_normals_grid(V_up_outer)
        foot_chk_mask = (
            side_v
            & (((B - foot_origin) @ e_up_foot) > 0.017)
            & (((B - p_ankle) @ e_shin) < 0.072)
        )
        chk_pts = B[foot_chk_mask]
        grid_pts_flat = V_up_outer[:, :-1, :].reshape(-1, 3)
        grid_nrm_flat = nrm_base[:, :-1, :].reshape(-1, 3)
        need_push = np.zeros(len(grid_pts_flat), dtype=np.float64)
        if len(chk_pts) > 0:
            # 向量化最近网格面点投影与法向间隙检查
            d2 = np.sum((chk_pts[:, None, :] - grid_pts_flat[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(d2, axis=1)
            pen = np.sum((chk_pts - grid_pts_flat[nearest]) * grid_nrm_flat[nearest], axis=1) + 0.0050
            pos_pen = pen > 0.0
            if np.any(pos_pen):
                np.maximum.at(need_push, nearest[pos_pen], pen[pos_pen])

        push_grid = need_push.reshape(n_v_up, n_u_up)
        if np.any(push_grid > 0.0):
            dil = _dilate_2d_grid(push_grid, rv=3, ru=5, periodic_u=True)
            sm = _smooth_2d_grid(dil, sigma_v=2.8, sigma_u=5.0, periodic_u=True)
            # 保持底口收入橡胶鞋底围条内 (j=0..2 权重渐入)
            w_bot = _smoothstep(0.0, 0.08, v_t_vals)[:, None]
            push_final = np.maximum(sm, push_grid * 0.95) * w_bot
            push_ext = np.column_stack([push_final, push_final[:, 0]])
            V_up_outer += nrm_base * push_ext[:, :, None]

        # 注入真实匡威三维结构微浮雕：鞋舌凹槽 + 鞋眼襟叠层台阶 (+1.5mm) + 橡胶包头微台阶 (+0.8mm)
        nrm_up = _compute_surface_normals_grid(V_up_outer)
        vt_2d = v_t_vals[:, None]
        th_2d = thetas_up[None, :]
        th_abs_2d = np.abs(th_2d)

        placket_active = _smoothstep(0.23, 0.28, vt_2d) * (1.0 - 0.65 * _smoothstep(0.91, 0.97, vt_2d))
        tongue_recess = -0.0009 * (1.0 - _smoothstep(EYESTAY_INNER_THETA - 0.018, EYESTAY_INNER_THETA + 0.012, th_abs_2d))
        eyestay_flap = +0.0014 * (
            _smoothstep(EYESTAY_INNER_THETA - 0.012, EYESTAY_INNER_THETA + 0.015, th_abs_2d)
            * (1.0 - _smoothstep(0.39, 0.46, th_abs_2d))
        )
        u_tip_max = float(hull_smooth[:, 0].max())
        rel_up_foot = V_up_outer - foot_origin[None, None, :]
        u_up_fwd = np.sum(rel_up_foot * e_fwd[None, None, :], axis=2)
        z_up_foot = np.sum(rel_up_foot * e_up_foot[None, None, :], axis=2)
        # 橡胶包头三维有符号场：底沿精准对齐前掌防撞包边竖缝 (u_tip - 59mm)，向脚背上方圆弧内收至第一鞋眼前方
        d_toecap_m = (u_up_fwd - (u_tip_max - 0.059)) - 0.38 * np.maximum(z_up_foot - 0.027, 0.0)
        toecap_step = +0.00085 * _smoothstep(-0.002, +0.002, d_toecap_m) * _smoothstep(0.01, 0.05, vt_2d)
        heel_strip_step = +0.0006 * (1.0 - _smoothstep(0.11, 0.14, math.pi - th_abs_2d)) * _smoothstep(0.06, 0.12, vt_2d)

        relief_m = placket_active * (tongue_recess + eyestay_flap) + toecap_step + heel_strip_step
        V_up_outer += nrm_up * relief_m[:, :, None]
        V_up_outer[:, -1, :] = V_up_outer[:, 0, :]

        # 构建圆润帆布领口包边与内衬环（4 圈翻转，壁厚 3.0mm，零 SOLIDIFY 重叠 UV）
        top_row = V_up_outer[-1]
        top_tangent_up = V_up_outer[-1] - V_up_outer[-2]
        top_tangent_up /= np.linalg.norm(top_tangent_up, axis=1, keepdims=True) + 1e-9
        top_out_nrm = _compute_surface_normals_grid(V_up_outer)[-1]
        wall_t = 0.0030
        rim_rows = [
            top_row + 0.0011 * top_tangent_up - 0.35 * wall_t * top_out_nrm,
            top_row + 0.0013 * top_tangent_up - 0.80 * wall_t * top_out_nrm,
            top_row - 0.0015 * top_tangent_up - 1.00 * wall_t * top_out_nrm,
            top_row - 0.0105 * top_tangent_up - 1.02 * wall_t * top_out_nrm,
        ]
        V_up_full = np.concatenate([V_up_outer, np.stack(rim_rows, axis=0)], axis=0)

        u0_u, u1_u, v0_u, v1_u = SHOE_UV_LAYOUT[f"upper_{side}"]
        _, _, _, v1_ur = SHOE_UV_LAYOUT[f"upper_{side}_rim"]
        u_cols_up = np.linspace(u0_u, u1_u, n_u_up + 1, dtype=np.float64)
        v_rows_up = np.concatenate([
            np.linspace(v0_u, v1_u, n_v_up, dtype=np.float64),
            np.linspace(v1_u, v1_ur, len(rim_rows) + 1, dtype=np.float64)[1:],
        ])
        v_upper, f_upper, uv_upper = _grid_to_mesh_arrays(V_up_full, u_cols_up, v_rows_up)

        # ------------------------------------------------------------------
        # B. 一体化硫化橡胶厚底 + 前掌防撞包边台阶 (shoe_sole)
        # ------------------------------------------------------------------
        # 先在 hull_smooth 上定位外侧前掌防撞包边竖缝点，再按真实周长等弧长重采样 144 段均匀网格！
        n_perim = 144
        theta_seam_fox = -med_sign * (0.355 * math.pi)
        hull_rel_a = hull_smooth - ankle_2d[None, :]
        hull_th = np.arctan2(hull_rel_a[:, 1], hull_rel_a[:, 0])
        i_start = int(np.argmin(np.abs((hull_th - theta_seam_fox + math.pi) % (2.0 * math.pi) - math.pi)))
        hull_rot = np.roll(hull_smooth, -i_start, axis=0)
        # 确保环绕方向使 cross(d_s, +e_up_foot) 朝外（即在 (e_fwd, e_side) 平面内逆时针）
        signed_area2 = float(np.sum(hull_rot[:, 0] * np.roll(hull_rot, -1, axis=0)[:, 1]
                                    - np.roll(hull_rot, -1, axis=0)[:, 0] * hull_rot[:, 1]))
        if signed_area2 < 0.0:
            hull_rot = np.vstack([hull_rot[:1], hull_rot[1:][::-1]])

        hull_closed = np.vstack([hull_rot, hull_rot[:1]])
        d_hull = np.linalg.norm(np.diff(hull_closed, axis=0), axis=1)
        cum_hull = np.concatenate([[0.0], np.cumsum(d_hull)])
        total_perim_m = float(cum_hull[-1])
        s_query = np.linspace(0.0, total_perim_m, n_perim, endpoint=False)
        fox_2d_per = np.column_stack([
            np.interp(s_query, cum_hull, hull_closed[:, 0]),
            np.interp(s_query, cum_hull, hull_closed[:, 1]),
        ])
        fox_thetas_per = np.arctan2(fox_2d_per[:, 1] - ankle_2d[1], fox_2d_per[:, 0] - ankle_2d[0])

        fox_2d = np.vstack([fox_2d_per, fox_2d_per[0]])
        fox_thetas = np.append(fox_thetas_per, fox_thetas_per[0])
        s_arc = np.linspace(0.0, 1.0, n_perim + 1, dtype=np.float64)

        # 前掌防撞包边区域掩码：按真实前向坐标 u_fwd 限定在鞋头前 60mm（使下黑护条完整穿过足弓透气孔下方直至包头接缝！）
        u_fox_max = float(fox_2d[:, 0].max())
        bumper_mask = _smoothstep(u_fox_max - 0.063, u_fox_max - 0.057, fox_2d[:, 0])

        # 径向单位外法线 (2D)
        tan_2d = np.roll(fox_2d_per, -1, axis=0) - np.roll(fox_2d_per, 1, axis=0)
        out_2d_per = np.column_stack([tan_2d[:, 1], -tan_2d[:, 0]])
        out_2d_per /= np.linalg.norm(out_2d_per, axis=1, keepdims=True) + 1e-9
        if float(np.mean(np.sum(out_2d_per * (fox_2d_per - ankle_2d[None, :]), axis=1))) < 0.0:
            out_2d_per = -out_2d_per
        out_2d = np.vstack([out_2d_per, out_2d_per[0]])

        # 8 层高程截面定义：(z_m, radial_scale, bumper_extra_m)
        sole_profile = [
            (-0.0020, 0.972, 0.0000),  # 0: 鞋底接地面（严格对齐 v2 z0=-0.0020 保持场景接地倾角不变）
            ( 0.0012, 0.996, 0.0003),  # 1: 深棕生胶大底斜切边
            ( 0.0036, 1.002, 0.0016),  # 2: 防撞包边下沿起台
            ( 0.0138, 1.006, 0.0017),  # 3: 腰线下黑线高度
            ( 0.0236, 1.004, 0.0016),  # 4: 前掌防撞包边上沿
            ( 0.0252, 1.002, 0.0000),  # 5: 包边收台过渡至主围条
            ( 0.0286, 0.996, 0.0000),  # 6: 通长上黑护条顶冠
            ( 0.0300, 0.928, 0.0000),  # 7: 内收嵌入帆布鞋帮底部
        ]
        n_z_sole = len(sole_profile)
        V_fox = np.zeros((n_z_sole, n_perim + 1, 3), dtype=np.float64)
        rel_fox = fox_2d - ankle_2d[None, :]
        for jz, (z_m, sc_r, bmp_m) in enumerate(sole_profile):
            p2 = ankle_2d[None, :] + rel_fox * sc_r + (bmp_m * bumper_mask)[:, None] * out_2d
            V_fox[jz] = (
                foot_origin[None, :]
                + p2[:, 0:1] * e_fwd[None, :]
                + p2[:, 1:2] * e_side[None, :]
                + z_m * e_up_foot[None, :]
            )

        # 验证 V_fox 法线方向：若 fox_thetas 递增方向使外法线朝内则翻转环向顺序
        chk_d_s = V_fox[3, 1] - V_fox[3, 0]
        chk_d_z = V_fox[4, 0] - V_fox[3, 0]
        chk_out = out_2d[0, 0] * e_fwd + out_2d[0, 1] * e_side
        if float(np.dot(np.cross(chk_d_s, chk_d_z), chk_out)) < 0.0:
            V_fox = V_fox[:, ::-1, :]
            fox_2d = fox_2d[::-1]
            fox_thetas = fox_thetas[::-1]
            bumper_mask = bumper_mask[::-1]
            s_arc = 1.0 - s_arc[::-1]

        u0_f, u1_f, v0_f, v1_f = SHOE_UV_LAYOUT[f"foxing_{side}"]
        u_cols_fox = u0_f + s_arc * (u1_f - u0_f)
        z_arr = np.array([p[0] for p in sole_profile], dtype=np.float64)
        v_rows_fox = v0_f + (z_arr - z_arr[0]) / (z_arr[-1] - z_arr[0]) * (v1_f - v0_f)
        v_fox_wall, f_fox_wall, uv_fox_wall = _grid_to_mesh_arrays(V_fox, u_cols_fox, v_rows_fox)

        # 鞋底底面与顶面扇形封盖（各自映射至独立无重叠平面 UV 岛）
        def make_sole_cap(ring_3d: np.ndarray, ring_2d: np.ndarray, z_m: float,
                          uv_rect: tuple[float, float, float, float], flip_normal: bool):
            c3 = foot_origin + ankle_2d[0] * e_fwd + ankle_2d[1] * e_side + z_m * e_up_foot
            pts_3d = np.vstack([c3[None, :], ring_3d[:-1]])
            pts_2d = np.vstack([ankle_2d[None, :], ring_2d[:-1]])
            mn2 = pts_2d.min(axis=0)
            mx2 = pts_2d.max(axis=0)
            span = np.maximum(mx2 - mn2, 1e-6)
            ru0, ru1, rv0, rv1 = uv_rect
            # y_side 对应横向 U，x_fwd 对应纵向 V（脚尖朝上）
            uv_c = np.column_stack([
                ru0 + 0.06 * (ru1 - ru0) + 0.88 * (ru1 - ru0) * (pts_2d[:, 1] - mn2[1]) / span[1],
                rv0 + 0.04 * (rv1 - rv0) + 0.92 * (rv1 - rv0) * (pts_2d[:, 0] - mn2[0]) / span[0],
            ])
            faces_c = []
            m = len(ring_3d) - 1
            for i in range(m):
                i_a = 1 + i
                i_b = 1 + ((i + 1) % m)
                # 校验底面法线朝 -e_up_foot，顶面法线朝 +e_up_foot
                tri = (0, i_a, i_b)
                fn = np.cross(pts_3d[tri[1]] - pts_3d[tri[0]], pts_3d[tri[2]] - pts_3d[tri[0]])
                want_dir = -e_up_foot if flip_normal else e_up_foot
                if float(np.dot(fn, want_dir)) < 0.0:
                    tri = (0, i_b, i_a)
                faces_c.append(tri)
            return pts_3d, np.asarray(faces_c, dtype=np.int32), uv_c

        p2_bot = ankle_2d[None, :] + (fox_2d - ankle_2d[None, :]) * sole_profile[0][1]
        p2_top = ankle_2d[None, :] + (fox_2d - ankle_2d[None, :]) * sole_profile[-1][1]
        v_cap_bot, f_cap_bot, uv_cap_bot = make_sole_cap(
            V_fox[0], p2_bot, sole_profile[0][0], SHOE_UV_LAYOUT[f"sole_bot_{side}"], flip_normal=True
        )
        v_cap_top, f_cap_top, uv_cap_top = make_sole_cap(
            V_fox[-1], p2_top, sole_profile[-1][0], SHOE_UV_LAYOUT[f"sole_top_{side}"], flip_normal=False
        )

        # 保存真实接地点集（与 v2 sole_bot 严格一致，保障 OPT_SCENE_V3 地面三点标定零漂移）
        sole_bottom_ring_pts = V_fox[0, :-1, :].copy()

        # ------------------------------------------------------------------
        # C. 交叉扁平编织棉鞋带 + 顶端蝴蝶结与鞋带头 (shoelaces)
        #    精确穿入前 6 对鞋眼孔中心，保留第 7 对顶孔敞开（100% 还原 2x 参考图细节！）
        # ------------------------------------------------------------------
        nrm_outer = _compute_surface_normals_grid(V_up_outer)

        def eval_upper_pt_nrm(th_val: float, vt_val: float) -> tuple[np.ndarray, np.ndarray]:
            u_n = (th_val + math.pi) / (2.0 * math.pi)
            p = _eval_bilinear_grid(V_up_outer, u_n, vt_val)
            n = _eval_bilinear_grid(nrm_outer, u_n, vt_val)
            n = n / (np.linalg.norm(n) + 1e-9)
            return p, n

        lace_specs = []  # list of (pts_3d, up_hints, half_w, half_t)

        # 1) 第 0 对鞋眼底横梁扁带
        n_seg = 15
        ts_bar = np.linspace(0.0, 1.0, n_seg)
        bar_pts, bar_nrms = [], []
        for s_b in ts_bar:
            th_b = -EYELET_THETA_ABS + 2.0 * EYELET_THETA_ABS * s_b
            p_s, n_s = eval_upper_pt_nrm(th_b, float(EYELET_V_LIST[0]))
            dip = -0.0016 + 0.0038 * math.sin(math.pi * s_b) ** 0.75
            bar_pts.append(p_s + n_s * dip)
            bar_nrms.append(n_s)
        lace_specs.append((np.asarray(bar_pts), np.asarray(bar_nrms), 0.0027, 0.00080))

        # 2) 5 组 (10 根) X 型上下交叠斜向扁棉鞋带 (k = 0..4 连接第 k 对与第 k+1 对鞋眼)
        for k_l in range(5):
            v_a = float(EYELET_V_LIST[k_l])
            v_b = float(EYELET_V_LIST[k_l + 1])
            for dir_idx, (th_a, th_b) in enumerate(((-EYELET_THETA_ABS, +EYELET_THETA_ABS),
                                                     (+EYELET_THETA_ABS, -EYELET_THETA_ABS))):
                pts_d, nrms_d = [], []
                # dir_idx==1 在中点拱高多 +1.3mm，实现真实上下编织交叠零穿插！
                mid_lift = 0.0022 if dir_idx == 0 else 0.0036
                for s_d in np.linspace(0.0, 1.0, n_seg):
                    th_cur = (1.0 - s_d) * th_a + s_d * th_b
                    vt_cur = (1.0 - s_d) * v_a + s_d * v_b
                    p_s, n_s = eval_upper_pt_nrm(th_cur, vt_cur)
                    arch = -0.0018 * (1.0 - math.sin(math.pi * s_d) ** 0.45) + mid_lift * (math.sin(math.pi * s_d) ** 0.85)
                    pts_d.append(p_s + n_s * arch)
                    nrms_d.append(n_s)
                lace_specs.append((np.asarray(pts_d), np.asarray(nrms_d), 0.00265, 0.00078))

        # 3) 鞋舌顶部立体蝴蝶结（中心绳结 + 左右双弧兔耳环 + 两条自然垂落鞋带尾）
        p_knot_s, n_knot = eval_upper_pt_nrm(0.0, float(EYELET_V_LIST[5]) + 0.012)
        p_knot = p_knot_s + n_knot * 0.0042
        # 中心横结箍
        knot_pts = np.array([
            p_knot - 0.0048 * e_side_shin - 0.0010 * n_knot,
            p_knot + 0.0012 * n_knot,
            p_knot + 0.0048 * e_side_shin - 0.0010 * n_knot,
        ])
        knot_t = np.linspace(0.0, 1.0, 9)
        kp_interp = np.array([
            (1 - t) ** 2 * knot_pts[0] + 2 * (1 - t) * t * knot_pts[1] + t ** 2 * knot_pts[2]
            for t in knot_t
        ])
        lace_specs.append((kp_interp, np.tile(n_knot, (len(kp_interp), 1)), 0.0028, 0.0015))

        # 左右蝴蝶结兔耳环（微向前上方与两侧舒展，契合 2x 照片鞋舌顶蝴蝶结剪影）
        for sgn_loop in (-1.0, +1.0):
            t_lp = np.linspace(0.0, 1.0, 22)
            ang_lp = t_lp * 2.0 * math.pi
            reach_side = sgn_loop * 0.021 * (np.sin(0.5 * ang_lp) ** 1.15)
            reach_up = 0.014 * np.sin(math.pi * t_lp) + 0.0045 * np.sin(2.0 * math.pi * t_lp)
            reach_fwd = 0.012 * (np.sin(math.pi * t_lp) ** 0.8)
            lp_pts = (
                p_knot[None, :]
                + reach_side[:, None] * e_side_shin[None, :]
                + reach_up[:, None] * e_shin[None, :]
                + reach_fwd[:, None] * n_knot[None, :]
            )
            up_h = np.tile(n_knot + 0.35 * e_shin, (len(t_lp), 1))
            up_h /= np.linalg.norm(up_h, axis=1, keepdims=True) + 1e-9
            lace_specs.append((lp_pts, up_h, 0.0024, 0.00072))

        # 两条垂落鞋带尾（顺脚背微向下外侧垂悬）
        for sgn_tail in (-1.0, +1.0):
            t_tl = np.linspace(0.0, 1.0, 14)
            tl_pts = (
                p_knot[None, :]
                + (sgn_tail * (0.003 + 0.017 * (t_tl ** 1.15)))[:, None] * e_side_shin[None, :]
                - (0.026 * (t_tl ** 1.25))[:, None] * e_shin[None, :]
                + (0.008 * np.sin(0.65 * math.pi * t_tl) + 0.003 * (1.0 - t_tl))[:, None] * n_knot[None, :]
            )
            up_h = np.tile(n_knot, (len(t_tl), 1))
            lace_specs.append((tl_pts, up_h, 0.0021, 0.00075))

        # 为全部鞋带部件分配严格不相交的独立 UV 子条带
        lu0, lu1, lv0, lv1 = SHOE_UV_LAYOUT[f"laces_{side}"]
        n_pieces = len(lace_specs)
        du_piece = (lu1 - lu0) / n_pieces
        lace_parts = []
        for idx_p, (c_pts, u_h, hw, ht) in enumerate(lace_specs):
            pu0 = lu0 + (idx_p + 0.08) * du_piece
            pu1 = lu0 + (idx_p + 0.92) * du_piece
            lace_parts.append(_build_ribbon_tube(c_pts, u_h, hw, ht, (pu0, pu1, lv0, lv1)))

        v_shoe, f_shoe, uv_shoe = _merge_mesh_parts([
            (v_upper, f_upper, uv_upper),
            (v_fox_wall, f_fox_wall, uv_fox_wall),
            (v_cap_bot, f_cap_bot, uv_cap_bot),
            (v_cap_top, f_cap_top, uv_cap_top),
            *lace_parts,
        ])

        # ------------------------------------------------------------------
        # D. 堆堆螺纹白袜独立网格 + 双层罗纹卷边口 (sock_{side})
        # ------------------------------------------------------------------
        n_u_sk = 88
        n_v_sk = 64
        sk_thetas_per = np.linspace(-math.pi, math.pi, n_u_sk, endpoint=False, dtype=np.float64)
        sk_thetas = np.append(sk_thetas_per, math.pi)
        sk_s = np.linspace(0.0, 1.0, n_v_sk, dtype=np.float64)
        h_sock_bot = 0.046
        h_sock_top = 0.153
        h_sock_vals = h_sock_bot + sk_s * (h_sock_top - h_sock_bot)

        r_leg_sock = _compute_leg_profile(
            B[leg_vids], p_ankle, e_shin, e_fwd_shin, e_side_shin, h_sock_vals, sk_thetas_per
        )
        r_leg_sock = np.column_stack([r_leg_sock, r_leg_sock[:, 0]])

        sk_dir_3d = (
            np.cos(sk_thetas)[None, :, None] * e_fwd_shin[None, None, :]
            + np.sin(sk_thetas)[None, :, None] * e_side_shin[None, None, :]
        )

        # 下段 (s < 0.34) 顺滑贴合于靴领内侧 (r_leg + 4.3mm < 靴筒 r_leg + 7.8mm，零穿插！)
        # 上段 (s ∈ [0.36, 0.96]) 形成 4 道饱满有机的堆堆袜横向松弛褶皱环 + 顶部木耳罗纹口
        s_2d = sk_s[:, None]
        th_sk_2d = sk_thetas[None, :]
        slouch_env = _smoothstep(0.32, 0.45, s_2d) * (1.0 - 0.30 * _smoothstep(0.92, 1.00, s_2d))
        # 微相位扰动令横向堆叠褶皱呈现真实棉袜手工松弛自然感
        phase_mod = 0.28 * np.sin(2.0 * th_sk_2d + 0.7 * rx_sign) + 0.16 * np.cos(3.0 * th_sk_2d - 0.5)
        wave_arg = 2.0 * math.pi * 4.15 * ((s_2d - 0.36) / 0.60) + phase_mod
        # 峰圆润饱满、谷柔和内收的棉织褶皱波形
        scrunch_wave = (0.5 + 0.5 * np.sin(wave_arg)) ** 0.72
        cuff_ruffle = 0.00055 * np.cos(20.0 * th_sk_2d) * _smoothstep(0.88, 0.98, s_2d)

        r_sock_outer = (
            r_leg_sock
            + 0.0040
            + slouch_env * (0.0014 + 0.00165 * scrunch_wave)
            + cuff_ruffle
        )
        r_sock_outer[:, -1] = r_sock_outer[:, 0]

        V_sock_outer = (
            p_ankle[None, None, :]
            + h_sock_vals[:, None, None] * e_shin[None, None, :]
            + r_sock_outer[:, :, None] * sk_dir_3d
        )

        # 双层罗纹袜口内卷封边（5 圈平滑内收贴合小腿皮肤，消除薄片断口且 0% UV 重叠）
        r_top_leg = r_leg_sock[-1]
        r_top_out = r_sock_outer[-1]
        h_top = h_sock_vals[-1]
        sock_rim_specs = [
            (h_top + 0.0012, 0.72 * r_top_out + 0.28 * (r_top_leg + 0.0014)),
            (h_top + 0.0014, 0.38 * r_top_out + 0.62 * (r_top_leg + 0.0014)),
            (h_top - 0.0008, r_top_leg + 0.0015),
            (h_top - 0.0065, r_top_leg + 0.0012),
        ]
        sock_rim_rows = [
            p_ankle[None, :] + h_r * e_shin[None, :] + r_r[:, None] * sk_dir_3d[0]
            for h_r, r_r in sock_rim_specs
        ]
        V_sock_full = np.concatenate([V_sock_outer, np.stack(sock_rim_rows, axis=0)], axis=0)

        su0, su1, sv0, sv1 = SOCK_UV_LAYOUT[f"sock_{side}"]
        _, _, _, sv1_r = SOCK_UV_LAYOUT[f"sock_{side}_rim"]
        u_cols_sk = np.linspace(su0, su1, n_u_sk + 1, dtype=np.float64)
        v_rows_sk = np.concatenate([
            np.linspace(sv0, sv1, n_v_sk, dtype=np.float64),
            np.linspace(sv1, sv1_r, len(sock_rim_rows) + 1, dtype=np.float64)[1:],
        ])
        v_sock, f_sock, uv_sock = _grid_to_mesh_arrays(V_sock_full, u_cols_sk, v_rows_sk)

        # ------------------------------------------------------------------
        # E. 为 PBR 贴图烘焙器记录靴身与围条精确三维地标（100% 三维等距无变形绘制！）
        # ------------------------------------------------------------------
        # 1) 内踝经典圆标三维中心与外部观察者局部正交切平面基向量 (T_right_view, T_up)
        theta_patch = med_sign * (0.585 * math.pi)  # 内踝骨正中偏后（对准鞋眼 #4-#6 后方，严格匹配 2x 照片）
        vt_patch = 0.745
        p_patch_3d, n_patch_3d = eval_upper_pt_nrm(theta_patch, vt_patch)
        t_up_patch = e_shin - n_patch_3d * np.dot(e_shin, n_patch_3d)
        t_up_patch /= np.linalg.norm(t_up_patch) + 1e-9
        # 令 T_fwd (外部正视右向) = cross(T_up, N_out)，确保左右双靴圆标弧形文字从靴外正视均严格从左向右不镜像！
        t_fwd_patch = np.cross(t_up_patch, n_patch_3d)
        t_fwd_patch /= np.linalg.norm(t_fwd_patch) + 1e-9

        # 2) 14 个鞋眼金属孔 + 2 个内侧足弓透气金属孔的精确三维坐标与局部切空间
        eyelets_3d = []
        for k_e, vt_e in enumerate(EYELET_V_LIST):
            for th_e in (-EYELET_THETA_ABS, +EYELET_THETA_ABS):
                pe, ne = eval_upper_pt_nrm(th_e, float(vt_e))
                eyelets_3d.append({
                    "pos": pe,
                    "nrm": ne,
                    "theta": th_e,
                    "vt": float(vt_e),
                    "unlaced": (k_e == 6),
                    "kind": "eyestay",
                })
        for th_v_abs, vt_v in ((0.50, 0.136), (0.86, 0.130)):
            th_v = med_sign * th_v_abs
            pv, nv = eval_upper_pt_nrm(th_v, vt_v)
            eyelets_3d.append({
                "pos": pv,
                "nrm": nv,
                "theta": th_v,
                "vt": vt_v,
                "unlaced": True,
                "kind": "vent",
            })

        # 3) 后跟中央在围条弧长 s_arc 上的精确坐标与外部正视右向符号（零接缝正立放置 "CONVERSE ALL ★ STAR"）
        heel_dir_2d = np.array([-1.0, 0.0], dtype=np.float64)
        fox_unit_2d = (fox_2d[:-1] - ankle_2d[None, :])
        fox_unit_2d /= np.linalg.norm(fox_unit_2d, axis=1, keepdims=True) + 1e-9
        idx_heel = int(np.argmax(fox_unit_2d @ heel_dir_2d))
        s_heel_center = float(s_arc[idx_heel])
        # 从后跟外侧正视鞋跟时：视线沿 +e_fwd，上方向为 +e_up_foot，因此观察者右方向为 cross(+e_fwd, +e_up_foot)
        viewer_right_heel = np.cross(e_fwd, e_up_foot)
        step_plus_s = V_fox[3, (idx_heel + 1) % n_perim] - V_fox[3, (idx_heel - 1) % n_perim]
        heel_lr_sign = 1.0 if float(np.dot(step_plus_s, viewer_right_heel)) >= 0.0 else -1.0

        shoe_focus_center = 0.38 * p_ankle + 0.44 * (foot_origin + 0.088 * e_fwd + 0.028 * e_up_foot) + 0.18 * (p_ankle + 0.085 * e_shin)
        heel_focus_center = foot_origin - 0.025 * e_fwd + 0.045 * e_up_foot

        out["feet"][side] = {
            "shoe_v": v_shoe,
            "shoe_f": f_shoe,
            "shoe_uv": uv_shoe,
            "sock_v": v_sock,
            "sock_f": f_sock,
            "sock_uv": uv_sock,
            "V_up_outer": V_up_outer,
            "u_tip_max": u_tip_max,
            "thetas_up": thetas_up,
            "v_t_vals": v_t_vals,
            "V_fox": V_fox,
            "s_arc": s_arc,
            "total_perim_m": total_perim_m,
            "bumper_mask": bumper_mask,
            "sole_z_arr": z_arr,
            "s_heel_center": s_heel_center,
            "heel_lr_sign": heel_lr_sign,
            "med_sign": med_sign,
            "patch_pos": p_patch_3d,
            "patch_nrm": n_patch_3d,
            "patch_t_fwd": t_fwd_patch,
            "patch_t_up": t_up_patch,
            "eyelets": eyelets_3d,
            "sole_bottom_pts": sole_bottom_ring_pts,
            "p_ankle": p_ankle,
            "foot_origin": foot_origin,
            "e_fwd": e_fwd,
            "e_side": e_side,
            "e_up_foot": e_up_foot,
            "e_shin": e_shin,
            "med_dir_3d": med_dir_3d,
            "focus_center": shoe_focus_center,
            "heel_focus": heel_focus_center,
        }

    return out


# ==============================================================================
# 4. UV 非重叠栅格化严格自检（确保左右靴、鞋底、鞋带及左右袜 0.00% UV 重叠）
# ==============================================================================
def verify_non_overlapping_uvs(faces: np.ndarray, uvs: np.ndarray, res: int = 1024, label: str = "atlas") -> tuple[float, int]:
    """采用精确浮点像素中心重心坐标测试统计所有三角面内部 UV 覆盖重叠率（避免整数边界线重复计数）。"""
    acc = np.zeros((res, res), dtype=np.uint16)
    tri_uv = uvs[faces].astype(np.float64)  # (F, 3, 2)
    cent = tri_uv.mean(axis=1, keepdims=True)
    tri_in = cent + (tri_uv - cent) * (1.0 - 1e-4)

    # 转换至连续像素坐标系 (x = u * res, y = (1 - v) * res)，测试像素中心 (ix + 0.5, iy + 0.5)
    px = np.empty_like(tri_in)
    px[..., 0] = np.clip(tri_in[..., 0] * res, 0.0, float(res))
    px[..., 1] = np.clip((1.0 - tri_in[..., 1]) * res, 0.0, float(res))

    for k in range(len(px)):
        x0, y0 = px[k, 0]
        x1, y1 = px[k, 1]
        x2, y2 = px[k, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        ix0 = max(0, int(math.floor(min(x0, x1, x2))))
        ix1 = min(res, int(math.ceil(max(x0, x1, x2))))
        iy0 = max(0, int(math.floor(min(y0, y1, y2))))
        iy1 = min(res, int(math.ceil(max(y0, y1, y2))))
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
        xc = xx + 0.5
        yc = yy + 0.5
        w0 = ((y1 - y2) * (xc - x2) + (x2 - x1) * (yc - y2)) / den
        w1 = ((y2 - y0) * (xc - x2) + (x0 - x2) * (yc - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 > 0.0) & (w1 > 0.0) & (w2 > 0.0)
        acc[iy0:iy1, ix0:ix1] += inside.astype(np.uint16)

    covered = int(np.count_nonzero(acc >= 1))
    overlapped = int(np.count_nonzero(acc >= 2))
    ratio = float(overlapped) / max(covered, 1)
    print(f"[uv_check:{label}] covered_px={covered} overlapped_px={overlapped} ({ratio * 100:.4f}%)")
    return ratio, overlapped


# ==============================================================================
# 5. Blender 四通道 PBR 材质构建与流形鞋袜对象装配接口（延迟导入 bpy）
# ==============================================================================
def _load_image_colorspace(path: Path, non_color: bool):
    import bpy

    img = bpy.data.images.load(str(path.resolve()), check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def _make_pbr_material(mat_name: str, albedo_p: Path, rough_p: Path,
                       normal_p: Path, metallic_p: Path | None = None,
                       sheen_weight: float = 0.18, sss_weight: float = 0.0,
                       normal_strength: float = 1.0):
    import bpy

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    # Base Color (sRGB)
    n_alb = nt.nodes.new("ShaderNodeTexImage")
    n_alb.image = _load_image_colorspace(albedo_p, non_color=False)
    n_alb.interpolation = "Smart"
    nt.links.new(n_alb.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness (Non-Color)
    n_rgh = nt.nodes.new("ShaderNodeTexImage")
    n_rgh.image = _load_image_colorspace(rough_p, non_color=True)
    n_rgh.interpolation = "Smart"
    nt.links.new(n_rgh.outputs["Color"], bsdf.inputs["Roughness"])

    # Metallic (Non-Color, optional)
    if metallic_p is not None and metallic_p.exists():
        n_met = nt.nodes.new("ShaderNodeTexImage")
        n_met.image = _load_image_colorspace(metallic_p, non_color=True)
        n_met.interpolation = "Smart"
        nt.links.new(n_met.outputs["Color"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Metallic"].default_value = 0.0

    # Tangent-Space Normal Map (Non-Color OpenGL)
    n_nrm_tex = nt.nodes.new("ShaderNodeTexImage")
    n_nrm_tex.image = _load_image_colorspace(normal_p, non_color=True)
    n_nrm_tex.interpolation = "Smart"
    n_map = nt.nodes.new("ShaderNodeNormalMap")
    n_map.space = "TANGENT"
    n_map.inputs["Strength"].default_value = normal_strength
    nt.links.new(n_nrm_tex.outputs["Color"], n_map.inputs["Color"])
    nt.links.new(n_map.outputs["Normal"], bsdf.inputs["Normal"])

    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.36
    if "Sheen Weight" in bsdf.inputs and sheen_weight > 0.0:
        bsdf.inputs["Sheen Weight"].default_value = sheen_weight
        bsdf.inputs["Sheen Roughness"].default_value = 0.48
    if "Subsurface Weight" in bsdf.inputs and sss_weight > 0.0:
        bsdf.inputs["Subsurface Weight"].default_value = sss_weight
        bsdf.inputs["Subsurface Scale"].default_value = 0.004
    return mat


def _create_uv_mesh_obj(name: str, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, mat):
    import bpy

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()

    uvl = me.uv_layers.new(name="UVMap")
    loop_uv = uvs[faces.reshape(-1)].reshape(-1).astype(np.float32)
    uvl.data.foreach_set("uv", loop_uv)

    for poly in me.polygons:
        poly.use_smooth = True
    me.update()

    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat)
    return ob


def build_shoes_v3(mesh_npz: str | Path, tex_dir: str | Path) -> dict:
    """在 Blender 场景中构建全套 v3 独立 PBR 高帮帆布鞋与堆堆白袜（仅 4 个流形网格对象，零几何贴花图元）。"""
    tex_p = Path(tex_dir).resolve()
    data = compute_footwear_data_v3(mesh_npz)

    mat_shoe = _make_pbr_material(
        "converse_boot_pbr_v3",
        albedo_p=tex_p / "shoe_albedo_v3.png",
        rough_p=tex_p / "shoe_roughness_v3.png",
        normal_p=tex_p / "shoe_normal_v3.png",
        metallic_p=tex_p / "shoe_metallic_v3.png",
        sheen_weight=0.20,
        normal_strength=1.0,
    )
    mat_sock = _make_pbr_material(
        "ribbed_sock_pbr_v3",
        albedo_p=tex_p / "sock_albedo_v3.png",
        rough_p=tex_p / "sock_roughness_v3.png",
        normal_p=tex_p / "sock_normal_v3.png",
        metallic_p=None,
        sheen_weight=0.45,
        sss_weight=0.04,
        normal_strength=0.95,
    )

    objs = []
    for side in ("L", "R"):
        fd = data["feet"][side]
        ob_shoe = _create_uv_mesh_obj(f"shoe_{side}", fd["shoe_v"], fd["shoe_f"], fd["shoe_uv"], mat_shoe)
        ob_sock = _create_uv_mesh_obj(f"sock_{side}", fd["sock_v"], fd["sock_f"], fd["sock_uv"], mat_sock)
        objs.extend([ob_shoe, ob_sock])

    total_v = sum(len(o.data.vertices) for o in objs)
    total_f = sum(len(o.data.polygons) for o in objs)
    print(f"[shoes_v3] 装配完成：对象数={len(objs)} (v2=24，消除全部贴花圆柱体与碎片部件)，总顶点={total_v}，总三角面={total_f}")
    return {
        "objs": objs,
        "feet": data["feet"],
        "sole_bottom_L": data["feet"]["L"]["sole_bottom_pts"],
        "sole_bottom_R": data["feet"]["R"]["sole_bottom_pts"],
    }
