"""生成 v3 高帮帆布鞋（4K 四通道 PBR）与堆堆螺纹白袜（2K 三通道 PBR）独立非重叠贴图套件。

输出文件（位于 work/q004-base/v3/tex/）：
  - shoe_albedo_v3.png     (4096x4096, sRGB)
  - shoe_roughness_v3.png  (4096x4096, Non-Color)
  - shoe_metallic_v3.png   (4096x4096, Non-Color)
  - shoe_normal_v3.png     (4096x4096, Non-Color OpenGL Tangent Normal)
  - sock_albedo_v3.png     (2048x2048, sRGB)
  - sock_roughness_v3.png  (2048x2048, Non-Color)
  - sock_normal_v3.png     (2048x2048, Non-Color OpenGL Tangent Normal)
  - uv_layout_footwear_v3.png (独立非重叠 UV 展开与贴图诊断图)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shoe_lib_v3 as SL3


def _smoothstep(e0: float, e1: float, x: np.ndarray | float) -> np.ndarray:
    denom = float(e1 - e0)
    arr = np.asarray(x, dtype=np.float64)
    if abs(denom) < 1e-12:
        return np.where(arr >= e0, 1.0, 0.0)
    t = np.clip((arr - e0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _star5_sdf_mm(x_mm: np.ndarray, y_mm: np.ndarray, r_out: float, r_in: float) -> np.ndarray:
    """计算标准正立五角星（顶点朝 +y_mm）的有符号距离（内部 < 0，外部 > 0，单位 mm）。"""
    # 利用 5 重旋转镜像对称折叠到主扇区 angle ∈ [0, pi/5]
    r = np.hypot(x_mm, y_mm)
    # 旋转使 +y_mm 对应角 0
    ang = np.arctan2(x_mm, y_mm)  # [-pi, pi], 0 at +y
    sec = math.pi / 5.0
    ang_mod = np.abs(((ang + sec) % (2.0 * sec)) - sec)  # [0, pi/5]
    px = r * np.sin(ang_mod)
    py = r * np.cos(ang_mod)

    # 扇区边界线段：从外顶点 A=(0, r_out) 到内凹顶点 B=(r_in*sin(pi/5), r_in*cos(pi/5))
    ax, ay = 0.0, r_out
    bx, by = r_in * math.sin(sec), r_in * math.cos(sec)
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    t = np.clip((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-12), 0.0, 1.0)
    dx = wx - t * vx
    dy = wy - t * vy
    dist = np.hypot(dx, dy)
    # 叉积判断内外：向量 A->B 右侧（靠近原点侧）为内部
    # A=(0, 10.8), B=(2.44, 3.36) => v=(+2.44, -7.44). 原点 O: w=(0, -10.8)
    # v_x * w_y - v_y * w_x = 2.44*(-10.8) - (-7.44)*0 = -26.35 < 0 => cross < 0 为内部！
    cross = vx * wy - vy * wx
    sign = np.where(cross < 0.0, -1.0, 1.0)
    return dist * sign


def _stitch_dash_field(dist_mm: np.ndarray, along_mm: np.ndarray,
                       pitch_mm: float = 2.55, half_w_mm: float = 0.36) -> tuple[np.ndarray, np.ndarray]:
    """返回车缝明线的 (thread_mask [0..1], height_delta_mm)：含饱满线迹凸起与针眼微凹。"""
    phase = (along_mm % pitch_mm) / pitch_mm  # [0, 1)
    # 每针占周期 74%，两端为针孔凹陷
    along_env = _smoothstep(0.06, 0.20, phase) * (1.0 - _smoothstep(0.80, 0.94, phase))
    cross_env = 1.0 - _smoothstep(half_w_mm * 0.45, half_w_mm * 1.15, np.abs(dist_mm))
    thread_mask = along_env * cross_env

    # 针孔微凹 + 压线沟槽微凹
    hole_d = np.minimum(phase, 1.0 - phase) * pitch_mm
    needle_hole = np.exp(-((hole_d / 0.34) ** 2 + (dist_mm / 0.38) ** 2))
    seam_groove = np.exp(-((dist_mm / 0.72) ** 2))
    h_mm = 0.21 * thread_mask - 0.17 * needle_hole - 0.05 * seam_groove
    return thread_mask, h_mm


def _render_heel_license_plate_stamp(w_px: int = 520, h_px: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """生成高清晰度匡威黑底白字 'CONVERSE / ALL ★ STAR' 橡胶后标图案 (white_ink_mask [0..1], bevel_mask [0..1])。"""
    ink = np.zeros((h_px, w_px), dtype=np.uint8)
    pad = 14
    # 外圈白色圆角线框
    cv2.rectangle(ink, (pad, pad), (w_px - 1 - pad, h_px - 1 - pad), 255, thickness=5, lineType=cv2.LINE_AA)

    # 顶部小字 "CONVERSE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    t1 = "CONVERSE"
    (tw1, th1), _ = cv2.getTextSize(t1, font, 0.92, 2)
    cv2.putText(ink, t1, ((w_px - tw1) // 2, 68), font, 0.92, 255, 2, cv2.LINE_AA)

    # 下部粗体主字 "ALL"  [五角星]  "STAR"
    font_b = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(ink, "ALL", (56, 152), font_b, 1.85, 255, 4, cv2.LINE_AA)
    cv2.putText(ink, "STAR", (296, 152), font_b, 1.85, 255, 4, cv2.LINE_AA)

    # 正中央锐利实心五角星 (cx=238, cy=125)
    yy, xx = np.mgrid[0:h_px, 0:w_px].astype(np.float64)
    sdf_st = _star5_sdf_mm((xx - 238.0) * 0.42, -(yy - 126.0) * 0.42, r_out=12.0, r_in=4.6)
    star_m = np.clip(0.5 - sdf_st / 0.55, 0.0, 1.0)

    ink_f = np.maximum(ink.astype(np.float32) / 255.0, star_m.astype(np.float32))
    ink_f = cv2.GaussianBlur(ink_f, (3, 3), 0.55)
    return ink_f, star_m.astype(np.float32)


def _render_patch_arc_text_mask(size_px: int = 512) -> np.ndarray:
    """在圆标内圈上下弧生成细腻压印弧形微字印记（增强微距特写考究质感）。"""
    strip_w, strip_h = 800, 90
    top_s = np.zeros((strip_h, strip_w), dtype=np.uint8)
    bot_s = np.zeros((strip_h, strip_w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(top_s, "CONVERSE", (235, 62), font, 1.45, 255, 3, cv2.LINE_AA)
    cv2.putText(bot_s, "ALL  STAR", (235, 62), font, 1.45, 255, 3, cv2.LINE_AA)

    yy, xx = np.mgrid[0:size_px, 0:size_px].astype(np.float32)
    c = 0.5 * (size_px - 1)
    dx = (xx - c) / c  # [-1, 1] 对应 [-22mm, +22mm]
    dy = -(yy - c) / c  # +up
    r = np.hypot(dx, dy)
    ang = np.arctan2(dx, dy)  # 0 at top (+dy)

    out = np.zeros((size_px, size_px), dtype=np.float32)
    # 上弧 r ∈ [0.56, 0.75], ang ∈ [-1.15, 1.15]
    m_top = (r >= 0.56) & (r <= 0.75) & (np.abs(ang) <= 1.15)
    if np.any(m_top):
        u_t = ((ang[m_top] + 1.15) / 2.30 * (strip_w - 1)).astype(np.int32)
        v_t = ((0.75 - r[m_top]) / 0.19 * (strip_h - 1)).astype(np.int32)
        out[m_top] = top_s[np.clip(v_t, 0, strip_h - 1), np.clip(u_t, 0, strip_w - 1)] / 255.0

    # 下弧 ang_bot 围绕 pi
    ang_b = np.arctan2(dx, -dy)  # 0 at bottom (-dy)
    m_bot = (r >= 0.56) & (r <= 0.75) & (np.abs(ang_b) <= 1.15)
    if np.any(m_bot):
        u_b = ((ang_b[m_bot] + 1.15) / 2.30 * (strip_w - 1)).astype(np.int32)
        v_b = ((r[m_bot] - 0.56) / 0.19 * (strip_h - 1)).astype(np.int32)
        out[m_bot] = np.maximum(out[m_bot], bot_s[np.clip(v_b, 0, strip_h - 1), np.clip(u_b, 0, strip_w - 1)] / 255.0)

    return cv2.GaussianBlur(out, (3, 3), 0.6)


def _height_mm_to_opengl_normal_bgr(height_mm: np.ndarray, mm_per_px_x: np.ndarray, mm_per_px_y: np.ndarray) -> np.ndarray:
    """将毫米高度场转换为 OpenGL 切线空间法线图（返回供 cv2.imwrite 直接保存的 uint8 BGR 数组）。"""
    # x_col 向右对应 +U；y_row 向下对应 -V！
    dh_dx = cv2.Sobel(height_mm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dh_dy = cv2.Sobel(height_mm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0

    nx = -dh_dx / np.maximum(mm_per_px_x, 0.04)
    ny = +dh_dy / np.maximum(mm_per_px_y, 0.04)
    nz = np.ones_like(nx, dtype=np.float32)

    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-12)
    nx *= inv_len
    ny *= inv_len
    nz *= inv_len

    bgr = np.empty((*height_mm.shape, 3), dtype=np.uint8)
    bgr[..., 2] = np.clip(np.round((0.5 + 0.5 * nx) * 255.0), 0, 255).astype(np.uint8)  # R (+U)
    bgr[..., 1] = np.clip(np.round((0.5 + 0.5 * ny) * 255.0), 0, 255).astype(np.uint8)  # G (+V)
    bgr[..., 0] = np.clip(np.round((0.5 + 0.5 * nz) * 255.0), 0, 255).astype(np.uint8)  # B (+N)
    return bgr


def _interp_grid_2d(V_grid: np.ndarray, u_norm: np.ndarray, v_norm: np.ndarray) -> np.ndarray:
    """在 (N_v, N_u+1, C) 网格上对二维数组 u_norm, v_norm 做向量化双线性插值。"""
    nv, nu1, ch = V_grid.shape
    fx = np.clip(u_norm, 0.0, 1.0) * (nu1 - 1)
    fy = np.clip(v_norm, 0.0, 1.0) * (nv - 1)
    ix = np.minimum(np.floor(fx).astype(np.int32), nu1 - 2)
    iy = np.minimum(np.floor(fy).astype(np.int32), nv - 2)
    tx = (fx - ix)[..., None]
    ty = (fy - iy)[..., None]
    p00 = V_grid[iy, ix]
    p10 = V_grid[iy, ix + 1]
    p01 = V_grid[iy + 1, ix]
    p11 = V_grid[iy + 1, ix + 1]
    return (1.0 - ty) * ((1.0 - tx) * p00 + tx * p10) + ty * ((1.0 - tx) * p01 + tx * p11)


def build_shoe_pbr_maps(data: dict, out_dir: Path, size: int = 4096):
    """合成 4K 四通道匡威高帮帆布鞋 PBR 贴图套件。"""
    # 默认底色采用柔和米白帆布色以避免空白区缩略图突兀
    albedo = np.full((size, size, 3), (228, 220, 211), dtype=np.float32)  # RGB [0..255]
    rough = np.full((size, size), 0.85, dtype=np.float32)
    metal = np.zeros((size, size), dtype=np.float32)
    height_mm = np.zeros((size, size), dtype=np.float32)
    mm_px_x = np.full((size, size), 0.22, dtype=np.float32)
    mm_px_y = np.full((size, size), 0.22, dtype=np.float32)

    rng = np.random.default_rng(20261108)
    slub_low = cv2.resize(rng.normal(0.0, 1.0, (256, 256)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    micro_grain = rng.normal(0.0, 1.0, (size // 2, size // 2)).astype(np.float32)
    micro_grain = cv2.resize(micro_grain, (size, size), interpolation=cv2.INTER_LINEAR)

    plate_ink, _ = _render_heel_license_plate_stamp()
    patch_arc_txt = _render_patch_arc_text_mask()

    pad_px = max(14, size // 220)

    def uv_to_slice(u0: float, u1: float, v0: float, v1: float):
        x0 = max(0, int(math.floor(u0 * size)) - pad_px)
        x1 = min(size, int(math.ceil(u1 * size)) + pad_px)
        y0 = max(0, int(math.floor((1.0 - v1) * size)) - pad_px)
        y1 = min(size, int(math.ceil((1.0 - v0) * size)) + pad_px)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        u_uv = (xx + 0.5) / size
        v_uv = 1.0 - (yy + 0.5) / size
        return x0, x1, y0, y1, u_uv, v_uv

    for side in ("L", "R"):
        fd = data["feet"][side]
        med_sign = float(fd["med_sign"])

        # ==================================================================
        # 1. 靴身与包头 + 卷边内衬岛 (upper_L / upper_R)
        # ==================================================================
        u0, u1, v0, v1 = SL3.SHOE_UV_LAYOUT[f"upper_{side}"]
        _, _, _, v1_rim = SL3.SHOE_UV_LAYOUT[f"upper_{side}_rim"]
        x0, x1, y0, y1, u_uv, v_uv = uv_to_slice(u0, u1, v0, v1_rim)

        # u_local 沿圆周支持周期延拓，确保后跟拼缝跨界双线性采样 100% 连续无裂纹
        u_loc = ((u_uv - u0) / (u1 - u0)) % 1.0
        v_loc = (v_uv - v0) / (v1 - v0)  # [0..1] 为外靴身, (1..1.095] 为卷边内衬
        vt_clamped = np.clip(v_loc, 0.0, 1.0)
        is_rim = v_loc > 1.0

        theta = -math.pi + 2.0 * math.pi * u_loc
        th_abs = np.abs(theta)

        # 局部物理毫米尺度
        s_u_mm = u_loc * 370.0
        s_v_mm = v_loc * 195.0
        sub_slub = slub_low[y0:y1, x0:x1]
        sub_grain = micro_grain[y0:y1, x0:x1]

        # 12oz 重磅原色帆布经纬平纹织理
        warp = np.sin(2.0 * math.pi * s_u_mm / 0.62)
        weft = np.sin(2.0 * math.pi * s_v_mm / 0.58)
        canvas_weave = 0.55 * warp * weft + 0.25 * warp + 0.20 * weft

        alb_sub = np.empty((*u_loc.shape, 3), dtype=np.float32)
        alb_sub[..., 0] = 235.0 + 2.8 * sub_slub + 2.0 * canvas_weave
        alb_sub[..., 1] = 228.0 + 2.6 * sub_slub + 1.9 * canvas_weave
        alb_sub[..., 2] = 219.0 + 2.5 * sub_slub + 1.8 * canvas_weave
        rgh_sub = 0.85 + 0.025 * canvas_weave + 0.015 * sub_grain
        met_sub = np.zeros_like(rgh_sub)
        h_sub = 0.070 * canvas_weave + 0.025 * sub_slub

        # 内衬环微暖暗化 AO
        rim_blend = _smoothstep(1.00, 1.03, v_loc)
        alb_sub *= (1.0 - 0.065 * rim_blend)[..., None]

        # 结构环境光遮蔽 (AO)：围条底沿折痕 + 鞋眼襟叠层阴影 + 鞋舌受鞋带微遮蔽
        ao_fox = 1.0 - 0.075 * (1.0 - _smoothstep(0.03, 0.09, vt_clamped))
        ao_placket = 1.0 - 0.095 * np.exp(-(((th_abs - SL3.EYESTAY_INNER_THETA) / 0.022) ** 2)) * _smoothstep(0.24, 0.29, vt_clamped)
        ao_tongue = 1.0 - 0.045 * (1.0 - _smoothstep(0.14, 0.20, th_abs)) * _smoothstep(0.26, 0.34, vt_clamped) * (1.0 - _smoothstep(0.80, 0.88, vt_clamped))
        alb_sub *= (ao_fox * ao_placket * ao_tongue)[..., None]

        # 预先计算外靴身逐像素精确三维坐标（供三维等距包头弧线、圆标五角星与全部金属鞋眼零拉伸绘制！）
        P3 = _interp_grid_2d(fd["V_up_outer"], u_loc, vt_clamped)  # (H_s, W_s, 3)
        rel_foot_p3 = P3 - fd["foot_origin"][None, None, :]
        u_p3_fwd = np.sum(rel_foot_p3 * fd["e_fwd"][None, None, :], axis=2)
        z_p3_foot = np.sum(rel_foot_p3 * fd["e_up_foot"][None, None, :], axis=2)
        d_tc_3d_mm = ((u_p3_fwd - (float(fd["u_tip_max"]) - 0.059)) - 0.38 * np.maximum(z_p3_foot - 0.027, 0.0)) * 1000.0

        # 亮面硫化橡胶包头 (Toe Cap) —— 三维坐标严格对齐前掌菱格防撞包边竖缝，形成真实匡威 T 字交汇！
        tc_mask = _smoothstep(-0.65, +0.65, d_tc_3d_mm) * (1.0 - rim_blend)
        rubber_rgb = np.array([242.0, 238.0, 231.0], dtype=np.float32) + 1.0 * sub_grain[..., None]
        alb_sub = alb_sub * (1.0 - tc_mask[..., None]) + rubber_rgb * tc_mask[..., None]
        rgh_sub = rgh_sub * (1.0 - tc_mask) + (0.33 + 0.02 * sub_grain) * tc_mask
        h_sub = h_sub * (1.0 - tc_mask) + (0.42 + 0.025 * sub_grain) * tc_mask

        # 全套匡威经典双排明线与结构拼缝
        stitch_rgb = np.array([232.0, 225.0, 216.0], dtype=np.float32)

        def apply_stitch(dist_m_mm: np.ndarray, along_m_mm: np.ndarray, active_m: np.ndarray):
            nonlocal alb_sub, rgh_sub, h_sub
            tm, dh = _stitch_dash_field(dist_m_mm, along_m_mm)
            w_s = tm * active_m
            # 压线槽微暗化 + 亮棉线凸起
            groove_ao = 1.0 - 0.08 * np.exp(-((dist_m_mm / 0.70) ** 2)) * active_m * (1.0 - 0.6 * tm)
            alb_sub = alb_sub * groove_ao[..., None]
            alb_sub = alb_sub * (1.0 - 0.75 * w_s[..., None]) + stitch_rgb * (0.75 * w_s[..., None])
            rgh_sub = rgh_sub * (1.0 - 0.5 * w_s) + 0.78 * (0.5 * w_s)
            h_sub = h_sub + dh * active_m

        outer_m = 1.0 - rim_blend
        # a) 橡胶包头后沿三维等距双车线（沿 d_tc_3d_mm = -1.6mm 与 -4.2mm 平行伴行）
        act_tc = (np.abs(d_tc_3d_mm + 2.9) < 5.5) * _smoothstep(0.04, 0.065, v_loc) * outer_m
        along_tc_mm = np.sum(rel_foot_p3 * fd["e_side"][None, None, :], axis=2) * 1000.0 + z_p3_foot * 600.0
        apply_stitch(d_tc_3d_mm + 1.6, along_tc_mm, act_tc)
        apply_stitch(d_tc_3d_mm + 4.2, along_tc_mm + 1.2, act_tc)

        # b) 鞋舌两侧包边缝线 + 鞋舌上部横向固定方框车线（鞋舌缝线！）
        act_tng_edge = _smoothstep(0.25, 0.28, v_loc) * outer_m
        d_tng_edge = (th_abs - 0.168) * 56.0
        apply_stitch(d_tng_edge, s_v_mm, act_tng_edge)
        act_tng_bar = (1.0 - _smoothstep(0.135, 0.155, th_abs)) * outer_m
        apply_stitch((v_loc - 0.755) * 195.0, s_u_mm, act_tng_bar)
        apply_stitch((v_loc - 0.775) * 195.0, s_u_mm + 1.1, act_tng_bar)

        # c) 鞋眼襟 (Eyestay) 内侧与外侧平行双排加固明线 + U 型喉底收线
        act_eye_st = _smoothstep(0.245, 0.268, v_loc) * (1.0 - _smoothstep(0.935, 0.960, v_loc)) * outer_m
        apply_stitch((th_abs - 0.222) * 58.0, s_v_mm, act_eye_st)
        apply_stitch((th_abs - 0.238) * 58.0, s_v_mm + 1.3, act_eye_st)
        apply_stitch((th_abs - 0.365) * 58.0, s_v_mm, act_eye_st)
        apply_stitch((th_abs - 0.382) * 58.0, s_v_mm + 1.3, act_eye_st)

        # d) Chuck 70 标志性侧翼弧形加固双缝线 (Wing Stitch)
        wing_t = np.clip((th_abs - 0.375) / 0.29, 0.0, 1.0)
        v_wing = 0.435 * ((1.0 - wing_t) ** 1.45) + 0.065
        act_wing = (th_abs >= 0.372) * (th_abs <= 0.665) * outer_m
        d_wing_mm = (v_loc - v_wing) * 185.0
        apply_stitch(d_wing_mm, th_abs * 62.0, act_wing)
        apply_stitch(d_wing_mm - 2.7, th_abs * 62.0 + 1.2, act_wing)

        # e) 后跟竖向帆布加固条与两侧双排明线 (Heel Backstay Strip)
        d_heel_rad = math.pi - th_abs
        heel_strip_m = (1.0 - _smoothstep(0.112, 0.125, d_heel_rad)) * _smoothstep(0.05, 0.08, v_loc) * outer_m
        h_sub += 0.28 * heel_strip_m
        alb_sub *= (1.0 - 0.025 * np.exp(-(((d_heel_rad - 0.118) / 0.012) ** 2)) * outer_m)[..., None]
        act_heel_st = _smoothstep(0.06, 0.09, v_loc) * (1.0 - _smoothstep(0.96, 0.99, v_loc)) * outer_m
        apply_stitch((d_heel_rad - 0.092) * 55.0, s_v_mm, act_heel_st)
        apply_stitch((d_heel_rad - 0.074) * 55.0, s_v_mm + 1.2, act_heel_st)

        # 计算局部纹素物理毫米分辨率供法线图精确求导
        dP_dx = np.linalg.norm(np.diff(P3, axis=1, prepend=P3[:, :1]), axis=2) * 1000.0
        dP_dy = np.linalg.norm(np.diff(P3, axis=0, prepend=P3[:1, :]), axis=2) * 1000.0
        med_dx = float(np.median(dP_dx[dP_dx > 1e-4])) if np.any(dP_dx > 1e-4) else 0.18
        med_dy = float(np.median(dP_dy[dP_dy > 1e-4])) if np.any(dP_dy > 1e-4) else 0.18
        mm_px_x[y0:y1, x0:x1] = np.clip(np.where(dP_dx > 0.02, dP_dx, med_dx), 0.06, 0.45)
        mm_px_y[y0:y1, x0:x1] = np.clip(np.where(dP_dy > 0.02, dP_dy, med_dy), 0.06, 0.45)

        # ------------------------------------------------------------------
        # 内踝经典匡威圆标 (All-Star Medial Ankle Patch，严格仅在解剖内侧！)
        # ------------------------------------------------------------------
        med_hemisphere = ((theta * med_sign) > 0.12) & (~is_rim)
        rel_patch = (P3 - fd["patch_pos"][None, None, :]) * 1000.0
        px_mm = np.sum(rel_patch * fd["patch_t_fwd"][None, None, :], axis=2)
        py_mm = np.sum(rel_patch * fd["patch_t_up"][None, None, :], axis=2)
        pr_mm = np.hypot(px_mm, py_mm)

        r_badge = 16.2
        badge_m = _smoothstep(r_badge + 0.40, r_badge - 0.40, pr_mm) * med_hemisphere
        if np.any(badge_m > 1e-3):
            # 圆标奶油白压纹皮质底盘 + 外沿阴影
            edge_ao = 1.0 - 0.10 * np.exp(-(((pr_mm - (r_badge + 0.55)) / 0.85) ** 2)) * med_hemisphere
            alb_sub *= edge_ao[..., None]

            badge_rgb = np.array([246.0, 241.0, 234.0], dtype=np.float32) + 1.0 * sub_grain[..., None]
            alb_sub = alb_sub * (1.0 - badge_m[..., None]) + badge_rgb * badge_m[..., None]
            rgh_sub = rgh_sub * (1.0 - badge_m) + 0.52 * badge_m
            h_sub += 0.54 * badge_m

            # 圆标内圈同心压线与细藏青环线
            ang_p = np.arctan2(px_mm, py_mm)
            tm_ring, dh_ring = _stitch_dash_field(pr_mm - 14.5, (ang_p + math.pi) * 14.5, pitch_mm=2.0, half_w_mm=0.28)
            h_sub += dh_ring * badge_m
            alb_sub *= (1.0 - 0.06 * tm_ring * badge_m)[..., None]

            navy_ring_m = np.exp(-(((pr_mm - 13.0) / 0.32) ** 2)) * badge_m
            navy_ink = np.array([30.0, 50.0, 86.0], dtype=np.float32)
            alb_sub = alb_sub * (1.0 - 0.72 * navy_ring_m[..., None]) + navy_ink * (0.72 * navy_ring_m[..., None])

            # 采样微压印弧形 CONVERSE / ALL STAR 字样
            ix_t = np.clip(((px_mm / 17.4 * 0.5 + 0.5) * 511.0).astype(np.int32), 0, 511)
            iy_t = np.clip(((-py_mm / 17.4 * 0.5 + 0.5) * 511.0).astype(np.int32), 0, 511)
            arc_m = patch_arc_txt[iy_t, ix_t] * badge_m * (pr_mm > 8.9) * (pr_mm < 12.8)
            alb_sub = alb_sub * (1.0 - 0.58 * arc_m[..., None]) + navy_ink * (0.58 * arc_m[..., None])
            h_sub -= 0.08 * arc_m

            # 正中经典藏青色五角星 (R_out=8.6mm, R_in=3.3mm，严格对齐 2x 实拍比例)
            sdf_star = _star5_sdf_mm(px_mm, py_mm, r_out=8.6, r_in=3.30)
            star_m = _smoothstep(+0.28, -0.28, sdf_star) * badge_m
            alb_sub = alb_sub * (1.0 - star_m[..., None]) + navy_ink * star_m[..., None]
            rgh_sub = rgh_sub * (1.0 - star_m) + 0.45 * star_m
            h_sub += 0.14 * star_m

        # ------------------------------------------------------------------
        # 16 枚金属鞋眼孔 (7×2 鞋眼襟孔 + 2 内侧足弓透气孔) -> Metallic + Roughness + Torus Normal
        # ------------------------------------------------------------------
        for eye in fd["eyelets"]:
            e_pos = eye["pos"]
            e_nrm = eye["nrm"]
            # 在球面/柱面上仅对靠近该鞋眼的参数窗口计算，兼顾速度与防对侧穿透
            ang_d = np.abs((theta - eye["theta"] + math.pi) % (2.0 * math.pi) - math.pi)
            near = (ang_d < 0.28) & (np.abs(v_loc - eye["vt"]) < 0.12) & (~is_rim)
            if not np.any(near):
                continue

            d3_mm = (P3[near] - e_pos[None, :]) * 1000.0
            # 切平面距离（剔除法向分量）
            dn = d3_mm @ e_nrm
            dt_mm = np.sqrt(np.maximum(np.sum(d3_mm * d3_mm, axis=1) - dn * dn, 0.0))

            r_in = 2.10 if eye["kind"] == "eyestay" else 1.95
            r_out = 4.30 if eye["kind"] == "eyestay" else 4.05
            r_mid = 0.5 * (r_in + r_out)
            r_hw = 0.5 * (r_out - r_in)

            # 金属外圈压痕微阴影
            collar_ao = 1.0 - 0.16 * np.exp(-(((dt_mm - (r_out + 0.45)) / 0.65) ** 2))
            alb_near = alb_sub[near] * collar_ao[:, None]
            rgh_near = rgh_sub[near].copy()
            met_near = met_sub[near].copy()
            h_near = h_sub[near].copy()

            # 拉丝镍银金属圆环 (Metallic = 0.92, Roughness = 0.22)
            rim_m = _smoothstep(r_out + 0.25, r_out - 0.20, dt_mm) * _smoothstep(r_in - 0.20, r_in + 0.22, dt_mm)
            torus_profile = np.sqrt(np.clip(1.0 - ((dt_mm - r_mid) / (r_hw + 0.15)) ** 2, 0.0, 1.0))
            metal_rgb = np.array([214.0, 218.0, 224.0], dtype=np.float32)

            alb_near = alb_near * (1.0 - rim_m[:, None]) + metal_rgb * rim_m[:, None]
            met_near = met_near * (1.0 - rim_m) + 0.92 * rim_m
            rgh_near = rgh_near * (1.0 - rim_m) + 0.22 * rim_m
            h_near = h_near * (1.0 - rim_m) + (0.95 * torus_profile) * rim_m

            # 鞋眼中心凹孔
            hole_m = _smoothstep(r_in + 0.20, r_in - 0.22, dt_mm)
            if eye["kind"] == "vent":
                hole_rgb = np.array([34.0, 32.0, 34.0], dtype=np.float32)
            elif eye["unlaced"]:
                # 顶部未穿鞋带的第 7 孔透出内部微受阴影的白袜棉织质感
                inner_ao = 0.68 + 0.32 * _smoothstep(r_in, 0.0, dt_mm)
                hole_rgb = np.array([186.0, 180.0, 173.0], dtype=np.float32)[None, :] * inner_ao[:, None]
            else:
                hole_rgb = np.array([62.0, 58.0, 55.0], dtype=np.float32)

            alb_near = alb_near * (1.0 - hole_m[:, None]) + hole_rgb * hole_m[:, None]
            met_near = met_near * (1.0 - hole_m)
            rgh_near = rgh_near * (1.0 - hole_m) + 0.93 * hole_m
            h_near = h_near * (1.0 - hole_m) - 1.20 * hole_m

            alb_sub[near] = alb_near
            rgh_sub[near] = rgh_near
            met_sub[near] = met_near
            h_sub[near] = h_near

        albedo[y0:y1, x0:x1] = alb_sub
        rough[y0:y1, x0:x1] = rgh_sub
        metal[y0:y1, x0:x1] = met_sub
        height_mm[y0:y1, x0:x1] = h_sub

        # ==================================================================
        # 2. 硫化橡胶围条岛 (foxing_L / foxing_R)：双黑护条 + 菱格前包边 + 立体橡胶后标
        # ==================================================================
        fu0, fu1, fv0, fv1 = SL3.SHOE_UV_LAYOUT[f"foxing_{side}"]
        fx0, fx1, fy0, fy1, fu_uv, fv_uv = uv_to_slice(fu0, fu1, fv0, fv1)

        s_loc = ((fu_uv - fu0) / (fu1 - fu0)) % 1.0
        vz_loc = np.clip((fv_uv - fv0) / (fv1 - fv0), 0.0, 1.0)

        perim_mm = float(fd["total_perim_m"]) * 1000.0
        s_mm = s_loc * perim_mm
        z_mm = -2.0 + vz_loc * 32.0  # [-2.0, +30.0] mm

        mm_px_x[fy0:fy1, fx0:fx1] = perim_mm / ((fu1 - fu0) * size)
        mm_px_y[fy0:fy1, fx0:fx1] = 32.0 / ((fv1 - fv0) * size)

        # 插值前掌防撞包边掩码 bmp
        s_grid = fd["s_arc"]
        bmp = np.interp(s_loc, s_grid, fd["bumper_mask"])

        f_grain = micro_grain[fy0:fy1, fx0:fx1]
        # 竖向细硫化胶带压纹
        v_rib = 0.038 * np.sin(2.0 * math.pi * s_mm / 1.15)
        f_alb = np.empty((*s_loc.shape, 3), dtype=np.float32)
        f_alb[..., 0] = 242.0 + 1.3 * f_grain
        f_alb[..., 1] = 238.0 + 1.2 * f_grain
        f_alb[..., 2] = 231.0 + 1.1 * f_grain
        f_rgh = 0.35 + 0.025 * f_grain
        f_h = v_rib + 0.02 * f_grain

        # 底部 1.4mm 深棕生胶大底外露边
        gum_m = 1.0 - _smoothstep(0.6, 1.6, z_mm)
        gum_rgb = np.array([86.0, 58.0, 42.0], dtype=np.float32)
        f_alb = f_alb * (1.0 - gum_m[..., None]) + gum_rgb * gum_m[..., None]
        f_rgh = f_rgh * (1.0 - gum_m) + 0.58 * gum_m

        # 前掌菱格纹防撞包边 (Front Diamond Toe Bumper) + 左右竖向包边台阶缝
        bmp_zone = bmp * _smoothstep(1.2, 2.2, z_mm) * (1.0 - _smoothstep(23.5, 24.8, z_mm))
        d1 = np.abs(np.sin(math.pi * (s_mm + z_mm) / 3.1))
        d2 = np.abs(np.sin(math.pi * (s_mm - z_mm) / 3.1))
        diamonds = (d1 * d2) ** 0.65
        f_h += bmp_zone * (0.35 + 0.26 * diamonds)
        f_rgh = f_rgh * (1.0 - 0.3 * bmp_zone) + 0.40 * (0.3 * bmp_zone)
        # 包边左右交界竖缝微阴影
        seam_bmp = np.exp(-(((bmp - 0.5) / 0.16) ** 2)) * _smoothstep(1.2, 2.2, z_mm) * (1.0 - _smoothstep(23.8, 25.0, z_mm))
        f_alb *= (1.0 - 0.12 * seam_bmp)[..., None]
        f_h -= 0.22 * seam_bmp

        stripe_rgb = np.array([23.0, 22.0, 25.0], dtype=np.float32)
        # 上黑护条：360° 通长连续环绕鞋口围条上沿 (z_mm ∈ [25.5, 28.1])
        up_st = _smoothstep(25.2, 25.7, z_mm) * (1.0 - _smoothstep(27.9, 28.4, z_mm))
        # 下黑护条：腰线高度 (z_mm ∈ [12.3, 14.7])，严格在前掌防撞包边竖缝处截断 (1 - bmp)！
        lo_st = _smoothstep(12.0, 12.5, z_mm) * (1.0 - _smoothstep(14.5, 15.0, z_mm)) * (1.0 - _smoothstep(0.25, 0.55, bmp))
        st_m = np.clip(up_st + lo_st, 0.0, 1.0)

        f_alb = f_alb * (1.0 - st_m[..., None]) + stripe_rgb * st_m[..., None]
        f_rgh = f_rgh * (1.0 - st_m) + 0.38 * st_m
        f_h += 0.20 * st_m

        # 后跟立体黑色橡胶后标 ("CONVERSE ALL ★ STAR", 38mm × 14.8mm)
        s_heel_mm = float(fd["s_heel_center"]) * perim_mm
        ds_h = (((s_mm - s_heel_mm + 0.5 * perim_mm) % perim_mm) - 0.5 * perim_mm) * float(fd["heel_lr_sign"])
        dz_h = z_mm - 10.2  # 中心高度 10.2mm (跨 [2.8, 17.6]mm)
        hw_p, hh_p, r_cor = 19.0, 7.4, 2.0
        qx = np.maximum(np.abs(ds_h) - (hw_p - r_cor), 0.0)
        qy = np.maximum(np.abs(dz_h) - (hh_p - r_cor), 0.0)
        sdf_plate = np.hypot(qx, qy) - r_cor
        plate_m = _smoothstep(+0.35, -0.35, sdf_plate)

        if np.any(plate_m > 1e-3):
            plaque_blk = np.array([20.0, 20.0, 23.0], dtype=np.float32)
            plaque_wht = np.array([238.0, 235.0, 228.0], dtype=np.float32)
            ph, pw = plate_ink.shape
            px_i = np.clip(((ds_h / (2.0 * hw_p) + 0.5) * (pw - 1)).astype(np.int32), 0, pw - 1)
            py_i = np.clip(((0.5 - dz_h / (2.0 * hh_p)) * (ph - 1)).astype(np.int32), 0, ph - 1)
            ink_val = plate_ink[py_i, px_i] * plate_m

            plaque_col = plaque_blk[None, None, :] * (1.0 - ink_val[..., None]) + plaque_wht[None, None, :] * ink_val[..., None]
            f_alb = f_alb * (1.0 - plate_m[..., None]) + plaque_col * plate_m[..., None]
            f_rgh = f_rgh * (1.0 - plate_m) + (0.33 + 0.10 * ink_val) * plate_m
            f_h = f_h * (1.0 - plate_m) + (0.85 + 0.22 * ink_val) * plate_m

        albedo[fy0:fy1, fx0:fx1] = f_alb
        rough[fy0:fy1, fx0:fx1] = f_rgh
        metal[fy0:fy1, fx0:fx1] = 0.0
        height_mm[fy0:fy1, fx0:fx1] = f_h

        # ==================================================================
        # 3. 生胶防滑华夫大底与鞋底上沿封盖 (sole_bot & sole_top)
        # ==================================================================
        bu0, bu1, bv0, bv1 = SL3.SHOE_UV_LAYOUT[f"sole_bot_{side}"]
        bx0, bx1, by0, by1, bu_uv, bv_uv = uv_to_slice(bu0, bu1, bv0, bv1)
        su_b = (bu_uv - bu0) / (bu1 - bu0) * 110.0
        sv_b = (bv_uv - bv0) / (bv1 - bv0) * 255.0
        waffle = np.abs(np.sin(math.pi * (su_b + sv_b) / 7.5)) * np.abs(np.sin(math.pi * (su_b - sv_b) / 7.5))
        albedo[by0:by1, bx0:bx1] = np.array([88.0, 60.0, 43.0], dtype=np.float32) * (0.88 + 0.16 * waffle[..., None])
        rough[by0:by1, bx0:bx1] = 0.60
        height_mm[by0:by1, bx0:bx1] = 0.65 * waffle

        tu0, tu1, tv0, tv1 = SL3.SHOE_UV_LAYOUT[f"sole_top_{side}"]
        tx0, tx1, ty0, ty1, _, _ = uv_to_slice(tu0, tu1, tv0, tv1)
        albedo[ty0:ty1, tx0:tx1] = np.array([234.0, 229.0, 221.0], dtype=np.float32)
        rough[ty0:ty1, tx0:tx1] = 0.42
        height_mm[ty0:ty1, tx0:tx1] = 0.0

        # ==================================================================
        # 4. 扁平人字纹编织棉鞋带 + 塑料鞋带头箍 (laces_L / laces_R)
        # ==================================================================
        lu0, lu1, lv0, lv1 = SL3.SHOE_UV_LAYOUT[f"laces_{side}"]
        lx0, lx1, ly0, ly1, lu_uv, lv_uv = uv_to_slice(lu0, lu1, lv0, lv1)
        lu_mm = (lu_uv - lu0) / (lu1 - lu0) * 260.0
        lv_mm = (lv_uv - lv0) / (lv1 - lv0) * 58.0
        braid = 0.5 * np.sin(2.0 * math.pi * (lu_mm * 2.2 + lv_mm) / 1.1) + 0.5 * np.sin(2.0 * math.pi * (lu_mm * 2.2 - lv_mm) / 1.1)
        l_grain = micro_grain[ly0:ly1, lx0:lx1]
        l_alb = np.empty((*lu_uv.shape, 3), dtype=np.float32)
        l_alb[..., 0] = 241.0 + 3.5 * braid + 1.5 * l_grain
        l_alb[..., 1] = 238.0 + 3.4 * braid + 1.5 * l_grain
        l_alb[..., 2] = 232.0 + 3.2 * braid + 1.4 * l_grain
        l_rgh = 0.90 + 0.03 * braid
        l_h = 0.16 * braid

        # 最后两根垂尾末端的半透明硬质塑料鞋带头 (Aglets)
        u_rel_l = (lu_uv - lu0) / (lu1 - lu0)
        v_rel_l = (lv_uv - lv0) / (lv1 - lv0)
        aglet_m = (u_rel_l > (14.0 / 16.0)) * _smoothstep(0.80, 0.84, v_rel_l)
        l_alb = l_alb * (1.0 - 0.35 * aglet_m[..., None]) + np.array([232.0, 230.0, 226.0], dtype=np.float32) * (0.35 * aglet_m[..., None])
        l_rgh = l_rgh * (1.0 - aglet_m) + 0.26 * aglet_m
        l_h = l_h * (1.0 - aglet_m) + 0.05 * aglet_m

        albedo[ly0:ly1, lx0:lx1] = l_alb
        rough[ly0:ly1, lx0:lx1] = l_rgh
        metal[ly0:ly1, lx0:lx1] = 0.0
        height_mm[ly0:ly1, lx0:lx1] = l_h
        mm_px_x[ly0:ly1, lx0:lx1] = 0.15
        mm_px_y[ly0:ly1, lx0:lx1] = 0.15

    # 保存 4 张 4K 靴身 PBR 贴图
    alb_bgr = np.clip(np.round(albedo[..., ::-1]), 0, 255).astype(np.uint8)
    rgh_u8 = np.clip(np.round(rough * 255.0), 0, 255).astype(np.uint8)
    met_u8 = np.clip(np.round(metal * 255.0), 0, 255).astype(np.uint8)
    nrm_bgr = _height_mm_to_opengl_normal_bgr(height_mm, mm_px_x, mm_px_y)

    cv2.imwrite(str(out_dir / "shoe_albedo_v3.png"), alb_bgr)
    cv2.imwrite(str(out_dir / "shoe_roughness_v3.png"), rgh_u8)
    cv2.imwrite(str(out_dir / "shoe_metallic_v3.png"), met_u8)
    cv2.imwrite(str(out_dir / "shoe_normal_v3.png"), nrm_bgr)
    print(f"[shoe_tex] 已写入 4K 帆布鞋四通道 PBR 贴图 ({size}x{size})")
    return alb_bgr, nrm_bgr, met_u8


def build_sock_pbr_maps(out_dir: Path, size: int = 2048):
    """合成 2K 独立非重叠螺纹堆堆白袜三通道 PBR 贴图套件。"""
    albedo = np.full((size, size, 3), (233, 227, 220), dtype=np.float32)
    rough = np.full((size, size), 0.93, dtype=np.float32)
    height_mm = np.zeros((size, size), dtype=np.float32)
    mm_px_x = np.full((size, size), 0.24, dtype=np.float32)
    mm_px_y = np.full((size, size), 0.16, dtype=np.float32)

    rng = np.random.default_rng(20261109)
    yarn_fuzz = cv2.resize(rng.normal(0.0, 1.0, (512, 512)).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    pad_px = 14

    for side, rx_sign in (("L", +1.0), ("R", -1.0)):
        su0, su1, sv0, sv1 = SL3.SOCK_UV_LAYOUT[f"sock_{side}"]
        _, _, _, sv1_r = SL3.SOCK_UV_LAYOUT[f"sock_{side}_rim"]
        x0 = max(0, int(math.floor(su0 * size)) - pad_px)
        x1 = min(size, int(math.ceil(su1 * size)) + pad_px)
        y0 = max(0, int(math.floor((1.0 - sv1_r) * size)) - pad_px)
        y1 = min(size, int(math.ceil((1.0 - sv0) * size)) + pad_px)

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        u_uv = (xx + 0.5) / size
        v_uv = 1.0 - (yy + 0.5) / size

        u_loc = ((u_uv - su0) / (su1 - su0)) % 1.0
        s_loc = (v_uv - sv0) / (sv1 - sv0)
        th = -math.pi + 2.0 * math.pi * u_loc

        # 纵向 2x2 精梳棉罗纹针织槽 (84 道纵向瓦楞 + 细腻横向线圈)
        rib_wale = np.cos(2.0 * math.pi * 84.0 * u_loc)
        cuff_tight = _smoothstep(0.86, 0.93, s_loc)
        rib_top = np.cos(2.0 * math.pi * 112.0 * u_loc)
        rib = (1.0 - cuff_tight) * rib_wale + cuff_tight * rib_top
        knit_course = np.sin(2.0 * math.pi * 145.0 * s_loc + 0.6 * np.sin(2.0 * math.pi * 84.0 * u_loc))

        # 与三维堆堆褶皱几何完全同相位的柔和折谷环境光遮蔽 (Crease AO)
        slouch_env = _smoothstep(0.32, 0.45, np.clip(s_loc, 0.0, 1.0)) * (1.0 - 0.30 * _smoothstep(0.92, 1.00, np.clip(s_loc, 0.0, 1.0)))
        phase_mod = 0.28 * np.sin(2.0 * th + 0.7 * rx_sign) + 0.16 * np.cos(3.0 * th - 0.5)
        wave_arg = 2.0 * math.pi * 4.15 * ((np.clip(s_loc, 0.0, 1.0) - 0.36) / 0.60) + phase_mod
        valley_ao = 1.0 - 0.062 * slouch_env * (0.5 - 0.5 * np.sin(wave_arg)) ** 1.35
        rib_ao = 1.0 - 0.024 * (0.5 - 0.5 * rib)
        rim_ao = 1.0 - 0.05 * _smoothstep(1.0, 1.06, s_loc)

        fz = yarn_fuzz[y0:y1, x0:x1]
        s_alb = np.empty((*u_loc.shape, 3), dtype=np.float32)
        s_alb[..., 0] = 245.0 + 2.0 * fz
        s_alb[..., 1] = 240.0 + 1.9 * fz
        s_alb[..., 2] = 234.0 + 1.8 * fz
        s_alb *= (valley_ao * rib_ao * rim_ao)[..., None]

        albedo[y0:y1, x0:x1] = s_alb
        rough[y0:y1, x0:x1] = np.clip(0.93 - 0.02 * rib + 0.02 * fz, 0.85, 0.98)
        height_mm[y0:y1, x0:x1] = 0.17 * rib + 0.050 * knit_course + 0.020 * fz

    alb_bgr = np.clip(np.round(albedo[..., ::-1]), 0, 255).astype(np.uint8)
    rgh_u8 = np.clip(np.round(rough * 255.0), 0, 255).astype(np.uint8)
    nrm_bgr = _height_mm_to_opengl_normal_bgr(height_mm, mm_px_x, mm_px_y)

    cv2.imwrite(str(out_dir / "sock_albedo_v3.png"), alb_bgr)
    cv2.imwrite(str(out_dir / "sock_roughness_v3.png"), rgh_u8)
    cv2.imwrite(str(out_dir / "sock_normal_v3.png"), nrm_bgr)
    print(f"[sock_tex] 已写入 2K 堆堆白袜三通道 PBR 贴图 ({size}x{size})")
    return alb_bgr, nrm_bgr


def save_uv_diagnostic_sheet(data: dict, shoe_alb_bgr: np.ndarray, shoe_nrm_bgr: np.ndarray,
                             sock_alb_bgr: np.ndarray, out_dir: Path):
    """输出独立非重叠 UV 网格叠印诊断图，直观展示左右靴、围条、大底、鞋带与白袜零重叠排布。"""
    panel = 900

    def draw_uv_wire(bg_bgr: np.ndarray, mesh_list: list[tuple[np.ndarray, np.ndarray]], title: str) -> np.ndarray:
        canvas = cv2.resize(bg_bgr, (panel, panel), interpolation=cv2.INTER_AREA)
        for f_arr, uv_arr in mesh_list:
            pts = np.empty_like(uv_arr)
            pts[:, 0] = uv_arr[:, 0] * (panel - 1)
            pts[:, 1] = (1.0 - uv_arr[:, 1]) * (panel - 1)
            pts_i = np.round(pts).astype(np.int32)
            # 抽样绘制三角边使网格线干净通透不糊片
            tris = pts_i[f_arr[::2]].reshape(-1, 3, 2)
            cv2.polylines(canvas, list(tris), isClosed=True, color=(52, 96, 48), thickness=1, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, (12, 12), (12 + len(title) * 11 + 24, 48), (24, 24, 28), -1)
        cv2.putText(canvas, title, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        return canvas

    shoe_meshes = [
        (data["feet"]["L"]["shoe_f"], data["feet"]["L"]["shoe_uv"]),
        (data["feet"]["R"]["shoe_f"], data["feet"]["R"]["shoe_uv"]),
    ]
    sock_meshes = [
        (data["feet"]["L"]["sock_f"], data["feet"]["L"]["sock_uv"]),
        (data["feet"]["R"]["sock_f"], data["feet"]["R"]["sock_uv"]),
    ]
    p1 = draw_uv_wire(shoe_alb_bgr, shoe_meshes, "Boot Albedo + Non-Overlapping UV (L & R)")
    p2 = draw_uv_wire(shoe_nrm_bgr, [], "Boot Tangent Normal Map (4K OpenGL)")
    p3 = draw_uv_wire(sock_alb_bgr, sock_meshes, "Crew Sock Albedo + Non-Overlapping UV (L & R)")

    sheet = np.hstack([p1, p2, p3])
    out_p = out_dir / "uv_layout_footwear_v3.png"
    cv2.imwrite(str(out_p), sheet)
    print(f"[uv_diag] 已写入独立非重叠 UV 诊断图 {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-npz", type=str, default=r"work\q004-base\v2\mesh_lod0_v2.npz")
    ap.add_argument("--out-dir", type=str, default=r"work\q004-base\v3\tex")
    ap.add_argument("--shoe-size", type=int, default=4096)
    ap.add_argument("--sock-size", type=int, default=2048)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = SL3.compute_footwear_data_v3(args.mesh_npz)

    # 合并左右双脚全部三角面做严格全局 UV 非重叠栅格校验
    v_off_sh = len(data["feet"]["L"]["shoe_uv"])
    all_shoe_uv = np.vstack([data["feet"]["L"]["shoe_uv"], data["feet"]["R"]["shoe_uv"]])
    all_shoe_f = np.vstack([data["feet"]["L"]["shoe_f"], data["feet"]["R"]["shoe_f"] + v_off_sh])
    ov_shoe_ratio, ov_shoe_px = SL3.verify_non_overlapping_uvs(all_shoe_f, all_shoe_uv, res=1024, label="shoe_L+R_combined")

    v_off_sk = len(data["feet"]["L"]["sock_uv"])
    all_sock_uv = np.vstack([data["feet"]["L"]["sock_uv"], data["feet"]["R"]["sock_uv"]])
    all_sock_f = np.vstack([data["feet"]["L"]["sock_f"], data["feet"]["R"]["sock_f"] + v_off_sk])
    ov_sock_ratio, ov_sock_px = SL3.verify_non_overlapping_uvs(all_sock_f, all_sock_uv, res=1024, label="sock_L+R_combined")

    if ov_shoe_ratio > 0.0005 or ov_sock_ratio > 0.0005:
        raise RuntimeError(f"检测到非零 UV 重叠: shoe={ov_shoe_px}px, sock={ov_sock_px}px")

    shoe_alb, shoe_nrm, _ = build_shoe_pbr_maps(data, out_dir, size=args.shoe_size)
    sock_alb, _ = build_sock_pbr_maps(out_dir, size=args.sock_size)
    save_uv_diagnostic_sheet(data, shoe_alb, shoe_nrm, sock_alb, out_dir)


if __name__ == "__main__":
    main()
