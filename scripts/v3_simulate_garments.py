"""v3 独立服装壳体剪裁建模与 Blender 物理布料解算（Shirt Shell + Denim Shorts Shell）。

核心能力：
1. 零狭长片（Zero-Sliver）水平集剪裁 + 边界切向光顺：
   - 剪裁前将 |field| < snap_tol (2.4mm) 的顶点精确吸附至 0 等值线并直接复用原顶点，
     从拓扑根源上 100% 杜绝 <0.1mm 微小狭长三角面与布料解算尖刺爆炸；
   - 对所有剪裁开口边界环执行切向光顺松弛，形成顺滑裁片弧线。
2. 宽松古巴领短款印花衬衫壳体 (shirt_shell)：
   - 单向凸桥接抚平胸沟/腋窝凹陷紧绷感，宽松直筒下摆外扩 (+9~29mm)、宽短袖筒外扩 (+9~27mm)；
   - 沿光顺领口曲线挤出平整挺括的三维立领折线与外翻古巴领尖 (Camp Lapels)，确保与胸前衣面保持真实空气间隙；
   - Blender CLOTH 轻柔棉麻/人造丝物理参数 + 真实 20.5° 俯拍重力垂坠解算。
3. 高腰毛边水洗牛仔短裤壳体 (shorts_shell)：
   - 前低后高人体工学高腰裁剪（前腰 95.8cm 露肚脐、后腰 99.8cm 完整包覆后腰臀上沿）；
   - 完整闭合裤裆底弧 (ry 下探至 73.8cm)，臀沟与腹股沟凸桥接 + 微喇裤口外扩 (+18mm)；
   - Blender CLOTH 12oz 重磅牛仔物理参数解算，挺括无折刺。
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import char_lib_v2 as CL2


def compute_vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    nrm = np.zeros_like(v, dtype=np.float64)
    v64 = v.astype(np.float64)
    fn = np.cross(v64[f[:, 1]] - v64[f[:, 0]], v64[f[:, 2]] - v64[f[:, 0]])
    np.add.at(nrm, f[:, 0], fn)
    np.add.at(nrm, f[:, 1], fn)
    np.add.at(nrm, f[:, 2], fn)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    return nrm.astype(np.float32)


def build_vertex_adjacency(n_verts: int, faces: np.ndarray) -> list[np.ndarray]:
    adj = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return [np.array(sorted(s), dtype=np.int32) for s in adj]


def clip_mesh_by_field(
    v3d: np.ndarray,
    rest3d: np.ndarray,
    uv2d: np.ndarray,
    faces: np.ndarray,
    tfaces: np.ndarray,
    field_vals: np.ndarray,
    cut_tag: int,
    vert_tags: np.ndarray,
    snap_tol: float = 0.24,
):
    """零狭长片水平集裁剪：吸附近界顶点直接复用，彻底根除微小退化三角面，同时保持 3D 流形焊接与 UV 独立。"""
    fv = field_vals.astype(np.float64).copy()
    fv[np.abs(fv) < snap_tol] = 0.0

    v_list = v3d.tolist()
    r_list = rest3d.tolist()
    tag_list = vert_tags.tolist()
    uv_list = uv2d.tolist()

    for i in range(len(fv)):
        if fv[i] == 0.0:
            tag_list[i] = cut_tag

    edge_3d_cache = {}
    edge_uv_cache = {}
    new_faces = []
    new_tfaces = []

    def get_split_3d(i0: int, i1: int, s0: float, s1: float) -> tuple[int, float]:
        if s0 == 0.0:
            tag_list[i0] = cut_tag
            return i0, 0.0
        if s1 == 0.0:
            tag_list[i1] = cut_tag
            return i1, 1.0
        t = float(s0 / (s0 - s1))
        if i0 < i1:
            key = (i0, i1)
            t_canon = t
        else:
            key = (i1, i0)
            t_canon = 1.0 - t
        if key in edge_3d_cache:
            return edge_3d_cache[key], t
        pa, pb = v_list[key[0]], v_list[key[1]]
        ra, rb = r_list[key[0]], r_list[key[1]]
        idx = len(v_list)
        v_list.append([pa[k] + t_canon * (pb[k] - pa[k]) for k in range(3)])
        r_list.append([ra[k] + t_canon * (rb[k] - ra[k]) for k in range(3)])
        tag_list.append(cut_tag)
        edge_3d_cache[key] = idx
        return idx, t

    def get_split_uv(u0: int, u1: int, t: float) -> int:
        if t <= 0.0:
            return u0
        if t >= 1.0:
            return u1
        if u0 < u1:
            key = (u0, u1)
            t_c = t
        else:
            key = (u1, u0)
            t_c = 1.0 - t
        if key in edge_uv_cache:
            return edge_uv_cache[key]
        ua, ub = uv_list[key[0]], uv_list[key[1]]
        idx = len(uv_list)
        uv_list.append([ua[0] + t_c * (ub[0] - ua[0]), ua[1] + t_c * (ub[1] - ua[1])])
        edge_uv_cache[key] = idx
        return idx

    def emit_tri(p: int, q: int, r: int, up: int, uq: int, ur: int):
        if p == q or q == r or p == r:
            return
        vp = np.asarray(v_list[p], dtype=np.float64)
        vq = np.asarray(v_list[q], dtype=np.float64)
        vr = np.asarray(v_list[r], dtype=np.float64)
        area2 = float(np.linalg.norm(np.cross(vq - vp, vr - vp)))
        if area2 < 1e-7:
            return
        new_faces.append([p, q, r])
        new_tfaces.append([up, uq, ur])

    for fi in range(len(faces)):
        vi = faces[fi]
        ui = tfaces[fi]
        si = [float(fv[vi[0]]), float(fv[vi[1]]), float(fv[vi[2]])]
        # 至少有一个严格正内部顶点才保留（避免仅靠零等值线孤立边的退化片）
        if max(si) <= 0.0:
            continue
        ins = [s >= 0.0 for s in si]
        cnt = sum(ins)
        if cnt == 3:
            emit_tri(int(vi[0]), int(vi[1]), int(vi[2]), int(ui[0]), int(ui[1]), int(ui[2]))
            continue

        target_state = (cnt == 1)
        rot = 0
        for k in range(3):
            if ins[k] == target_state:
                rot = k
                break
        idx_order = [rot, (rot + 1) % 3, (rot + 2) % 3]
        a, b, c = [int(vi[k]) for k in idx_order]
        ua, ub, uc = [int(ui[k]) for k in idx_order]
        sa, sb, sc = [si[k] for k in idx_order]

        v_ab, t_ab = get_split_3d(a, b, sa, sb)
        v_ac, t_ac = get_split_3d(a, c, sa, sc)
        uv_ab = get_split_uv(ua, ub, t_ab)
        uv_ac = get_split_uv(ua, uc, t_ac)

        if cnt == 1:
            emit_tri(a, v_ab, v_ac, ua, uv_ab, uv_ac)
        else:
            emit_tri(v_ab, b, c, uv_ab, ub, uc)
            emit_tri(v_ab, c, v_ac, uv_ab, uc, uv_ac)

    f_arr = np.asarray(new_faces, dtype=np.int32)
    tf_arr = np.asarray(new_tfaces, dtype=np.int32)

    used_v = np.unique(f_arr)
    v_remap = np.full(len(v_list), -1, dtype=np.int32)
    v_remap[used_v] = np.arange(len(used_v), dtype=np.int32)

    used_uv = np.unique(tf_arr)
    uv_remap = np.full(len(uv_list), -1, dtype=np.int32)
    uv_remap[used_uv] = np.arange(len(used_uv), dtype=np.int32)

    v_out = np.asarray(v_list, dtype=np.float32)[used_v]
    r_out = np.asarray(r_list, dtype=np.float32)[used_v]
    t_out = np.asarray(tag_list, dtype=np.int32)[used_v]
    uv_out = np.asarray(uv_list, dtype=np.float32)[used_uv]
    return v_out, r_out, uv_out, v_remap[f_arr], uv_remap[tf_arr], t_out


def smooth_cut_boundaries(v: np.ndarray, r: np.ndarray, faces: np.ndarray, tags: np.ndarray, iters: int = 8):
    """对裁剪产生的开口边界环执行切向光顺，消除三角网格锯齿折线，使领口/袖口/下摆/裤口圆润齐整。"""
    half_edges = set()
    for a, b, c in faces:
        half_edges.add((int(a), int(b)))
        half_edges.add((int(b), int(c)))
        half_edges.add((int(c), int(a)))

    b_adj = [set() for _ in range(len(v))]
    is_bnd = np.zeros(len(v), dtype=bool)
    for a, b in half_edges:
        if (b, a) not in half_edges:
            b_adj[a].add(b)
            b_adj[b].add(a)
            is_bnd[a] = True
            is_bnd[b] = True

    adj_all = build_vertex_adjacency(len(v), faces)
    v_cur = v.astype(np.float64).copy()
    r_cur = r.astype(np.float64).copy()

    for _ in range(iters):
        v_next = v_cur.copy()
        r_next = r_cur.copy()
        for i in range(len(v_cur)):
            if is_bnd[i] and tags[i] > 0 and len(b_adj[i]) == 2:
                nbs = list(b_adj[i])
                # 仅在同一裁剪标签段内做强光顺，保留领尖与袖口转角
                if tags[nbs[0]] == tags[i] and tags[nbs[1]] == tags[i]:
                    v_next[i] = 0.50 * v_cur[i] + 0.25 * (v_cur[nbs[0]] + v_cur[nbs[1]])
                    r_next[i] = 0.70 * r_cur[i] + 0.15 * (r_cur[nbs[0]] + r_cur[nbs[1]])
        v_cur = v_next
        r_cur = r_next

    # 对紧邻边界的一环内部顶点做 2 轮轻微过渡松弛，确保边界内侧三角面匀称饱满
    near_bnd = np.zeros(len(v), dtype=bool)
    for i in np.where(is_bnd)[0]:
        for nb in adj_all[i]:
            if not is_bnd[nb]:
                near_bnd[nb] = True
    for _ in range(2):
        v_next = v_cur.copy()
        for i in np.where(near_bnd)[0]:
            if len(adj_all[i]) > 0:
                v_next[i] = 0.72 * v_cur[i] + 0.28 * v_cur[adj_all[i]].mean(axis=0)
        v_cur = v_next

    return v_cur.astype(np.float32), r_cur.astype(np.float32)


def convex_bridge_relax(
    v_shell: np.ndarray,
    faces: np.ndarray,
    b_ref: np.ndarray,
    n_ref: np.ndarray,
    min_clear: np.ndarray,
    pin_mask: np.ndarray,
    iters: int = 20,
    omega: float = 0.55,
) -> np.ndarray:
    """单向凸桥接松弛：拉平胸沟/腋窝/臀沟等人体凹陷，使衣物像真实布料一样跨过波峰，同时保证绝不穿透体表。"""
    adj = build_vertex_adjacency(len(v_shell), faces)
    p = v_shell.astype(np.float64).copy()
    b64 = b_ref.astype(np.float64)
    n64 = n_ref.astype(np.float64)
    mc64 = min_clear.astype(np.float64)

    for _ in range(iters):
        p_next = p.copy()
        for i in range(len(p)):
            if pin_mask[i] or len(adj[i]) == 0:
                continue
            avg = p[adj[i]].mean(axis=0)
            cand = (1.0 - omega) * p[i] + omega * avg
            signed_d = float(np.dot(cand - b64[i], n64[i]))
            if signed_d < mc64[i]:
                cand = cand + n64[i] * (mc64[i] - signed_d)
            p_next[i] = cand
        p = p_next
    return p.astype(np.float32)


def extrude_cuban_collar(
    v_sh: np.ndarray,
    r_sh: np.ndarray,
    uv_sh: np.ndarray,
    f_sh: np.ndarray,
    tf_sh: np.ndarray,
    tags_sh: np.ndarray,
    head_up: np.ndarray,
):
    """沿高度光顺后的领口边界环挤出平整挺括的三维立领座与外翻古巴领尖（消除手风琴褶皱）。"""
    half_edges = set()
    for a, b, c in f_sh:
        half_edges.add((int(a), int(b)))
        half_edges.add((int(b), int(c)))
        half_edges.add((int(c), int(a)))

    neck_next = {}
    neck_prev = set()
    for a, b in half_edges:
        if (b, a) in half_edges:
            continue
        if (tags_sh[a] == 4 or tags_sh[b] == 4) and r_sh[a, 1] >= 127.2 and r_sh[b, 1] >= 127.2:
            if abs(r_sh[a, 0]) < 10.5 and abs(r_sh[b, 0]) < 10.5:
                neck_next[a] = b
                neck_prev.add(b)

    if len(neck_next) < 10:
        print("[collar] 未找到足够领口边界边，跳过翻领挤出")
        return v_sh, r_sh, uv_sh, f_sh, tf_sh, tags_sh

    starts = [k for k in neck_next.keys() if k not in neck_prev]
    cur = starts[0] if starts else next(iter(neck_next.keys()))
    chain = [cur]
    visited = {cur}
    while cur in neck_next:
        nxt = neck_next[cur]
        if nxt in visited:
            break
        chain.append(nxt)
        visited.add(nxt)
        cur = nxt

    m_len = len(chain)
    if m_len < 10:
        return v_sh, r_sh, uv_sh, f_sh, tf_sh, tags_sh

    # 1. 先将领口基线链本身光顺为 C2 级裁缝样条曲线，并同步回写衬衫顶点消除锯齿根源
    v_work = v_sh.astype(np.float64).copy()
    pts0 = v_work[chain].copy()
    for _ in range(14):
        pts0[1:-1] = 0.25 * pts0[:-2] + 0.50 * pts0[1:-1] + 0.25 * pts0[2:]
    v_work[chain] = pts0

    adj_sh = build_vertex_adjacency(len(v_work), f_sh)
    chain_set = set(chain)
    ring1 = [nb for ci in chain for nb in adj_sh[ci] if nb not in chain_set]
    ring1 = sorted(set(ring1))
    for _ in range(3):
        v_next = v_work.copy()
        for idx in ring1:
            v_next[idx] = 0.65 * v_work[idx] + 0.35 * v_work[adj_sh[idx]].mean(axis=0)
        v_work = v_next

    nrm = compute_vertex_normals(v_work.astype(np.float32), f_sh).astype(np.float64)
    pts0 = v_work[chain].copy()
    seg_d = np.linalg.norm(np.diff(pts0, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_d)])
    s_param = arc / max(float(arc[-1]), 1e-6)

    # 2. 计算高度平滑的外翻方向场与法向场（24 轮滤波彻底杜绝局部扭转和百褶裙效应）
    out_dirs = []
    n_dirs = []
    widths = []
    for k in range(m_len):
        p_prev = pts0[max(k - 2, 0)]
        p_next = pts0[min(k + 2, m_len - 1)]
        tang = p_next - p_prev
        tang /= np.linalg.norm(tang) + 1e-9
        nv = nrm[chain[k]]
        nv /= np.linalg.norm(nv) + 1e-9
        across = np.cross(nv, tang)
        across /= np.linalg.norm(across) + 1e-9
        out_dirs.append(across)
        n_dirs.append(nv)

        rz_k = float(r_sh[chain[k], 2])
        # 经典古巴领尖廓形 (Notched Camp Collar)：V 领底端收尖 -> 锁骨胸前外展宽尖领 (5.8cm) -> 后颈圆顺立翻领 (3.1cm)
        s_k = float(s_param[k])
        d_end = min(s_k, 1.0 - s_k)
        end_ramp = float(np.clip((d_end - 0.01) / 0.065, 0.12, 1.0))
        lapel_peak = math.exp(-((d_end - 0.17) / 0.095) ** 2)
        front_w = 0.5 * (1.0 + math.tanh(rz_k / 2.5))
        w_val = (0.031 * (1.0 - front_w) + (0.043 + 0.017 * lapel_peak) * front_w) * end_ramp
        widths.append(w_val)

    out_arr = np.asarray(out_dirs, dtype=np.float64)
    n_arr = np.asarray(n_dirs, dtype=np.float64)
    w_arr = np.asarray(widths, dtype=np.float64)
    for _ in range(24):
        out_arr[1:-1] = 0.25 * out_arr[:-2] + 0.50 * out_arr[1:-1] + 0.25 * out_arr[2:]
        out_arr /= np.linalg.norm(out_arr, axis=1, keepdims=True) + 1e-9
        n_arr[1:-1] = 0.25 * n_arr[:-2] + 0.50 * n_arr[1:-1] + 0.25 * n_arr[2:]
        n_arr /= np.linalg.norm(n_arr, axis=1, keepdims=True) + 1e-9
    for _ in range(5):
        w_arr[1:-1] = 0.25 * w_arr[:-2] + 0.50 * w_arr[1:-1] + 0.25 * w_arr[2:]

    e_up = head_up.astype(np.float64)
    row1_pts = pts0 + n_arr * 0.0068 + e_up[None, :] * 0.0052 + out_arr * (w_arr[:, None] * 0.16)
    row2_pts = pts0 + out_arr * w_arr[:, None] + n_arr * 0.0135 - e_up[None, :] * 0.0045

    # 对翻领外沿执行平滑松弛并保证相对下方胸前衣面悬浮 >= 7.5mm 空气间隙
    kd_sh = KDTree(len(v_work))
    for i, pt in enumerate(v_work):
        kd_sh.insert(pt.tolist(), i)
    kd_sh.balance()

    for _ in range(8):
        row1_pts[1:-1] = 0.22 * row1_pts[:-2] + 0.56 * row1_pts[1:-1] + 0.22 * row1_pts[2:]
        row2_pts[1:-1] = 0.25 * row2_pts[:-2] + 0.50 * row2_pts[1:-1] + 0.25 * row2_pts[2:]
        for k in range(m_len):
            _, idx_near, _ = kd_sh.find(row2_pts[k].tolist())
            gap_k = float(np.dot(row2_pts[k] - v_work[idx_near], nrm[idx_near]))
            if gap_k < 0.0075:
                row2_pts[k] += nrm[idx_near] * (0.0075 - gap_k)

    v_list = v_work.tolist()
    r_list = r_sh.tolist()
    t_list = tags_sh.tolist()
    uv_list = uv_sh.tolist()
    f_list = f_sh.tolist()
    tf_list = tf_sh.tolist()

    row0_uv = []
    for k in range(m_len):
        u_idx = len(uv_list)
        uv_list.append([0.03 + 0.44 * float(s_param[k]), 0.025])
        row0_uv.append(u_idx)

    rows_v = [chain]
    rows_uv = [row0_uv]
    for r_pts, v_coord in ((row1_pts, 0.058), (row2_pts, 0.105)):
        cur_v = []
        cur_uv = []
        for k in range(m_len):
            vi_new = len(v_list)
            v_list.append(r_pts[k].tolist())
            r_list.append(r_sh[chain[k]].tolist())
            t_list.append(7)
            cur_v.append(vi_new)

            ui_new = len(uv_list)
            uv_list.append([0.03 + 0.44 * float(s_param[k]), v_coord])
            cur_uv.append(ui_new)
        rows_v.append(cur_v)
        rows_uv.append(cur_uv)

    for r_idx in range(2):
        rv0, rv1 = rows_v[r_idx], rows_v[r_idx + 1]
        ru0, ru1 = rows_uv[r_idx], rows_uv[r_idx + 1]
        for k in range(m_len - 1):
            f_list.append([rv0[k], rv1[k], rv1[k + 1]])
            tf_list.append([ru0[k], ru1[k], ru1[k + 1]])
            f_list.append([rv0[k], rv1[k + 1], rv0[k + 1]])
            tf_list.append([ru0[k], ru1[k + 1], ru0[k + 1]])

    print(f"[collar] 已生成挺括三维外翻古巴领 ({m_len} 段光顺边界, 新增 {2 * m_len} 顶点)")
    return (
        np.asarray(v_list, dtype=np.float32),
        np.asarray(r_list, dtype=np.float32),
        np.asarray(uv_list, dtype=np.float32),
        np.asarray(f_list, dtype=np.int32),
        np.asarray(tf_list, dtype=np.int32),
        np.asarray(t_list, dtype=np.int32),
    )


def build_shirt_shell_geom(B0: np.ndarray, rest_v: np.ndarray, UV: np.ndarray, F: np.ndarray,
                           TF: np.ndarray, head_up: np.ndarray, floor_n: np.ndarray):
    """构建零狭长片剪裁、挺括古巴翻领与宽松短袖的衬衫预垂坠壳体。"""
    rx, ry, rz = rest_v[:, 0], rest_v[:, 1], rest_v[:, 2]

    cand_v = (ry >= 109.0) & (ry <= 150.5) & (np.abs(rx) <= 34.5)
    f_idx = np.where(cand_v[F].all(axis=1))[0]

    v_cur = B0.copy()
    r_cur = rest_v.copy()
    uv_cur = UV.copy()
    f_cur = F[f_idx].copy()
    tf_cur = TF[f_idx].copy()
    tags = np.zeros(len(v_cur), dtype=np.int32)

    used_v = np.unique(f_cur)
    rmap_v = np.full(len(v_cur), -1, dtype=np.int32)
    rmap_v[used_v] = np.arange(len(used_v), dtype=np.int32)
    used_uv = np.unique(tf_cur)
    rmap_uv = np.full(len(uv_cur), -1, dtype=np.int32)
    rmap_uv[used_uv] = np.arange(len(used_uv), dtype=np.int32)
    v_cur = v_cur[used_v]
    r_cur = r_cur[used_v]
    tags = tags[used_v]
    uv_cur = uv_cur[used_uv]
    f_cur = rmap_v[f_cur]
    tf_cur = rmap_uv[tf_cur]

    # (a) 左袖口 (rx <= 29.8cm)
    v_cur, r_cur, uv_cur, f_cur, tf_cur, tags = clip_mesh_by_field(
        v_cur, r_cur, uv_cur, f_cur, tf_cur, 29.8 - r_cur[:, 0], cut_tag=2, vert_tags=tags
    )
    # (b) 右袖口 (rx >= -29.8cm)
    v_cur, r_cur, uv_cur, f_cur, tf_cur, tags = clip_mesh_by_field(
        v_cur, r_cur, uv_cur, f_cur, tf_cur, r_cur[:, 0] + 29.8, cut_tag=3, vert_tags=tags
    )
    # (c) 露脐短款宽松下摆（右前襟门襟微垂下探 3.0cm 还原原图飘逸门襟角）
    rx_c, ry_c, rz_c = r_cur[:, 0], r_cur[:, 1], r_cur[:, 2]
    w_front_placket = (1.0 / (1.0 + np.exp(-(rz_c - 3.0) / 2.0))) * np.exp(-((rx_c + 2.6) / 5.2) ** 2)
    hem_y = 117.4 - 3.0 * w_front_placket
    v_cur, r_cur, uv_cur, f_cur, tf_cur, tags = clip_mesh_by_field(
        v_cur, r_cur, uv_cur, f_cur, tf_cur, ry_c - hem_y, cut_tag=1, vert_tags=tags
    )
    # (d) 后颈弧线 + 圆润底部收口的古巴 V 领（用双曲圆角替代尖折角，消除胸骨处撕裂尖缺口）
    rx_c, ry_c, rz_c = r_cur[:, 0], r_cur[:, 1], r_cur[:, 2]
    w_front = 0.5 * (1.0 + np.tanh(rz_c / 2.4))
    y_cut_back = 145.4 + 4.2 * ((rx_c / 6.7) ** 2)
    smooth_abs_x = np.sqrt(rx_c ** 2 + 0.75 ** 2) - 0.75
    y_cut_front = 125.4 + 20.0 * ((smooth_abs_x / 6.4) ** 1.18)
    y_neck_limit = (1.0 - w_front) * y_cut_back + w_front * y_cut_front
    v_cur, r_cur, uv_cur, f_cur, tf_cur, tags = clip_mesh_by_field(
        v_cur, r_cur, uv_cur, f_cur, tf_cur, y_neck_limit - ry_c, cut_tag=4, vert_tags=tags
    )

    # 切向光顺所有裁剪边缘（袖口、下摆、领口）
    v_cur, r_cur = smooth_cut_boundaries(v_cur, r_cur, f_cur, tags, iters=10)

    # 构造宽松直筒版型与落肩宽袖筒
    b_ref = v_cur.copy()
    n_ref = compute_vertex_normals(v_cur, f_cur)
    rx_c, ry_c, rz_c = r_cur[:, 0], r_cur[:, 1], r_cur[:, 2]
    abs_x = np.abs(rx_c)

    hem_t = np.clip((134.0 - ry_c) / 17.5, 0.0, 1.0)
    sleeve_t = np.clip((abs_x - 16.5) / 13.3, 0.0, 1.0)
    neck_taper = np.clip((145.0 - ry_c) / 4.0, 0.40, 1.0)
    ease_m = (0.0085 + 0.020 * (hem_t ** 1.25) * (1.0 - sleeve_t) + 0.019 * (sleeve_t ** 1.15)) * neck_taper

    v_init = b_ref + n_ref * ease_m[:, None]
    grav_dir = -floor_n.astype(np.float32)
    v_init += grav_dir[None, :] * (0.010 * (sleeve_t ** 1.3))[:, None]

    pin_neck = (tags == 4) & (ry_c > 140.0)
    v_bridged = convex_bridge_relax(
        v_init, f_cur, b_ref, n_ref, min_clear=ease_m * 0.84, pin_mask=pin_neck, iters=22, omega=0.56
    )

    v_out, r_out, uv_out, f_out, tf_out, tags_out = extrude_cuban_collar(
        v_bridged, r_cur, uv_cur, f_cur, tf_cur, tags, head_up
    )
    return v_out, r_out, uv_out, f_out, tf_out, tags_out


def build_shorts_shell_geom(B0: np.ndarray, rest_v: np.ndarray, UV: np.ndarray, F: np.ndarray, TF: np.ndarray):
    """构建前低后高人体工学高腰、裆底完整闭合、裤管微喇齐整的水洗牛仔短裤壳体。"""
    rx, ry, rz = rest_v[:, 0], rest_v[:, 1], rest_v[:, 2]

    cand_v = (ry >= 71.0) & (ry <= 103.5) & (np.abs(rx) <= 21.5)
    f_idx = np.where(cand_v[F].all(axis=1))[0]

    v_cur = B0.copy()
    r_cur = rest_v.copy()
    uv_cur = UV.copy()
    f_cur = F[f_idx].copy()
    tf_cur = TF[f_idx].copy()
    tags = np.zeros(len(v_cur), dtype=np.int32)

    used_v = np.unique(f_cur)
    rmap_v = np.full(len(v_cur), -1, dtype=np.int32)
    rmap_v[used_v] = np.arange(len(used_v), dtype=np.int32)
    used_uv = np.unique(tf_cur)
    rmap_uv = np.full(len(uv_cur), -1, dtype=np.int32)
    rmap_uv[used_uv] = np.arange(len(used_uv), dtype=np.int32)
    v_cur = v_cur[used_v]
    r_cur = r_cur[used_v]
    tags = tags[used_v]
    uv_cur = uv_cur[used_uv]
    f_cur = rmap_v[f_cur]
    tf_cur = rmap_uv[tf_cur]

    # 1. 高腰立体裁剪：前腰 97.0cm（紧贴肚脐下沿）、两侧胯骨提拉补偿坐姿屈髋折陷 (+2.1cm)、后腰 101.4cm 饱满护腰
    rx_c, ry_c, rz_c = r_cur[:, 0], r_cur[:, 1], r_cur[:, 2]
    hip_side_lift = 2.1 * np.exp(-((np.abs(rx_c) - 11.5) / 4.8) ** 2) * (0.5 + 0.5 * np.tanh(rz_c / 4.0))
    waist_top = 99.2 - 2.2 * np.tanh(rz_c / 4.5) + hip_side_lift
    v_cur, r_cur, uv_cur, f_cur, tf_cur, tags = clip_mesh_by_field(
        v_cur, r_cur, uv_cur, f_cur, tf_cur, waist_top - ry_c, cut_tag=10, vert_tags=tags
    )

    # 2. 双腿毛边裤管柔和斜切：内侧裆底 |rx|<=5.0cm 完整保留至 ry=73.8cm（闭合无缺口），外侧平稳过渡至 78.8cm
    rx_c, ry_c, rz_c = r_cur[:, 0], r_cur[:, 1], r_cur[:, 2]
    leg_hem = 73.8 + 0.38 * np.clip(np.abs(rx_c) - 5.0, 0.0, 14.5)
    v_cur, r_cur, uv_cur, f_cur, tf_cur, tags = clip_mesh_by_field(
        v_cur, r_cur, uv_cur, f_cur, tf_cur, ry_c - leg_hem, cut_tag=11, vert_tags=tags
    )

    # 切向光顺高腰头与双腿裤管口边界
    v_cur, r_cur = smooth_cut_boundaries(v_cur, r_cur, f_cur, tags, iters=12)

    # 3. 高腰贴合 (+5.2mm) + 裤腿微喇外扩 (+18.0mm) + 全域凸桥接（含腰侧屈髋凹折区一并拉平，彻底消除胯侧挤压折痕！）
    b_ref = v_cur.copy()
    n_ref = compute_vertex_normals(v_cur, f_cur)
    rx_c, ry_c, rz_c = r_cur[:, 0], r_cur[:, 1], r_cur[:, 2]
    hip_lift_c = 2.1 * np.exp(-((np.abs(rx_c) - 11.5) / 4.8) ** 2) * (0.5 + 0.5 * np.tanh(rz_c / 4.0))
    wb_line = 99.2 - 2.2 * np.tanh(rz_c / 4.5) + hip_lift_c - 3.0

    cuff_t = np.clip((wb_line - ry_c) / 14.5, 0.0, 1.0)
    waist_band_bump = np.where(ry_c >= wb_line, 0.0014, 0.0)
    ease_dn = 0.0052 + 0.0135 * (cuff_t ** 1.15) + waist_band_bump

    v_init = b_ref + n_ref * ease_dn[:, None]
    # 仅固定最顶沿边界环做凸桥接，使两侧屈髋腹股沟深折也被硬挺牛仔布自然跨接拉平！
    pin_top_rim = tags == 10
    v_bridged = convex_bridge_relax(
        v_init, f_cur, b_ref, n_ref, min_clear=ease_dn * 0.88, pin_mask=pin_top_rim, iters=26, omega=0.60
    )
    # 再对顶沿腰口环做一次平滑圆环松弛，使整个高腰头呈完美顺滑椭圆环
    v_bridged, r_cur = smooth_cut_boundaries(v_bridged, r_cur, f_cur, tags, iters=8)
    return v_bridged, r_cur, uv_cur, f_cur, tf_cur, tags


def create_blender_sim_mesh(name: str, verts: np.ndarray, faces: np.ndarray) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def max_mesh_edge_mm(verts: np.ndarray, faces: np.ndarray) -> float:
    e0 = np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 0]], axis=1)
    e1 = np.linalg.norm(verts[faces[:, 2]] - verts[faces[:, 1]], axis=1)
    e2 = np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 2]], axis=1)
    return float(max(e0.max(), e1.max(), e2.max()) * 1000.0)


def enforce_body_clearance(v_sim: np.ndarray, f_sim: np.ndarray, b_body: np.ndarray, n_body: np.ndarray,
                           min_gap: float = 0.0024) -> tuple[np.ndarray, np.ndarray]:
    """物理解算后温和光顺与安全间隙兜底：确保任意视角下 0.0% 皮肤穿模且网格丝滑无尖刺。"""
    adj = build_vertex_adjacency(len(v_sim), f_sim)
    v_fixed = v_sim.astype(np.float64).copy()

    # 1 轮全局微幅弹性光顺滤除数值高频噪点，同时完整保留布料自然垂坠大褶
    v_sm = v_fixed.copy()
    for i in range(len(v_fixed)):
        if len(adj[i]) > 0:
            v_sm[i] = 0.82 * v_fixed[i] + 0.18 * v_fixed[adj[i]].mean(axis=0)
    v_fixed = v_sm

    kd = KDTree(len(b_body))
    for i, pt in enumerate(b_body):
        kd.insert(pt.tolist(), i)
    kd.balance()

    def query_nearest(pts_arr: np.ndarray) -> np.ndarray:
        return np.array([kd.find(pt.tolist())[1] for pt in pts_arr], dtype=np.int32)

    idx = query_nearest(v_fixed)
    closest_p = b_body[idx]
    closest_n = n_body[idx]
    signed_gap = ((v_fixed - closest_p) * closest_n).sum(axis=1)

    too_close = signed_gap < min_gap
    if np.any(too_close):
        push = (min_gap - signed_gap[too_close])[:, None] * closest_n[too_close]
        v_fixed[too_close] += push

        for _ in range(2):
            v_next = v_fixed.copy()
            for i in np.where(too_close)[0]:
                if len(adj[i]) > 0:
                    v_next[i] = 0.68 * v_fixed[i] + 0.32 * v_fixed[adj[i]].mean(axis=0)
            v_fixed = v_next
        idx2 = query_nearest(v_fixed)
        signed_gap2 = ((v_fixed - b_body[idx2]) * n_body[idx2]).sum(axis=1)
        still = signed_gap2 < min_gap * 0.95
        if np.any(still):
            v_fixed[still] += (min_gap - signed_gap2[still])[:, None] * n_body[idx2[still]]
        signed_gap = ((v_fixed - b_body[idx2]) * n_body[idx2]).sum(axis=1)

    return v_fixed.astype(np.float32), signed_gap.astype(np.float32)


def main():
    t0 = time.time()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mesh_npz = argv[0] if len(argv) > 0 else r"work\q004-base\v2\mesh_lod0_v2.npz"
    out_dir = (Path(argv[1]) if len(argv) > 1 else Path(r"work\q004-base\v3")).resolve()
    sim_frames = int(argv[2]) if len(argv) > 2 else 46
    out_dir.mkdir(parents=True, exist_ok=True)

    CL2.clear_scene()
    sc = bpy.context.scene

    d = np.load(str(Path(mesh_npz).resolve()), allow_pickle=True)
    V_sam = d["v"].astype(np.float32)
    F = d["faces"].astype(np.int32)
    UV = d["tex"].astype(np.float32)
    TF = d["tfaces"].astype(np.int32)
    rest_v = d["rest_v"].astype(np.float32)
    head_up = d["head_up"].astype(np.float32)
    B0 = CL2.sam_to_bl(V_sam)
    N0 = compute_vertex_normals(B0, F)

    # 1. 计算实拍俯拍地面法向 (20.5°)，将物理重力严格指向真实卧室地面正下方
    rx, ry, rz = rest_v[:, 0], rest_v[:, 1], rest_v[:, 2]
    pL = B0[(rx > 0) & (ry < 2.2)].mean(axis=0) - np.array([0.0, 0.0, 0.028], dtype=np.float32)
    pR = B0[(rx < 0) & (ry < 2.2)].mean(axis=0) - np.array([0.0, 0.0, 0.028], dtype=np.float32)
    hp = B0[(ry >= 79.0) & (ry <= 87.0) & (rz < 0.0) & (np.abs(rx) < 12.0)]
    pH = hp[hp[:, 2] <= np.percentile(hp[:, 2], 10)].mean(axis=0)
    floor_n = np.cross(pL - pR, pH - 0.5 * (pL + pR))
    if floor_n[2] < 0:
        floor_n = -floor_n
    floor_n /= np.linalg.norm(floor_n) + 1e-9

    sc.use_gravity = True
    sc.gravity = Vector((-9.81 * floor_n).tolist())

    # 2. 构建精简躯干+四肢碰撞代理网格
    coll_v_mask = (ry >= 42.0) & (ry <= 154.0) & (np.abs(rx) <= 46.0) & (~d["is_oral_eye"].astype(bool))
    coll_f = F[coll_v_mask[F].all(axis=1)]
    u_coll = np.unique(coll_f)
    rmap_c = np.full(len(B0), -1, dtype=np.int32)
    rmap_c[u_coll] = np.arange(len(u_coll), dtype=np.int32)
    v_coll = B0[u_coll] + N0[u_coll] * 0.0012
    f_coll = rmap_c[coll_f]

    ob_coll = create_blender_sim_mesh("collision_body", v_coll, f_coll)
    ob_coll.modifiers.new("body_col", "COLLISION")
    ob_coll.collision.thickness_outer = 0.0032
    ob_coll.collision.thickness_inner = 0.0080
    ob_coll.collision.cloth_friction = 7.5
    print(f"[sim] 碰撞代理体构建完成: {len(v_coll)} 顶点, {len(f_coll)} 三角面")

    # 3. 构建零狭长片衬衫壳体与高腰牛仔短裤壳体
    v_sh, r_sh, uv_sh, f_sh, tf_sh, tag_sh = build_shirt_shell_geom(B0, rest_v, UV, F, TF, head_up, floor_n)
    v_dn, r_dn, uv_dn, f_dn, tf_dn, tag_dn = build_shorts_shell_geom(B0, rest_v, UV, F, TF)

    # 4. 配置衬衫 Blender CLOTH 物理修改器（禁用近邻自碰撞排斥爆炸，提升解算子步精度至 12）
    ob_sh = create_blender_sim_mesh("shirt_shell_sim", v_sh, f_sh)
    vg_sh_pin = ob_sh.vertex_groups.new(name="shirt_pin")
    vg_sh_ease = ob_sh.vertex_groups.new(name="shirt_ease")

    for i in range(len(v_sh)):
        ry_i, rx_i = float(r_sh[i, 1]), abs(float(r_sh[i, 0]))
        tg = int(tag_sh[i])
        if tg == 7:
            # 熨烫挺括古巴翻领，仅微幅随重力自然贴伏
            vg_sh_pin.add([i], 0.94, "REPLACE")
            vg_sh_ease.add([i], 0.0, "REPLACE")
        elif tg == 4 and ry_i >= 127.0:
            # 领根缝合线保持圆顺不塌陷
            vg_sh_pin.add([i], 0.88, "REPLACE")
            vg_sh_ease.add([i], 0.0, "REPLACE")
        elif ry_i >= 141.0 and rx_i <= 18.5:
            w_p = float(np.clip(0.62 + 0.30 * (ry_i - 141.0) / 5.0, 0.62, 0.92))
            vg_sh_pin.add([i], w_p, "REPLACE")
            vg_sh_ease.add([i], 0.04, "REPLACE")
        elif ry_i >= 135.5 and rx_i <= 19.5:
            w_p = float(np.clip(0.50 * (ry_i - 135.5) / 5.5, 0.08, 0.50))
            vg_sh_pin.add([i], w_p, "REPLACE")
            vg_sh_ease.add([i], 0.30, "REPLACE")
        else:
            ease_w = float(np.clip(0.50 + 0.50 * max((134.0 - ry_i) / 16.0, (rx_i - 18.0) / 11.0), 0.35, 1.0))
            vg_sh_ease.add([i], ease_w, "REPLACE")

    m_sh = ob_sh.modifiers.new("cloth_shirt", "CLOTH")
    s_sh = m_sh.settings
    s_sh.quality = 12
    s_sh.mass = 0.18
    s_sh.tension_stiffness = 16.0
    s_sh.compression_stiffness = 6.0
    s_sh.shear_stiffness = 6.0
    s_sh.bending_stiffness = 0.18
    s_sh.air_damping = 1.4
    s_sh.vertex_group_mass = "shirt_pin"
    s_sh.pin_stiffness = 2.4
    s_sh.vertex_group_shrink = "shirt_ease"
    s_sh.shrink_min = -0.035
    s_sh.shrink_max = 0.0

    cs_sh = m_sh.collision_settings
    cs_sh.collision_quality = 6
    cs_sh.distance_min = 0.0034
    cs_sh.friction = 6.0
    cs_sh.use_self_collision = False

    # 5. 配置牛仔短裤 Blender CLOTH 物理修改器（12oz 挺括水洗牛仔布，腰头稳固贴腰）
    ob_dn = create_blender_sim_mesh("shorts_shell_sim", v_dn, f_dn)
    vg_dn_pin = ob_dn.vertex_groups.new(name="shorts_pin")
    vg_dn_ease = ob_dn.vertex_groups.new(name="shorts_ease")

    for i in range(len(v_dn)):
        rx_i, ry_i, rz_i = float(r_dn[i, 0]), float(r_dn[i, 1]), float(r_dn[i, 2])
        hip_lift_i = 2.1 * math.exp(-((abs(rx_i) - 11.5) / 4.8) ** 2) * (0.5 + 0.5 * math.tanh(rz_i / 4.0))
        wb_line_i = 99.2 - 2.2 * math.tanh(rz_i / 4.5) + hip_lift_i - 3.0
        if ry_i >= wb_line_i:
            vg_dn_pin.add([i], 0.98, "REPLACE")
            vg_dn_ease.add([i], 0.0, "REPLACE")
        elif ry_i >= wb_line_i - 5.5:
            w_p = float(np.clip(0.72 * (ry_i - (wb_line_i - 5.5)) / 5.5, 0.15, 0.72))
            vg_dn_pin.add([i], w_p, "REPLACE")
            vg_dn_ease.add([i], 0.20, "REPLACE")
        else:
            vg_dn_ease.add([i], 1.0, "REPLACE")

    m_dn = ob_dn.modifiers.new("cloth_shorts", "CLOTH")
    s_dn = m_dn.settings
    s_dn.quality = 12
    s_dn.mass = 0.45
    s_dn.tension_stiffness = 55.0
    s_dn.compression_stiffness = 30.0
    s_dn.shear_stiffness = 35.0
    s_dn.bending_stiffness = 2.60
    s_dn.air_damping = 1.6
    s_dn.vertex_group_mass = "shorts_pin"
    s_dn.pin_stiffness = 6.5
    s_dn.vertex_group_shrink = "shorts_ease"
    s_dn.shrink_min = -0.012
    s_dn.shrink_max = 0.0

    cs_dn = m_dn.collision_settings
    cs_dn.collision_quality = 6
    cs_dn.distance_min = 0.0034
    cs_dn.friction = 8.5
    cs_dn.use_self_collision = False

    # 6. 逐帧推进 Blender 物理引擎解算
    sc.frame_start = 1
    sc.frame_end = sim_frames
    t_sim0 = time.time()
    for frame in range(1, sim_frames + 1):
        sc.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()

    me_sh_eval = ob_sh.evaluated_get(dg).to_mesh()
    v_sh_sim = np.array([v.co[:] for v in me_sh_eval.vertices], dtype=np.float32)
    ob_sh.evaluated_get(dg).to_mesh_clear()

    me_dn_eval = ob_dn.evaluated_get(dg).to_mesh()
    v_dn_sim = np.array([v.co[:] for v in me_dn_eval.vertices], dtype=np.float32)
    ob_dn.evaluated_get(dg).to_mesh_clear()
    sim_sec = time.time() - t_sim0
    print(f"[sim] 物理布料解算完成 ({sim_frames} 帧, 耗时 {sim_sec:.2f}s)")

    # 7. 后处理安全穿模兜底并校验最大边长（杜绝任何尖刺异常顶点）
    v_sh_final, gap_sh = enforce_body_clearance(v_sh_sim, f_sh, B0, N0, min_gap=0.0024)
    v_dn_final, gap_dn = enforce_body_clearance(v_dn_sim, f_dn, B0, N0, min_gap=0.0024)

    max_e_sh = max_mesh_edge_mm(v_sh_final, f_sh)
    max_e_dn = max_mesh_edge_mm(v_dn_final, f_dn)
    print(f"[qa] 物理壳体最大边长检查: shirt={max_e_sh:.2f}mm, shorts={max_e_dn:.2f}mm")

    def bl_to_sam(b_arr: np.ndarray) -> np.ndarray:
        return np.stack([b_arr[:, 0], -b_arr[:, 2], b_arr[:, 1]], axis=1).astype(np.float32)

    out_npz = out_dir / "garments_sim_v3.npz"
    np.savez_compressed(
        out_npz,
        shirt_v_bl=v_sh_final,
        shirt_v_sam=bl_to_sam(v_sh_final),
        shirt_rest=r_sh,
        shirt_uv=uv_sh,
        shirt_f=f_sh,
        shirt_tf=tf_sh,
        shirt_tags=tag_sh,
        shirt_gap_m=gap_sh,
        shorts_v_bl=v_dn_final,
        shorts_v_sam=bl_to_sam(v_dn_final),
        shorts_rest=r_dn,
        shorts_uv=uv_dn,
        shorts_f=f_dn,
        shorts_tf=tf_dn,
        shorts_tags=tag_dn,
        shorts_gap_m=gap_dn,
        floor_normal_bl=floor_n.astype(np.float32),
    )

    stats = {
        "sim_frames": sim_frames,
        "sim_elapsed_sec": round(sim_sec, 2),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "shirt_verts": int(len(v_sh_final)),
        "shirt_faces": int(len(f_sh)),
        "shirt_max_edge_mm": round(max_e_sh, 2),
        "shirt_gap_mm_p10_p50_p90_max": [round(float(x) * 1000.0, 2) for x in np.percentile(gap_sh, [10, 50, 90, 100])],
        "shorts_verts": int(len(v_dn_final)),
        "shorts_faces": int(len(f_dn)),
        "shorts_max_edge_mm": round(max_e_dn, 2),
        "shorts_gap_mm_p10_p50_p90_max": [round(float(x) * 1000.0, 2) for x in np.percentile(gap_dn, [10, 50, 90, 100])],
    }
    (out_dir / "garments_sim_v3.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[done] 导出物理壳体 -> {out_npz}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
