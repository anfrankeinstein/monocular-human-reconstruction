"""v3 独立服装壳体物理模拟 + 6.4 万根写实 3D 曲线毛发系统一键流水线入口。

流水线阶段：
  G1: 独立衬衫壳体（含 3D 外翻古巴领）+ 闭裆牛仔短裤壳体裁剪与 Blender CLOTH 物理布料模拟
  G2: 三通道解耦 4K 贴图构建（纯皮肤 skin_body_v3 + 衬衫壳体 shirt_atlas_v3 + 牛仔短裤壳体 shorts_atlas_v3）
  G3: 角色 9 视角渲染（5 标准机位 + 前/后服装特写 + 侧前/后颈毛发微距特写 -> OPT_CHAR_V3）
  G4: 卧室实景复刻渲染（OPT_SCENE_V3）
  G5: v2 vs v3 全身、服装微距、毛发微距及场景对比图集生成

用法（在 Q:\\stb-assets 下执行）：
  Q:\\stb-assets\\envs\\mhr\\Scripts\\python.exe scripts\\v3_run_all.py
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_MHR = ROOT / "envs" / "mhr" / "Scripts" / "python.exe"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")


def run_step(title: str, cmd: list[str]):
    print(f"\n==================== {title} ====================")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    if res.returncode != 0:
        raise SystemExit(f"[FAIL] {title} 退出码 {res.returncode}")
    print(f"[OK] {title} ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-frames", type=int, default=46)
    ap.add_argument("--atlas-size", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--char-res", type=int, default=1000)
    ap.add_argument("--scene-res-y", type=int, default=840)
    ap.add_argument("--skip-sim", action="store_true", help="复用已有 garments_sim_v3.npz")
    ap.add_argument("--skip-tex", action="store_true", help="复用已有皮肤与服装 4K 贴图")
    ap.add_argument("--skip-shoe-tex", action="store_true", help="复用已有帆布鞋与白袜 PBR 贴图")
    args = ap.parse_args()

    t_all = time.time()
    if not args.skip_sim:
        run_step("G1: 独立服装壳体裁剪与 Blender CLOTH 物理模拟", [
            str(BLENDER), "--python-exit-code", "1", "-b", "-P", "scripts/v3_simulate_garments.py", "--",
            "work/q004-base/v2/mesh_lod0_v2.npz",
            "work/q004-base/v3",
            str(args.sim_frames),
        ])

    if not args.skip_tex:
        run_step("G2: 三通道解耦 4K 贴图构建 (纯肤色 + 衬衫壳 + 牛仔短裤壳)", [
            str(PY_MHR), "scripts/v3_build_garment_tex.py", "--size", str(args.atlas_size)
        ])

    if not args.skip_shoe_tex:
        run_step("G2b: 帆布鞋与白袜独立非重叠 UV 校验及 4K/2K 四通道 PBR 贴图构建", [
            str(PY_MHR), "scripts/v3_build_shoe_tex.py", "--shoe-size", str(args.atlas_size)
        ])

    run_step("G3: 角色多视角与微距特写渲染 (OPT_CHAR_V3)", [
        str(BLENDER), "--python-exit-code", "1", "-b", "-P", "scripts/v3_render_char.py", "--",
        "work/q004-base/v2/mesh_lod0_v2.npz",
        "work/q004-base/v3/garments_sim_v3.npz",
        "work/q004-base/v3/tex",
        "work/q004-base/renders/OPT_CHAR_V3",
        str(args.samples),
        str(args.char_res),
    ])

    run_step("G4: 卧室实景复刻渲染 (OPT_SCENE_V3)", [
        str(BLENDER), "--python-exit-code", "1", "-b", "-P", "scripts/v3_build_scene.py", "--",
        "work/q004-base/v2/mesh_lod0_v2.npz",
        "work/q004-base/v3/garments_sim_v3.npz",
        "work/q004-base/v3/tex",
        "work/q004-base/scene/tex",
        "work/q004-base/renders/OPT_SCENE_V3",
        str(args.scene_res_y),
        str(args.samples),
    ])

    run_step("G5: 生成 v2 vs v3 同视角与微距对比图集", [
        str(PY_MHR), "scripts/v3_make_sheets.py"
    ])
    print(f"\n[ALL DONE] v3 服装独立壳体与布料模拟全流程执行完毕，总耗时 {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
