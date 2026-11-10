"""v2 端到端一键流水线入口（M1 分割与原生 LOD0 -> M2 Z-Buffer 图集合成 -> M3 角色渲染 -> M4 场景渲染 -> 对比图集）。

用法（在 Q:\\stb-assets 下执行）：
  Q:\\stb-assets\\envs\\mhr\\Scripts\\python.exe scripts\\v2_run_all.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_MHR = ROOT / "envs" / "mhr" / "Scripts" / "python.exe"
PY_MHR0 = ROOT / "envs" / "mhr0" / "python.exe"
MHR0_BIN = ROOT / "envs" / "mhr0" / "Library" / "bin"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")


def run_step(title: str, cmd: list[str], extra_path: Path | None = None):
    print(f"\n==================== {title} ====================")
    env = os.environ.copy()
    if extra_path is not None:
        env["PATH"] = f"{extra_path};{env.get('PATH', '')}"
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if res.returncode != 0:
        raise SystemExit(f"[FAIL] {title} 退出码 {res.returncode}")
    print(f"[OK] {title} ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--char-res", type=int, default=1000)
    ap.add_argument("--scene-res-y", type=int, default=840)
    ap.add_argument("--atlas-size", type=int, default=4096)
    ap.add_argument("--skip-mesh", action="store_true", help="复用已有 mesh_lod0_v2.npz")
    args = ap.parse_args()

    t_all = time.time()
    if not args.skip_mesh:
        run_step("M1-A: 前景分割与可微体态优化", [str(PY_MHR), "scripts/v2_segment_and_fit.py"])
        run_step("M1-B: 原生 MHR LOD0 与解剖语义生成", [str(PY_MHR0), "scripts/v2_prepare_mesh.py"], extra_path=MHR0_BIN)

    run_step("M2: 屏幕空间 Z-Buffer 反投影与角色图集合成", [
        str(PY_MHR), "scripts/v2_build_atlas.py", "--size", str(args.atlas_size)
    ])

    run_step("M3: 角色 5 视角渲染 (OPT_CHAR)", [
        str(BLENDER), "--python-exit-code", "1", "-b", "-P", "scripts/v2_render_char.py", "--",
        "work/q004-base/v2/mesh_lod0_v2.npz",
        "work/q004-base/v2/tex/char_atlas_v2.png",
        "work/q004-base/renders/OPT_CHAR",
        str(args.samples), str(args.char_res),
    ])

    run_step("M4: 俯拍斜面对齐场景渲染 (OPT_SCENE)", [
        str(BLENDER), "--python-exit-code", "1", "-b", "-P", "scripts/v2_build_scene.py", "--",
        "work/q004-base/v2/mesh_lod0_v2.npz",
        "work/q004-base/v2/tex/char_atlas_v2.png",
        "work/q004-base/scene/tex",
        "work/q004-base/renders/OPT_SCENE",
        str(args.scene_res_y), str(args.samples),
    ])

    run_step("M5: 生成优化前后对比图集", [str(PY_MHR), "scripts/v2_make_sheets.py"])
    print(f"\n[ALL DONE] 全流程执行完毕，总耗时 {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
