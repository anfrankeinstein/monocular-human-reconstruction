"""生成优化前后同视角直观对比图集（Character 4-view Before/After + Scene 3-way Comparison）。"""
from pathlib import Path
import cv2
import numpy as np


def load_rgba_on_bg(path: Path, size: tuple[int, int], bg_bgr=(162, 162, 164)) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return np.full((size[1], size[0], 3), bg_bgr, dtype=np.uint8)
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[..., 3:4].astype(np.float32) / 255.0
        bg = np.full((size[1], size[0], 3), bg_bgr, dtype=np.float32)
        return np.clip(im[..., :3].astype(np.float32) * a + bg * (1.0 - a), 0, 255).astype(np.uint8)
    return im[..., :3]


def put_banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (12, 12), (12 + len(text) * 13 + 18, 48), (28, 28, 30), -1)
    cv2.putText(out, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def main():
    base = Path(r"work\q004-base")
    old_char = base / "renders" / "SHOE1"
    new_char = base / "renders" / "OPT_CHAR"
    old_scene = base / "renders" / "S8" / "scene_match.png"
    new_scene = base / "renders" / "OPT_SCENE" / "scene_match.png"
    new_side = base / "renders" / "OPT_SCENE" / "scene_side.png"
    ref_img = base / "v2" / "ref_1x.png"

    # 1. 角色 4 视角上下对比 (Top: Before SHOE1, Bottom: After OPT_CHAR)
    views = [
        ("hair_front.png", "Front"),
        ("hair_3q.png", "3/4 View"),
        ("hair_back.png", "Back"),
        ("hair_face.png", "Face Closeup"),
    ]
    cell = (600, 600)
    top_row, bot_row = [], []
    for fname, title in views:
        top_row.append(put_banner(load_rgba_on_bg(old_char / fname, cell), f"Before: {title}"))
        bot_row.append(put_banner(load_rgba_on_bg(new_char / fname, cell), f"After (v2): {title}"))

    char_cmp = np.vstack([np.hstack(top_row), np.hstack(bot_row)])
    out_char_cmp = new_char / "compare_before_after.png"
    cv2.imwrite(str(out_char_cmp), char_cmp)
    print(f"[sheet] wrote {out_char_cmp}")

    # 2. 场景三联对比 (Ref Photo | Old S8 | New OPT_SCENE) + 侧视接地点检验
    sc_h = 640
    sc_w = int(round(sc_h * 1402 / 1122))
    sz = (sc_w, sc_h)
    p_ref = put_banner(load_rgba_on_bg(ref_img, sz), "Reference Photo")
    p_old = put_banner(load_rgba_on_bg(old_scene, sz), "Before (S8)")
    p_new = put_banner(load_rgba_on_bg(new_scene, sz), "After (OPT_SCENE v2)")
    p_side = put_banner(load_rgba_on_bg(new_side, sz), "After (Side Contact Check)")

    scene_cmp = np.vstack([
        np.hstack([p_ref, p_old]),
        np.hstack([p_new, p_side]),
    ])
    out_sc_cmp = base / "renders" / "OPT_SCENE" / "compare_scene.png"
    cv2.imwrite(str(out_sc_cmp), scene_cmp)
    print(f"[sheet] wrote {out_sc_cmp}")


if __name__ == "__main__":
    main()
