"""生成 v2（体表法向膨胀）与 v3（独立 3D 壳建模 + 物理布料模拟）同视角对比图集。"""
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


def crop_and_resize_rgba(path: Path, box_rel: tuple[float, float, float, float],
                         size: tuple[int, int], bg_bgr=(162, 162, 164)) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return np.full((size[1], size[0], 3), bg_bgr, dtype=np.uint8)
    h, w = im.shape[:2]
    x0, y0, x1, y1 = (
        int(round(box_rel[0] * w)),
        int(round(box_rel[1] * h)),
        int(round(box_rel[2] * w)),
        int(round(box_rel[3] * h)),
    )
    crop = im[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    if crop.ndim == 3 and crop.shape[2] == 4:
        a = crop[..., 3:4].astype(np.float32) / 255.0
        bg = np.full((size[1], size[0], 3), bg_bgr, dtype=np.float32)
        return np.clip(crop[..., :3].astype(np.float32) * a + bg * (1.0 - a), 0, 255).astype(np.uint8)
    return crop[..., :3]


def put_banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (12, 12), (12 + len(text) * 12 + 20, 48), (28, 28, 30), -1)
    cv2.putText(out, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def main():
    base = Path(r"work\q004-base")
    char_v2 = base / "renders" / "OPT_CHAR"
    char_v3 = base / "renders" / "OPT_CHAR_V3"
    scene_v2 = base / "renders" / "OPT_SCENE" / "scene_match.png"
    scene_v3 = base / "renders" / "OPT_SCENE_V3" / "scene_match.png"
    side_v3 = base / "renders" / "OPT_SCENE_V3" / "scene_side.png"
    ref_img = base / "v2" / "ref_1x.png"

    # 1. 角色 4 视角上下对比 (Top: v2 Body Vertex Inflate, Bottom: v3 Standalone Shell + Cloth Sim)
    views = [
        ("hair_front.png", "Front"),
        ("hair_3q.png", "3/4 View"),
        ("hair_side.png", "Side"),
        ("hair_back.png", "Back"),
    ]
    cell = (620, 620)
    top_row, bot_row = [], []
    for fname, title in views:
        top_row.append(put_banner(load_rgba_on_bg(char_v2 / fname, cell), f"v2 (Body Inflate): {title}"))
        bot_row.append(put_banner(load_rgba_on_bg(char_v3 / fname, cell), f"v3 (Shell + Cloth Sim): {title}"))

    char_cmp = np.vstack([np.hstack(top_row), np.hstack(bot_row)])
    out_char_cmp = char_v3 / "compare_v2_vs_v3.png"
    cv2.imwrite(str(out_char_cmp), char_cmp)
    print(f"[sheet_v3] wrote {out_char_cmp}")

    # 2. 服装局部微距细节四联对比 (Ref Crop | v2 3/4 Crop | v3 3/4 Garment Closeup | v3 Back Garment Closeup)
    dcell = (650, 650)
    p_ref_crop = put_banner(crop_and_resize_rgba(ref_img, (0.28, 0.24, 0.72, 0.79), dcell), "Ref Photo (Garments)")
    p_v2_crop = put_banner(crop_and_resize_rgba(char_v2 / "hair_3q.png", (0.18, 0.24, 0.82, 0.88), dcell), "v2 Body Inflate (3/4)")
    p_v3_front = put_banner(load_rgba_on_bg(char_v3 / "garment_closeup.png", dcell), "v3 Cloth Shell (3/4 Closeup)")
    p_v3_back = put_banner(load_rgba_on_bg(char_v3 / "garment_back_closeup.png", dcell), "v3 Cloth Shell (Back Closeup)")

    detail_cmp = np.hstack([p_ref_crop, p_v2_crop, p_v3_front, p_v3_back])
    out_det_cmp = char_v3 / "compare_garment_detail_v3.png"
    cv2.imwrite(str(out_det_cmp), detail_cmp)
    print(f"[sheet_v3] wrote {out_det_cmp}")

    # 2b. 毛发真实度四联微距对比 (Ref Head | v2 Ribbon/Sphere Hair | v3 3D Strand Hair Front | v3 3D Strand Hair Back)
    hcell = (650, 650)
    h_ref = put_banner(crop_and_resize_rgba(base / "scene" / "gpen_2x.png", (0.37, 0.10, 0.63, 0.42), hcell), "Ref Photo (Head & Hair)")
    h_v2 = put_banner(load_rgba_on_bg(char_v2 / "hair_face.png", hcell), "v2 (Cap + Ribbon + Sphere)")
    h_v3_f = put_banner(load_rgba_on_bg(char_v3 / "hair_face.png", hcell), "v3 (64k 3D Strands + Hair BSDF)")
    h_v3_b = put_banner(load_rgba_on_bg(char_v3 / "hair_back_closeup.png", hcell), "v3 (Updo Flow & Nape Baby Hairs)")
    hair_cmp = np.hstack([h_ref, h_v2, h_v3_f, h_v3_b])
    out_hair_cmp = char_v3 / "compare_hair_v2_vs_v3.png"
    cv2.imwrite(str(out_hair_cmp), hair_cmp)
    print(f"[sheet_v3] wrote {out_hair_cmp}")

    # 2c. 帆布鞋与白袜独立 PBR 四联微距对比 (Ref Shoe | v2 Cylinder Decal | v3 PBR Medial Star & Eyelets | v3 PBR Heel Plate)
    scell = (650, 650)
    sh_ref = put_banner(crop_and_resize_rgba(base / "scene" / "gpen_2x.png", (0.14, 0.62, 0.41, 0.96), scell), "Ref Photo (Converse & Sock)")
    sh_v2 = put_banner(crop_and_resize_rgba(char_v2 / "hair_front.png", (0.06, 0.65, 0.33, 0.95), scell), "v2 (24 Objs + Cylinder Decal)")
    sh_v3_m = put_banner(load_rgba_on_bg(char_v3 / "shoe_medial_closeup.png", scell), "v3 PBR (Star Patch & Eyelets)")
    sh_v3_h = put_banner(load_rgba_on_bg(char_v3 / "shoe_heel_closeup.png", scell), "v3 PBR (ALL*STAR Heel Plate)")
    shoe_cmp = np.hstack([sh_ref, sh_v2, sh_v3_m, sh_v3_h])
    out_shoe_cmp = char_v3 / "compare_shoe_v2_vs_v3.png"
    cv2.imwrite(str(out_shoe_cmp), shoe_cmp)
    print(f"[sheet_v3] wrote {out_shoe_cmp}")

    # 3. 卧室场景对比 (Ref Photo | v2 OPT_SCENE | v3 OPT_SCENE_V3 | v3 Side Contact Check)
    sc_h = 640
    sc_w = int(round(sc_h * 1402 / 1122))
    sz = (sc_w, sc_h)
    s_ref = put_banner(load_rgba_on_bg(ref_img, sz), "Reference Photo")
    s_v2 = put_banner(load_rgba_on_bg(scene_v2, sz), "v2 Scene (Body Inflate)")
    s_v3 = put_banner(load_rgba_on_bg(scene_v3, sz), "v3 Scene (Shell + Cloth Sim)")
    s_side = put_banner(load_rgba_on_bg(side_v3, sz), "v3 Side Contact & Drape Check")

    scene_cmp = np.vstack([
        np.hstack([s_ref, s_v2]),
        np.hstack([s_v3, s_side]),
    ])
    out_sc_cmp = base / "renders" / "OPT_SCENE_V3" / "compare_scene_v3.png"
    cv2.imwrite(str(out_sc_cmp), scene_cmp)
    print(f"[sheet_v3] wrote {out_sc_cmp}")


if __name__ == "__main__":
    main()
