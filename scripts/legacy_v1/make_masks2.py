"""S6 皮肤 mask v2：按解剖分段裁剪，排除袖口/短裤/鞋，面部用凸包代替圆。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == idx).astype(np.uint8)


def seg_band(canvas, p0, p1, radius, t0=0.0, t1=1.0, cap=True):
    """沿 p0->p1 画带宽度的胶囊，只取 [t0,t1] 参数段。"""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    a = p0 + (p1 - p0) * t0
    b = p0 + (p1 - p0) * t1
    cv2.line(canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 1, int(radius) * 2)
    if cap:
        for p in (a, b):
            cv2.circle(canvas, tuple(np.round(p).astype(int)), int(radius), 1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--kpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--person", default=None, help="复用已算好的 person.png")
    ap.add_argument("--bbox", default="270,170,1160,1040")
    ap.add_argument("--arm-radius", type=int, default=34)
    ap.add_argument("--leg-radius", type=int, default=40)
    ap.add_argument("--erode-px", type=int, default=14)
    ap.add_argument("--face-dilate", type=int, default=26)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    h, w = img.shape[:2]

    if args.person and Path(args.person).exists():
        person = (cv2.imread(args.person, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    else:
        x1, y1, x2, y2 = [int(v) for v in args.bbox.split(",")]
        gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
        person = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        person = largest_component(person)

    kd = json.loads(Path(args.kpts).read_text(encoding="utf-8"))
    pts = {p["i"]: (p["x"], p["y"]) for p in kd["pose"] if p["v"] > 0.3}

    skin = np.zeros((h, w), np.uint8)
    R, L = args.arm_radius, args.leg_radius

    # 手臂：肩->肘 从 0.35 起（避开短袖袖口），肘->腕 到 0.92 止（避开掌根）
    seg_band(skin, pts[11], pts[13], R, 0.50, 1.00)
    seg_band(skin, pts[13], pts[15], R, 0.00, 0.66)
    seg_band(skin, pts[12], pts[14], R, 0.50, 1.00)
    seg_band(skin, pts[14], pts[16], R, 0.00, 0.66)

    # 腿：髋->膝 到 0.86 止（避开短裤），膝->踝 0.04~0.90（避开鞋帮）
    seg_band(skin, pts[23], pts[25], L, 0.00, 0.86)
    seg_band(skin, pts[24], pts[26], L, 0.00, 0.86)
    seg_band(skin, pts[25], pts[27], L, 0.04, 0.52)
    seg_band(skin, pts[26], pts[28], L, 0.04, 0.52)

    # 面部：两耳 + 鼻 + 嘴 的凸包外扩
    face_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fpts = np.array([pts[i] for i in face_idx if i in pts], np.int32)
    if len(fpts) >= 3:
        hull = cv2.convexHull(fpts)
        fm = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(fm, hull, 255)
        fm = cv2.dilate(fm, np.ones((args.face_dilate * 2 + 1,) * 2, np.uint8))
        skin |= (fm > 0).astype(np.uint8)

    # 手：21 点小圆
    for hd in kd.get("hands", []):
        for p in hd["points"]:
            cv2.circle(skin, (int(p["x"]), int(p["y"])), 16, 1, -1)

    skin = skin & person
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = largest_component(skin) if False else skin
    print(f"[skin] 覆盖 {skin.mean():.2%}, 占人物 {skin.sum()/max(person.sum(),1):.1%}")

    skin_e = cv2.erode(skin, np.ones((9, 9), np.uint8))
    person_e = cv2.erode(person, np.ones((args.erode_px * 2 + 1,) * 2, np.uint8))

    cv2.imwrite(str(outdir / "person.png"), person * 255)
    cv2.imwrite(str(outdir / "person_eroded.png"), person_e * 255)
    cv2.imwrite(str(outdir / "skin.png"), skin * 255)
    cv2.imwrite(str(outdir / "skin_eroded.png"), skin_e * 255)

    vis = img.copy()
    vis[person > 0] = (vis[person > 0] * 0.6 + np.array([80, 170, 255]) * 0.4).astype(np.uint8)
    vis[skin > 0] = (vis[skin > 0] * 0.35 + np.array([50, 50, 235]) * 0.65).astype(np.uint8)
    cv2.imwrite(str(outdir / "mask_vis.png"), vis)

    stats = {"person_ratio": float(person.mean()), "skin_ratio": float(skin.mean()),
             "skin_of_person": float(skin.sum() / max(person.sum(), 1))}
    (outdir / "mask_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("[out]", stats)


if __name__ == "__main__":
    main()
