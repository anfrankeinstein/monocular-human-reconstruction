"""生成比对实验所需 mask：人物轮廓（最大连通域）+ 裸露皮肤（按关键点几何区域）。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEG_MODEL = r"Q:\stb-assets\models\selfie_segmenter.tflite"

POSE_MODEL = r"Q:\stb-assets\models\pose_landmarker.task"


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == idx).astype(np.uint8)


def draw_limb(canvas, p0, p1, radius):
    cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, radius * 2)
    for p in (p0, p1):
        cv2.circle(canvas, (int(p[0]), int(p[1])), radius, 1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kpts", default=None, help="mediapipe kpts.json（缺省则现算）")
    ap.add_argument("--erode-px", type=int, default=14)
    ap.add_argument("--bbox", default="270,170,1160,1040", help="grabcut 初始框 x1,y1,x2,y2")
    ap.add_argument("--arm-radius", type=int, default=34)
    ap.add_argument("--leg-radius", type=int, default=40)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    print(f"[in] {w}x{h}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    x1, y1, x2, y2 = [int(v) for v in args.bbox.split(",")]
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    person = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    person = largest_component(person)
    print(f"[person] 覆盖 {person.mean():.2%}")

    if args.kpts and Path(args.kpts).exists():
        pose = json.loads(Path(args.kpts).read_text(encoding="utf-8"))["pose"]
        pts = {p["i"]: (p["x"], p["y"]) for p in pose if p["v"] > 0.3}
    else:
        popts = vision.PoseLandmarkerOptions(
            base_options=mp_py.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_poses=1)
        with vision.PoseLandmarker.create_from_options(popts) as lm:
            res = lm.detect(mp_img)
        pts = {i: (p.x * w, p.y * h) for i, p in enumerate(res.pose_landmarks[0])}

    skin = np.zeros((h, w), np.uint8)
    arms = [(11, 13), (13, 15), (12, 14), (14, 16)]
    legs = [(25, 27), (26, 28)]
    for a, b in arms:
        if a in pts and b in pts:
            draw_limb(skin, pts[a], pts[b], args.arm_radius)
    for a, b in legs:
        if a in pts and b in pts:
            draw_limb(skin, pts[a], pts[b], args.leg_radius)
    if 0 in pts:
        cv2.circle(skin, (int(pts[0][0]), int(pts[0][1])), 70, 1, -1)

    skin = skin & person
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    print(f"[skin] 覆盖 {skin.mean():.2%}, 占人物 {skin.sum()/max(person.sum(),1):.1%}")

    eroded = cv2.erode(person, np.ones((args.erode_px * 2 + 1,) * 2, np.uint8))

    cv2.imwrite(str(outdir / "person.png"), person * 255)
    cv2.imwrite(str(outdir / "skin.png"), skin * 255)
    cv2.imwrite(str(outdir / "eroded.png"), eroded * 255)

    vis = img.copy()
    vis[person > 0] = (vis[person > 0] * 0.6 + np.array([80, 170, 255]) * 0.4).astype(np.uint8)
    vis[skin > 0] = (vis[skin > 0] * 0.35 + np.array([50, 50, 235]) * 0.65).astype(np.uint8)
    cv2.imwrite(str(outdir / "mask_vis.png"), vis)

    stats = {
        "person_ratio": float(person.mean()),
        "skin_ratio": float(skin.mean()),
        "skin_of_person": float(skin.sum() / max(person.sum(), 1)),
        "erode_px": args.erode_px,
    }
    (outdir / "mask_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[out] {stats}")


if __name__ == "__main__":
    main()
