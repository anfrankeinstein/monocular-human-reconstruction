"""S0 预处理：提取身体 33 点与双手 21 点（mediapipe tasks API）。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

POSE_MODEL = r"Q:\stb-assets\models\pose_landmarker.task"
HAND_MODEL = r"Q:\stb-assets\models\hand_landmarker.task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"无法读取图片: {args.image}")
    h, w = img.shape[:2]
    print(f"[in] {w}x{h}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision

    print(f"[mp] {mp.__version__}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = {"image": str(args.image), "size": [w, h], "pose": None, "hands": []}

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    with vision.PoseLandmarker.create_from_options(pose_opts) as lm:
        res = lm.detect(mp_img)
        if res.pose_landmarks:
            pts = [{"i": i, "x": p.x * w, "y": p.y * h, "v": getattr(p, "visibility", 1.0)}
                   for i, p in enumerate(res.pose_landmarks[0])]
            result["pose"] = pts
            print(f"[pose] {len(pts)} points, min vis {min(p['v'] for p in pts):.2f}")
        else:
            print("[pose] NO DETECTION")

    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    with vision.HandLandmarker.create_from_options(hand_opts) as lm:
        res = lm.detect(mp_img)
        if res.hand_landmarks:
            for lms, handed in zip(res.hand_landmarks, res.handedness or []):
                label = handed[0].category_name if handed else "?"
                score = handed[0].score if handed else 0.0
                pts = [{"i": i, "x": p.x * w, "y": p.y * h} for i, p in enumerate(lms)]
                result["hands"].append({"label": label, "score": float(score), "points": pts})
                print(f"[hand] {label} score={score:.2f} points={len(pts)}")
        else:
            print("[hands] NO DETECTION")

    (outdir / "kpts.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    vis = img.copy()
    if result["pose"]:
        for p in result["pose"]:
            if p["v"] > 0.3:
                cv2.circle(vis, (int(p["x"]), int(p["y"])), 3, (0, 255, 0), -1)
    for hand in result["hands"]:
        for p in hand["points"]:
            cv2.circle(vis, (int(p["x"]), int(p["y"])), 3, (0, 0, 255), -1)
    cv2.imwrite(str(outdir / "kpts_vis.png"), vis)
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()
