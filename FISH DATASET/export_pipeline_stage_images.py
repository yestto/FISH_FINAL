import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from build_fish_dataset import (
    build_background,
    combine_large_components,
    find_first_video,
    load_calibration,
    measure_height,
    measure_top,
    static_exclude_mask,
    blur_score,
)


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _motion_stages(frame_bgr: np.ndarray, bg_bgr: np.ndarray, exclude: np.ndarray, diff_thresh: int) -> dict[str, np.ndarray]:
    f_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    bg_blur = cv2.GaussianBlur(bg_bgr, (5, 5), 0)
    d = cv2.absdiff(f_blur, bg_blur)
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    _, m0 = cv2.threshold(g, diff_thresh, 255, cv2.THRESH_BINARY)
    m1 = cv2.bitwise_and(m0, cv2.bitwise_not(exclude)) if exclude is not None else m0.copy()
    m2 = cv2.morphologyEx(
        m1,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    m3 = cv2.morphologyEx(
        m2,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    m4 = cv2.erode(m3, None, iterations=1)
    m5 = cv2.dilate(m4, None, iterations=2)
    return {
        "frame_blur": f_blur,
        "bg_blur": bg_blur,
        "absdiff": d,
        "diff_gray": g,
        "threshold": m0,
        "exclude_mask": exclude,
        "after_exclude": m1,
        "open": m2,
        "close": m3,
        "erode": m4,
        "dilate_final": m5,
    }


def _draw_top_geometry(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return vis
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(vis, [box], True, (0, 255, 0), 2)
    return vis


def _draw_front_height(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return vis
    y0, y1 = int(ys.min()), int(ys.max())
    xmid = int(np.median(xs))
    cv2.line(vis, (xmid, y0), (xmid, y1), (255, 255, 0), 2)
    cv2.circle(vis, (xmid, y0), 4, (255, 0, 0), -1)
    cv2.circle(vis, (xmid, y1), 4, (0, 0, 255), -1)
    return vis


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    best = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for i in range(2, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_area:
            best = i
            best_area = a
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=".")
    ap.add_argument("--fish-id", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--out-dir", default="pipeline_stage_proof")
    ap.add_argument("--diff-thresh-top", type=int, default=12)
    ap.add_argument("--diff-thresh-front", type=int, default=10)
    ap.add_argument("--min-area-top", type=int, default=300)
    ap.add_argument("--min-area-front", type=int, default=250)
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    fish_dir = root / args.fish_id
    top_dir = fish_dir / "top view"
    front_dir = fish_dir / "front view"
    top_video = find_first_video(str(top_dir))
    front_video = find_first_video(str(front_dir))
    if top_video is None or front_video is None:
        raise RuntimeError(f"Missing top/front video for {args.fish_id}")

    calib = load_calibration(str(root))
    top_bg = build_background(top_video)
    front_bg = build_background(front_video)
    top_ex = static_exclude_mask(top_bg, "top")
    front_ex = static_exclude_mask(front_bg, "front")

    top_frame = _read_frame(top_video, args.frame_index)
    front_frame = _read_frame(front_video, args.frame_index)

    top_st = _motion_stages(top_frame, top_bg, top_ex, args.diff_thresh_top)
    front_st = _motion_stages(front_frame, front_bg, front_ex, args.diff_thresh_front)

    top_comp = combine_large_components(top_st["dilate_final"], min_area_px=max(1, int(args.min_area_top)))
    front_comp = combine_large_components(front_st["dilate_final"], min_area_px=max(1, int(args.min_area_front)))
    top_comp = _keep_largest_component(top_comp)
    front_comp = _keep_largest_component(front_comp)

    L, W, A, P = measure_top(top_comp, calib.cm_ground_per_px)
    H = measure_height(front_comp, calib.cm_vertical_per_px)
    bt = blur_score(top_frame)
    bf = blur_score(front_frame)
    score = (bt + bf) + 0.02 * float(cv2.countNonZero(top_comp))

    out = root / args.out_dir / args.fish_id / f"frame_{args.frame_index:06d}"
    out.mkdir(parents=True, exist_ok=True)

    # Top numbered stages
    _save(out / "01_top_raw.jpg", top_frame)
    _save(out / "02_top_background.jpg", top_bg)
    _save(out / "03_top_absdiff.jpg", top_st["absdiff"])
    _save(out / "04_top_diff_gray.jpg", _to_bgr(top_st["diff_gray"]))
    _save(out / "05_top_threshold.jpg", _to_bgr(top_st["threshold"]))
    _save(out / "06_top_exclude_mask.jpg", _to_bgr(top_st["exclude_mask"]))
    _save(out / "07_top_after_exclude.jpg", _to_bgr(top_st["after_exclude"]))
    _save(out / "08_top_open.jpg", _to_bgr(top_st["open"]))
    _save(out / "09_top_close.jpg", _to_bgr(top_st["close"]))
    _save(out / "10_top_erode.jpg", _to_bgr(top_st["erode"]))
    _save(out / "11_top_dilate.jpg", _to_bgr(top_st["dilate_final"]))
    _save(out / "12_top_components.jpg", _to_bgr(top_comp))
    _save(out / "13_top_geometry_overlay.jpg", _draw_top_geometry(top_frame, top_comp))

    # Front numbered stages
    _save(out / "21_front_raw.jpg", front_frame)
    _save(out / "22_front_background.jpg", front_bg)
    _save(out / "23_front_absdiff.jpg", front_st["absdiff"])
    _save(out / "24_front_diff_gray.jpg", _to_bgr(front_st["diff_gray"]))
    _save(out / "25_front_threshold.jpg", _to_bgr(front_st["threshold"]))
    _save(out / "26_front_exclude_mask.jpg", _to_bgr(front_st["exclude_mask"]))
    _save(out / "27_front_after_exclude.jpg", _to_bgr(front_st["after_exclude"]))
    _save(out / "28_front_open.jpg", _to_bgr(front_st["open"]))
    _save(out / "29_front_close.jpg", _to_bgr(front_st["close"]))
    _save(out / "30_front_erode.jpg", _to_bgr(front_st["erode"]))
    _save(out / "31_front_dilate.jpg", _to_bgr(front_st["dilate_final"]))
    _save(out / "32_front_components.jpg", _to_bgr(front_comp))
    _save(out / "33_front_height_overlay.jpg", _draw_front_height(front_frame, front_comp))

    # Final masks and summary
    _save(out / "40_top_final_mask.png", top_comp)
    _save(out / "41_front_final_mask.png", front_comp)

    summary = pd.DataFrame(
        [
            {
                "FishID": args.fish_id,
                "FrameIndex": int(args.frame_index),
                "cm_ground_per_px": float(calib.cm_ground_per_px),
                "cm_vertical_per_px": float(calib.cm_vertical_per_px),
                "Length (cm)": float(L),
                "Width (cm)": float(W),
                "Height (cm)": float(H),
                "Area (cm²)": float(A),
                "Perimeter (cm)": float(P),
                "TopMaskPixels": int(cv2.countNonZero(top_comp)),
                "FrontMaskPixels": int(cv2.countNonZero(front_comp)),
                "BlurTop": float(bt),
                "BlurFront": float(bf),
                "Score": float(score),
            }
        ]
    )
    summary.to_csv(out / "99_measurement_summary.csv", index=False)

    print("Saved stage proof folder:", out)
    print("Summary CSV:", out / "99_measurement_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

