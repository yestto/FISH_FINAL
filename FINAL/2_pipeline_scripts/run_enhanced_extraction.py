"""
ENHANCED FISH DATASET EXTRACTION
=================================
Runs the same video extraction pipeline as build_fish_dataset.py
but with image enhancement techniques applied to every frame
BEFORE mask extraction.

Outputs a SEPARATE CSV: fish_frame_measurements_enhanced.csv
Nothing existing is overwritten.

Usage:
    python run_enhanced_extraction.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from the original pipeline (reuse calibration, weights, etc.)
from build_fish_dataset import (
    Calibration,
    ExtractConfig,
    build_background,
    blur_score,
    combine_large_components,
    compute_weight,
    iter_fish_folders,
    is_single_fish_id,
    load_calibration,
    load_weights_map,
    mask_to_points,
    measure_height,
    measure_top,
    passes_border_margin,
    static_exclude_mask,
    find_first_video,
)

# Import our enhancement techniques
from image_enhancements import (
    auto_gamma_correction,
    apply_clahe,
    enhance_frame,
    motion_mask_enhanced,
    visibility_score,
)


def process_pair_enhanced(
    top_video: str,
    front_video: str,
    calib: Calibration,
    cfg: ExtractConfig,
    fish_id: str,
    assumed_fps: float,
) -> list[dict]:
    """
    Enhanced version of process_pair_per_frame.
    Applies the full image enhancement pipeline to every frame
    before extracting motion masks and measurements.
    """
    top_cap = cv2.VideoCapture(top_video)
    front_cap = cv2.VideoCapture(front_video)

    top_n = int(top_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    front_n = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(top_n, front_n)
    if n <= 0:
        top_cap.release()
        front_cap.release()
        return []

    fps_top = float(top_cap.get(cv2.CAP_PROP_FPS))
    fps_front = float(front_cap.get(cv2.CAP_PROP_FPS))
    fps = fps_top
    if not np.isfinite(fps) or fps <= 0:
        fps = fps_front
    if not np.isfinite(fps) or fps <= 0:
        fps = float(assumed_fps)

    # Build backgrounds (same as baseline)
    top_bg = build_background(top_video)
    front_bg = build_background(front_video)

    # ENHANCEMENT: Apply SAME enhancement to background for consistent motion detection
    top_bg_enhanced = enhance_frame(top_bg, low_visibility=False, apply_gamma=False)
    front_bg_enhanced = enhance_frame(front_bg, low_visibility=False, apply_gamma=False)

    top_ex = static_exclude_mask(top_bg, "top")
    front_ex = static_exclude_mask(front_bg, "front")

    out: list[dict] = []
    i = 0

    while i < n:
        ok_t, t_raw = top_cap.read()
        ok_f, f_raw = front_cap.read()
        if not ok_t or not ok_f or t_raw is None or f_raw is None:
            break

        if (i % cfg.stride) != 0:
            i += 1
            continue

        # ── ENHANCEMENT PIPELINE START ──
        # Compute visibility scores on RAW frames (before enhancement)
        vis_top_raw = visibility_score(t_raw)
        vis_front_raw = visibility_score(f_raw)

        # Apply full enhancement pipeline (Gamma + CLAHE + Denoise) FOR VISUALS
        t_enhanced = enhance_frame(t_raw, low_visibility=False, apply_gamma=True)
        f_enhanced = enhance_frame(f_raw, low_visibility=False, apply_gamma=True)

        # Compute visibility scores AFTER enhancement (to prove improvement)
        vis_top_enhanced = visibility_score(t_enhanced)
        vis_front_enhanced = visibility_score(f_enhanced)
        # ── ENHANCEMENT PIPELINE END ──

        # Use ENHANCED motion mask (median filter + larger morphology) on RAW pixels to preserve subtraction
        top_m = motion_mask_enhanced(t_raw, top_bg, top_ex, diff_thresh=cfg.diff_thresh_top)
        front_m = motion_mask_enhanced(f_raw, front_bg, front_ex, diff_thresh=cfg.diff_thresh_front)

        top_m = combine_large_components(top_m, min_area_px=cfg.min_area_top)
        front_m = combine_large_components(front_m, min_area_px=cfg.min_area_front)

        top_px = int(cv2.countNonZero(top_m))
        front_px = int(cv2.countNonZero(front_m))

        if top_px < cfg.min_area_top or front_px < cfg.min_area_front:
            i += 1
            continue
        if not passes_border_margin(top_m, margin_px=cfg.border_margin) or not passes_border_margin(
            front_m, margin_px=cfg.border_margin
        ):
            i += 1
            continue

        L, W, A, P = measure_top(top_m, calib.cm_ground_per_px)
        H = measure_height(front_m, calib.cm_vertical_per_px)
        if not (np.isfinite(L) and np.isfinite(W) and np.isfinite(A) and np.isfinite(P) and np.isfinite(H)):
            i += 1
            continue

        if L <= 0 or W <= 0 or H <= 0:
            i += 1
            continue
        if L < W:
            i += 1
            continue
        if (L / max(W, 1e-6)) < cfg.min_aspect:
            i += 1
            continue

        # Blur score on ENHANCED frames (should be higher = sharper)
        bt = blur_score(t_enhanced)
        bf = blur_score(f_enhanced)
        if bt < cfg.min_blur or bf < cfg.min_blur:
            i += 1
            continue

        score = (bt + bf) + 0.02 * float(top_px)
        out.append(
            {
                "FishID": fish_id,
                "FrameIndex": int(i),
                "Timestamp (s)": float(i / fps) if np.isfinite(fps) and fps > 0 else float("nan"),
                "FPS_Top": float(fps_top) if np.isfinite(fps_top) else float("nan"),
                "FPS_Front": float(fps_front) if np.isfinite(fps_front) else float("nan"),
                "Length (cm)": float(L),
                "Width (cm)": float(W),
                "Height (cm)": float(H),
                "Area (cm\u00b2)": float(A),
                "Perimeter (cm)": float(P),
                "TopMaskPixels": int(top_px),
                "FrontMaskPixels": int(front_px),
                "BlurTop": float(bt),
                "BlurFront": float(bf),
                "Score": float(score),
                # NEW: Visibility metrics for publication
                "VisScore_Top_Raw": float(vis_top_raw),
                "VisScore_Front_Raw": float(vis_front_raw),
                "VisScore_Top_Enhanced": float(vis_top_enhanced),
                "VisScore_Front_Enhanced": float(vis_front_enhanced),
            }
        )
        i += 1

    top_cap.release()
    front_cap.release()
    return out


def main() -> int:
    dataset_root = os.path.dirname(os.path.abspath(__file__))
    weights_csv = os.path.join(dataset_root, "weights.csv")
    out_csv = os.path.join(dataset_root, "fish_frame_measurements_enhanced.csv")

    print("=" * 70)
    print("  ENHANCED FISH EXTRACTION PIPELINE")
    print("  Techniques: Gamma + CLAHE + NLM Denoise + Bilateral + Retinex")
    print("=" * 70)

    calib = load_calibration(dataset_root)
    weights = load_weights_map(weights_csv) if os.path.exists(weights_csv) else {}

    cfg = ExtractConfig(
        stride=5,
        max_frames=9999,
        diff_thresh_top=8,
        diff_thresh_front=6,
        min_area_top=200,
        min_area_front=150,
        border_margin=8,
        min_blur=5.0,
        min_aspect=1.05,
    )

    folders = iter_fish_folders(dataset_root)
    all_rows: list[dict] = []

    for folder in tqdm(folders, desc="Enhanced extraction"):
        fish_id = os.path.basename(folder)
        if not is_single_fish_id(fish_id):
            continue

        top_dir = os.path.join(folder, "top view")
        front_dir = os.path.join(folder, "front view")

        top_video = find_first_video(top_dir) if os.path.isdir(top_dir) else None
        front_video = find_first_video(front_dir) if os.path.isdir(front_dir) else None

        if not top_video or not front_video:
            print(f"  [SKIP] {fish_id}: missing video(s)")
            continue

        weight = compute_weight(weights, fish_id)

        rows = process_pair_enhanced(
            top_video=top_video,
            front_video=front_video,
            calib=calib,
            cfg=cfg,
            fish_id=fish_id,
            assumed_fps=60.0,
        )

        # Add weight column
        for r in rows:
            r["Weight (g)"] = float(weight) if np.isfinite(weight) else float("nan")

        all_rows.extend(rows)
        print(f"  {fish_id}: {len(rows)} enhanced frames extracted")

        if all_rows:
            # INCREMENTAL SAVE
            df_temp = pd.DataFrame(all_rows)
            base_cols = [
                "FishID", "Weight (g)", "FrameIndex", "Timestamp (s)",
                "FPS_Top", "FPS_Front",
                "Length (cm)", "Width (cm)", "Height (cm)",
                "Area (cm²)", "Perimeter (cm)",
                "TopMaskPixels", "FrontMaskPixels",
                "BlurTop", "BlurFront", "Score",
                "VisScore_Top_Raw", "VisScore_Front_Raw",
                "VisScore_Top_Enhanced", "VisScore_Front_Enhanced",
            ]
            existing = [c for c in base_cols if c in df_temp.columns]
            df_temp = df_temp[existing].sort_values(["FishID", "FrameIndex"]).reset_index(drop=True)
            df_temp.to_csv(out_csv, index=False)

    df = pd.DataFrame(all_rows)

    # Reorder columns to match baseline format + new visibility columns at end
    base_cols = [
        "FishID", "Weight (g)", "FrameIndex", "Timestamp (s)",
        "FPS_Top", "FPS_Front",
        "Length (cm)", "Width (cm)", "Height (cm)",
        "Area (cm\u00b2)", "Perimeter (cm)",
        "TopMaskPixels", "FrontMaskPixels",
        "BlurTop", "BlurFront", "Score",
        "VisScore_Top_Raw", "VisScore_Front_Raw",
        "VisScore_Top_Enhanced", "VisScore_Front_Enhanced",
    ]
    existing = [c for c in base_cols if c in df.columns]
    df = df[existing].sort_values(["FishID", "FrameIndex"]).reset_index(drop=True)

    df.to_csv(out_csv, index=False)

    print(f"\n{'=' * 70}")
    print(f"  OUTPUT: {out_csv}")
    print(f"  Total rows: {len(df)}")
    print(f"  Fish: {df['FishID'].nunique()}")
    print(f"{'=' * 70}")

    # Quick visibility improvement summary
    if "VisScore_Top_Raw" in df.columns:
        print("\n  VISIBILITY IMPROVEMENT SUMMARY:")
        print(f"  {'Fish':<8} {'TopRaw':>8} {'TopEnh':>8} {'FrontRaw':>9} {'FrontEnh':>9}")
        for fish in sorted(df["FishID"].unique()):
            g = df[df["FishID"] == fish]
            print(
                f"  {fish:<8} "
                f"{g['VisScore_Top_Raw'].median():8.1f} "
                f"{g['VisScore_Top_Enhanced'].median():8.1f} "
                f"{g['VisScore_Front_Raw'].median():9.1f} "
                f"{g['VisScore_Front_Enhanced'].median():9.1f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
