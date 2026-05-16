"""
CONTROLLED ENHANCEMENT TEST
============================
Takes the EXACT frames from fish_frames_200_clean_unique_no_repeat.csv,
goes back to the original videos, re-extracts those SAME frames with
enhancement techniques applied, and compares the measurements.

This is the scientifically correct A/B test:
  - Same frames, same fish
  - Only difference: enhanced vs raw pixel processing
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_fish_dataset import (
    Calibration,
    ExtractConfig,
    build_background,
    blur_score,
    combine_large_components,
    load_calibration,
    load_weights_map,
    measure_height,
    measure_top,
    passes_border_margin,
    static_exclude_mask,
    find_first_video,
    iter_fish_folders,
    is_single_fish_id,
)

from image_enhancements import (
    enhance_frame,
    motion_mask_enhanced,
    visibility_score,
)


def main():
    dataset_root = os.path.dirname(os.path.abspath(__file__))
    
    # ── Load the PROVEN clean dataset ──
    clean_csv = os.path.join(dataset_root, "fish_frames_200_clean_unique_no_repeat.csv")
    df_clean = pd.read_csv(clean_csv)
    print(f"Loaded {len(df_clean)} proven frames from clean dataset")
    
    # Build a lookup: {fish_id: set of frame indices}
    target_frames = {}
    for _, row in df_clean.iterrows():
        fid = str(row["FishID"]).strip()
        fidx = int(row["FrameIndex"])
        if fid not in target_frames:
            target_frames[fid] = set()
        target_frames[fid].add(fidx)
    
    print(f"Target: {sum(len(v) for v in target_frames.values())} frames across {len(target_frames)} fish\n")
    
    calib = load_calibration(dataset_root)
    
    cfg = ExtractConfig(
        stride=1,          # We check every frame since we need exact indices
        max_frames=99999,
        diff_thresh_top=15,
        diff_thresh_front=12,
        min_area_top=200,
        min_area_front=150,
        border_margin=8,
        min_blur=1.0,       # Very lenient - we already know these frames are good
        min_aspect=1.05,
    )
    
    folders = list(iter_fish_folders(dataset_root))
    all_rows = []
    
    for folder in tqdm(folders, desc="Re-extracting with enhancement"):
        fish_id = os.path.basename(folder)
        if not is_single_fish_id(fish_id):
            continue
        if fish_id not in target_frames:
            continue
        
        wanted = target_frames[fish_id]
        
        top_dir = os.path.join(folder, "top view")
        front_dir = os.path.join(folder, "front view")
        top_video = find_first_video(top_dir) if os.path.isdir(top_dir) else None
        front_video = find_first_video(front_dir) if os.path.isdir(front_dir) else None
        
        if not top_video or not front_video:
            print(f"  [SKIP] {fish_id}: missing video(s)")
            continue
        
        # Build backgrounds
        top_bg = build_background(top_video)
        front_bg = build_background(front_video)
        top_ex = static_exclude_mask(top_bg, "top")
        front_ex = static_exclude_mask(front_bg, "front")
        
        top_cap = cv2.VideoCapture(top_video)
        front_cap = cv2.VideoCapture(front_video)
        top_n = int(top_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        front_n = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(top_n, front_n)
        
        fps_top = float(top_cap.get(cv2.CAP_PROP_FPS))
        fps_front = float(front_cap.get(cv2.CAP_PROP_FPS))
        fps = fps_top if np.isfinite(fps_top) and fps_top > 0 else 20.0
        
        found = 0
        i = 0
        while i < n:
            ok_t, t_raw = top_cap.read()
            ok_f, f_raw = front_cap.read()
            if not ok_t or not ok_f or t_raw is None or f_raw is None:
                break
            
            if i not in wanted:
                i += 1
                continue
            
            # ── ENHANCED PATH ──
            t_enhanced = enhance_frame(t_raw, low_visibility=False, apply_gamma=True)
            f_enhanced = enhance_frame(f_raw, low_visibility=False, apply_gamma=True)
            
            # Enhanced motion mask (median blur + morphology)
            top_m = motion_mask_enhanced(t_raw, top_bg, top_ex, diff_thresh=cfg.diff_thresh_top)
            front_m = motion_mask_enhanced(f_raw, front_bg, front_ex, diff_thresh=cfg.diff_thresh_front)
            top_m = combine_large_components(top_m, min_area_px=cfg.min_area_top)
            front_m = combine_large_components(front_m, min_area_px=cfg.min_area_front)
            
            top_px = int(cv2.countNonZero(top_m))
            front_px = int(cv2.countNonZero(front_m))
            
            if top_px < 50 or front_px < 50:
                i += 1
                continue
            
            L, W, A, P = measure_top(top_m, calib.cm_ground_per_px)
            H = measure_height(front_m, calib.cm_vertical_per_px)
            
            if not all(np.isfinite(v) and v > 0 for v in [L, W, H, A, P]):
                i += 1
                continue
            
            bt = blur_score(t_enhanced)
            bf = blur_score(f_enhanced)
            score = (bt + bf) + 0.02 * float(top_px)
            
            all_rows.append({
                "FishID": fish_id,
                "FrameIndex": int(i),
                "Timestamp (s)": float(i / fps),
                "FPS_Top": float(fps_top),
                "FPS_Front": float(fps_front),
                "Length (cm)": float(L),
                "Width (cm)": float(W),
                "Height (cm)": float(H),
                "Area (cm²)": float(A),
                "Perimeter (cm)": float(P),
                "TopMaskPixels": int(top_px),
                "FrontMaskPixels": int(front_px),
                "BlurTop": float(bt),
                "BlurFront": float(bf),
                "Score": float(score),
            })
            found += 1
            i += 1
        
        top_cap.release()
        front_cap.release()
        print(f"  {fish_id}: re-extracted {found}/{len(wanted)} frames with enhancement")
    
    # Save
    df_enh = pd.DataFrame(all_rows)
    out_path = os.path.join(dataset_root, "fish_frames_200_clean_ENHANCED_controlled.csv")
    df_enh.to_csv(out_path, index=False)
    
    # ── HEAD-TO-HEAD COMPARISON ──
    LWH = ["Length (cm)", "Width (cm)", "Height (cm)"]
    
    print(f"\n{'='*65}")
    print(f"  CONTROLLED A/B TEST RESULTS")
    print(f"{'='*65}")
    print(f"  Enhanced re-extraction: {len(df_enh)} frames recovered")
    
    # Merge on FishID + FrameIndex to compare same frames
    merged = pd.merge(
        df_clean[["FishID", "FrameIndex"] + LWH],
        df_enh[["FishID", "FrameIndex"] + LWH],
        on=["FishID", "FrameIndex"],
        suffixes=("_raw", "_enh"),
        how="inner"
    )
    print(f"  Matched frames (both have data): {len(merged)}")
    
    print(f"\n  {'Fish':<10} {'N':>4}  {'CV_L_raw':>9} {'CV_L_enh':>9}  {'CV_W_raw':>9} {'CV_W_enh':>9}  {'CV_H_raw':>9} {'CV_H_enh':>9}")
    print(f"  {'-'*80}")
    
    for fid in sorted(merged['FishID'].unique()):
        g = merged[merged['FishID'] == fid]
        n = len(g)
        row_str = f"  {fid:<10} {n:>4}"
        for col in LWH:
            raw_cv = g[col + "_raw"].std() / g[col + "_raw"].mean() * 100 if g[col + "_raw"].mean() > 0 else 0
            enh_cv = g[col + "_enh"].std() / g[col + "_enh"].mean() * 100 if g[col + "_enh"].mean() > 0 else 0
            better = "↓" if enh_cv < raw_cv else "↑"
            row_str += f"  {raw_cv:>8.2f}% {enh_cv:>7.2f}%{better}"
        print(row_str)
    
    # Overall
    print(f"\n  OVERALL MEDIAN CV COMPARISON:")
    for col in LWH:
        raw_cvs = []
        enh_cvs = []
        for fid in merged['FishID'].unique():
            g = merged[merged['FishID'] == fid]
            if g[col + "_raw"].mean() > 0:
                raw_cvs.append(g[col + "_raw"].std() / g[col + "_raw"].mean() * 100)
            if g[col + "_enh"].mean() > 0:
                enh_cvs.append(g[col + "_enh"].std() / g[col + "_enh"].mean() * 100)
        print(f"    {col:<15}: RAW median={np.median(raw_cvs):.2f}%  ENH median={np.median(enh_cvs):.2f}%  {'ENHANCED WINS ✓' if np.median(enh_cvs) < np.median(raw_cvs) else 'RAW WINS ✓'}")
    
    print(f"\n  Output saved: {out_path}")


if __name__ == "__main__":
    main()
