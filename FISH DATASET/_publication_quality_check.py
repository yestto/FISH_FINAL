"""
HEAD-TO-HEAD: Which dataset is publication-quality?
  A: fish_frames_200_clean_unique_no_repeat.csv        (OLD baseline)
  B: fish_frames_200_ENHANCED_clean_unique_no_repeat.csv (NEW enhanced)
"""
import pandas as pd
import numpy as np

A = pd.read_csv("fish_frames_200_clean_unique_no_repeat.csv")
B = pd.read_csv("fish_frames_200_ENHANCED_clean_unique_no_repeat.csv")

LWH = ["Length (cm)", "Width (cm)", "Height (cm)"]

def report(df, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total rows           : {len(df)}")
    print(f"  Unique fish           : {df['FishID'].nunique()}")
    fish_list = sorted(df['FishID'].unique())
    print(f"  Fish IDs              : {', '.join(fish_list)}")
    
    # Frames per fish
    fpc = df.groupby('FishID').size()
    print(f"  Frames/fish (min)     : {fpc.min()}")
    print(f"  Frames/fish (max)     : {fpc.max()}")
    print(f"  Frames/fish (median)  : {fpc.median():.0f}")
    
    # Per-fish CV
    print(f"\n  {'Fish':<10} {'N':>5} {'CV_L%':>8} {'CV_W%':>8} {'CV_H%':>8} {'MaxCV%':>8}  {'Pass?':>6}")
    print(f"  {'-'*55}")
    all_pass = True
    fish_cvs = []
    for fid in sorted(df['FishID'].unique()):
        g = df[df['FishID'] == fid]
        n = len(g)
        cvs = []
        for col in LWH:
            m = g[col].mean()
            cv = (g[col].std() / m * 100) if m > 0 else 0
            cvs.append(cv)
        worst = max(cvs)
        passed = worst < 15.0
        if not passed:
            all_pass = False
        fish_cvs.append(worst)
        mark = "  ✓" if passed else "  ✗ FAIL"
        print(f"  {fid:<10} {n:>5} {cvs[0]:>8.2f} {cvs[1]:>8.2f} {cvs[2]:>8.2f} {worst:>8.2f}{mark}")
    
    median_cv = np.median(fish_cvs)
    print(f"\n  Median worst-CV       : {median_cv:.2f}%")
    print(f"  All fish CV < 15%?    : {'YES ✓' if all_pass else 'NO ✗'}")
    
    # Duplicate frame check
    dup = df.duplicated(subset=['FishID', 'FrameIndex']).sum()
    print(f"  Duplicate frames      : {dup}")
    
    # Weight coverage
    if 'Weight (g)' in df.columns:
        w = df.groupby('FishID')['Weight (g)'].first()
        print(f"  Weight range          : {w.min():.1f}g - {w.max():.1f}g")
    
    return len(df), df['FishID'].nunique(), median_cv, all_pass

print("\n" + "▓"*60)
print("  PUBLICATION QUALITY HEAD-TO-HEAD COMPARISON")
print("▓"*60)

nA, fishA, cvA, passA = report(A, "Dataset A: fish_frames_200_clean_unique_no_repeat.csv (OLD)")
nB, fishB, cvB, passB = report(B, "Dataset B: fish_frames_200_ENHANCED_clean_unique_no_repeat.csv (NEW)")

print("\n" + "="*60)
print("  FINAL VERDICT")
print("="*60)
print(f"  {'Metric':<25} {'OLD':>12} {'NEW':>12} {'Winner':>10}")
print(f"  {'-'*60}")
print(f"  {'Total Rows':<25} {nA:>12} {nB:>12} {'OLD' if nA > nB else 'NEW':>10}")
print(f"  {'Fish Coverage':<25} {fishA:>12} {fishB:>12} {'OLD' if fishA > fishB else 'NEW':>10}")
print(f"  {'Median Worst CV%':<25} {cvA:>12.2f} {cvB:>12.2f} {'OLD' if cvA < cvB else 'NEW':>10}")
print(f"  {'All CV < 15%?':<25} {str(passA):>12} {str(passB):>12} {'OLD' if passA else 'NEW' if passB else 'NEITHER':>10}")
