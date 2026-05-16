import pandas as pd
import numpy as np

df_clean = pd.read_csv('fish_frames_200_clean_unique_no_repeat.csv')
df_enh = pd.read_csv('fish_frames_200_clean_ENHANCED_controlled.csv')

LWH = ['Length (cm)', 'Width (cm)', 'Height (cm)']
merged = pd.merge(df_clean[['FishID', 'FrameIndex'] + LWH], df_enh[['FishID', 'FrameIndex'] + LWH], on=['FishID', 'FrameIndex'], suffixes=('_raw', '_enh'))

print(f"\n{'='*80}")
print(f"  CONTROLLED A/B TEST RESULTS: ENHANCED vs RAW")
print(f"{'='*80}")
print(f"  Tested exactly {len(merged)} matching frames back-to-back.")
print(f"\n  {'Fish':<10} {'N':>4}  {'CV_L_raw':>9} {'CV_L_enh':>9}  {'CV_W_raw':>9} {'CV_W_enh':>9}  {'CV_H_raw':>9} {'CV_H_enh':>9}")
print(f"  {'-'*85}")

for fid in sorted(merged['FishID'].unique()):
    g = merged[merged['FishID'] == fid]
    n = len(g)
    row_str = f"  {fid:<10} {n:>4}"
    for col in LWH:
        raw_cv = g[col + '_raw'].std() / g[col + '_raw'].mean() * 100 if g[col + '_raw'].mean() > 0 else 0
        enh_cv = g[col + '_enh'].std() / g[col + '_enh'].mean() * 100 if g[col + '_enh'].mean() > 0 else 0
        better = 'v' if enh_cv < raw_cv else '^'
        row_str += f"  {raw_cv:>8.2f}% {enh_cv:>7.2f}%{better}"
    print(row_str)

print(f"\n  OVERALL MEDIAN CV COMPARISON:")
for col in LWH:
    raw_cvs, enh_cvs = [], []
    for fid in merged['FishID'].unique():
        g = merged[merged['FishID'] == fid]
        if g[col + '_raw'].mean() > 0: raw_cvs.append(g[col + '_raw'].std() / g[col + '_raw'].mean() * 100)
        if g[col + '_enh'].mean() > 0: enh_cvs.append(g[col + '_enh'].std() / g[col + '_enh'].mean() * 100)
    
    winner = 'ENHANCED BETTER' if np.median(enh_cvs) < np.median(raw_cvs) else 'RAW BETTER'
    print(f"    {col:<15}: RAW median={np.median(raw_cvs):>5.2f}%  |  ENH median={np.median(enh_cvs):>6.2f}%   >> {winner}")
