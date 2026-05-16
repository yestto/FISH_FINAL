import pandas as pd
import numpy as np

# Load our final production dataset
df = pd.read_csv('fish_frames_200_clean_unique_no_repeat.csv')

# Calculate per-fish medians from our visual bounding boxes
vision_stats = {}
for fish in sorted(df['FishID'].unique()):
    g = df[df['FishID'] == fish]
    vision_stats[fish] = {
        'v_len': g['Length (cm)'].median(),
        'v_wid': g['Width (cm)'].median(),
        'v_hgt': g['Height (cm)'].median()
    }

# Load truth values
truth_file = 'fish_truth_measurements.csv'
try:
    tdf = pd.read_csv(truth_file)
    l_col = 'Length_truth (cm)'
    w_col = 'Width_truth (cm)'
    # Note: no height truth in this dataset apparently? 
    # Let's check
    h_col = 'Height_truth (cm)' if 'Height_truth (cm)' in tdf.columns else None
    
    print("\n--- COMPARISON TO TRUTH ---")
    print(f"{'Fish':8s} | {'LENGTH (cm)':^14s} | {'WIDTH (cm)':^14s} |")
    print(f"{'':8s} | {'Vision':>6s} vs {'Truth':>4s} | {'Vision':>6s} vs {'Truth':>4s} |")
    print("-" * 45)
    
    diffs_l = []
    diffs_w = []
    
    for _, r in tdf.drop_duplicates(subset=['FishID']).iterrows():
        fish = str(r['FishID']).strip()
        if fish in vision_stats:
            vs = vision_stats[fish]
            tl = float(r[l_col])
            tw = float(r[w_col])
            
            # Absolute differences
            diff_l = abs(vs['v_len'] - tl)
            diff_w = abs(vs['v_wid'] - tw)
            
            print(f"{fish:8s} | {vs['v_len']:6.1f} vs {tl:4.1f} | {vs['v_wid']:6.1f} vs {tw:4.1f} |")
            
            if not np.isnan(diff_l): diffs_l.append(diff_l)
            if not np.isnan(diff_w): diffs_w.append(diff_w)
            
    print("-" * 45)
    print(f"Mean Abs Error : Length = {np.mean(diffs_l):.2f} cm")
    print(f"Mean Abs Error : Width  = {np.mean(diffs_w):.2f} cm")

except Exception as e:
    print(f"Error loading truth: {e}")
