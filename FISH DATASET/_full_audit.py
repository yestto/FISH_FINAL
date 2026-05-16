import pandas as pd
import numpy as np

df = pd.read_csv('fish_frames_200_clean_unique_no_repeat.csv')
wdf = pd.read_csv('weights.csv')
weights = dict(zip(wdf['FishID'].astype(str), wdf['Weight']))

print("=" * 80)
print("CURRENT DATASET AUDIT")
print("=" * 80)

print(f"\nTotal rows: {len(df)}")
print(f"Total fish: {df['FishID'].nunique()}")
print(f"Fish IDs: {sorted(df['FishID'].unique())}")
print(f"\nColumns: {df.columns.tolist()}")

print("\n--- Per-Fish Summary ---")
print(f"{'Fish':<8} {'Wt(g)':>6} {'Rows':>5} {'medL':>6} {'medW':>6} {'medH':>6} {'L/W':>5} {'CV_L%':>6} {'CV_W%':>6}")
print("-" * 65)

for fish in sorted(df['FishID'].unique()):
    g = df[df['FishID'] == fish]
    wt = weights.get(fish, 0)
    ml = g['Length (cm)'].median()
    mw = g['Width (cm)'].median()
    mh = g['Height (cm)'].median()
    lw = ml / mw if mw > 0 else 0
    cv_l = (g['Length (cm)'].std() / g['Length (cm)'].mean()) * 100
    cv_w = (g['Width (cm)'].std() / g['Width (cm)'].mean()) * 100
    print(f"{fish:<8} {wt:6.1f} {len(g):5d} {ml:6.2f} {mw:6.2f} {mh:6.2f} {lw:5.2f} {cv_l:6.2f} {cv_w:6.2f}")

print("\n--- Missing Fish ---")
all_fish = set(wdf['FishID'].astype(str))
present = set(df['FishID'].unique())
missing = all_fish - present
print(f"Missing: {sorted(missing) if missing else 'None'}")

print("\n--- Duplicate Check ---")
dupes = df.duplicated(subset=['FishID', 'FrameIndex']).sum()
print(f"Exact FishID+FrameIndex duplicates: {dupes}")

print("\n--- Value Range Check ---")
for col in ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']:
    neg = (df[col] < 0).sum()
    zero = (df[col] == 0).sum()
    print(f"  {col}: min={df[col].min():.3f} max={df[col].max():.3f} negatives={neg} zeros={zero}")
