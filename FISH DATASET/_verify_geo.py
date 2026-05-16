import pandas as pd
df = pd.read_csv('fish_frames_200_clean_unique_no_repeat.csv')

print(f"Dataset: {len(df)} rows, {df['FishID'].nunique()} fish, {len(df.columns)} columns\n")
print("ALL COLUMNS:")
for i, c in enumerate(df.columns):
    print(f"  {i+1:2d}. {c}")

print("\n--- Sample: fish01 (first 3 rows) ---")
geo_cols = [c for c in df.columns if c not in ['FPS_Top', 'FPS_Front', 'TopMaskPixels', 'FrontMaskPixels', 'BlurTop', 'BlurFront', 'Score', 'Timestamp (s)']]
g = df[df['FishID'] == 'fish01'][geo_cols].head(3)
print(g.to_string())

print("\n--- Median geometry per fish ---")
new_cols = [c for c in df.columns if c in ['Volume (cm\u00b3)', 'Surface Area (cm\u00b2)', 'Aspect Ratio', 'Elongation', 'Compactness', 'Condition Factor (K)', 'Rectangularity', 'Equivalent Diameter (cm)']]
for fish in sorted(df['FishID'].unique()):
    row = df[df['FishID'] == fish]
    parts = [f"{fish:<8}"]
    for c in new_cols:
        parts.append(f"{c}={row[c].median():.3f}")
    print("  ".join(parts))
