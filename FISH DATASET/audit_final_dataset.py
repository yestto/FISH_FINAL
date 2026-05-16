import pandas as pd
import numpy as np

df = pd.read_csv("fish_frames_production_FINAL.csv")

print("=" * 65)
print("COMPREHENSIVE PREPROCESSING AUDIT")
print("=" * 65)

# 1. Basic Info
print(f"\n1. BASIC INFO")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Columns: {list(df.columns)}")

# 2. Missing Values
print(f"\n2. MISSING VALUES")
nulls = df.isnull().sum()
if nulls.sum() == 0:
    print("   PASS: No missing values found.")
else:
    print("   FAIL: Missing values detected!")
    print(nulls[nulls > 0])

# 3. Duplicate Rows
print(f"\n3. DUPLICATE ROWS")
dupes = df.duplicated().sum()
print(f"   Duplicates: {dupes}")

# 4. Infinite Values
print(f"\n4. INFINITE VALUES")
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_count = np.isinf(df[numeric_cols]).sum().sum()
if inf_count == 0:
    print("   PASS: No infinite values.")
else:
    print(f"   FAIL: {inf_count} infinite values detected!")

# 5. Physical Sanity Check
print(f"\n5. PHYSICAL SANITY CHECKS")
bad_height = (df["Height (cm)"] >= df["Length (cm)"]).sum()
print(f"   Height >= Length: {bad_height} rows (should be 0)")
bad_width = (df["Width (cm)"] >= df["Length (cm)"]).sum()
print(f"   Width >= Length: {bad_width} rows (should be 0)")
negative = (df[numeric_cols] < 0).any(axis=1).sum()
print(f"   Rows with negative values: {negative}")

# 6. Per-Fish Distribution
print(f"\n6. PER-FISH FRAME COUNT")
for fid, g in df.groupby("FishID"):
    print(f"   {fid}: {len(g)} frames | Weight={g['Weight (g)'].iloc[0]}g")

# 7. Feature Statistics
print(f"\n7. FEATURE STATISTICS (min/mean/max)")
for col in numeric_cols:
    if col == "Weight (g)":
        continue
    print(f"   {col:30s}: min={df[col].min():8.2f}  mean={df[col].mean():8.2f}  max={df[col].max():8.2f}")

# 8. Correlation with Weight (how useful each feature is)
print(f"\n8. CORRELATION WITH WEIGHT (higher = more useful)")
for col in numeric_cols:
    if col == "Weight (g)":
        continue
    corr = df[col].corr(df["Weight (g)"])
    bar = "+" * int(abs(corr) * 20)
    print(f"   {col:30s}: {corr:+.3f} {bar}")
