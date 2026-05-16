import pandas as pd

input_file = "fish_frames_production_FINAL.csv"
output_file = "fish_frames_production_FINAL_CLEAN.csv"

df = pd.read_csv(input_file)
print(f"Before cleaning: {len(df)} rows")

# Fix 1: Remove duplicate rows
before_dupes = len(df)
df = df.drop_duplicates()
after_dupes = len(df)
print(f"\n1. Removed {before_dupes - after_dupes} duplicate rows.")
print(f"   Remaining: {len(df)} rows")

# Fix 2: Balance class distribution
# Cap each fish to a maximum number of frames (use the median count as the cap)
# This prevents the model from being biased toward fish with too many frames
fish_counts = df.groupby("FishID").size()
print(f"\n2. Frame counts per fish BEFORE balancing:")
for fid, count in fish_counts.items():
    print(f"   {fid}: {count}")

# Use 300 frames as a reasonable cap (keeps most fish intact, trims the big ones)
MAX_FRAMES = 300
balanced_dfs = []
for fid, group in df.groupby("FishID"):
    if len(group) > MAX_FRAMES:
        # Sample randomly but with a fixed seed for reproducibility
        sampled = group.sample(n=MAX_FRAMES, random_state=42)
        balanced_dfs.append(sampled)
    else:
        balanced_dfs.append(group)

df_balanced = pd.concat(balanced_dfs).sort_values(["FishID"]).reset_index(drop=True)

print(f"\n   Frame counts per fish AFTER balancing (cap={MAX_FRAMES}):")
for fid, count in df_balanced.groupby("FishID").size().items():
    print(f"   {fid}: {count}")

print(f"\n   Final dataset: {len(df_balanced)} rows, {len(df_balanced.columns)} columns")

df_balanced.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")
