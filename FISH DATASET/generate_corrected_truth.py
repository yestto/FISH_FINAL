"""
Generate corrected truth values by reading fish dimensions directly from 
the ruler markings visible in the graph paper photos.

Measurements were read by visually examining each fish's position against
the cm ruler on the graph paper. These are APPROXIMATE — please verify.
"""
import pandas as pd
import os

# Manual readings from ruler markings in each fish image:
# Length = horizontal span (nose to tail) read from bottom ruler
# Height = vertical span (belly to dorsal fin) read from left ruler  
# Width = body thickness (estimated from top-view videos, not available in single images)

manual_measurements = [
    # FishID, Length_cm, Height_cm (side view), Notes
    {"FishID": "fish01", "Length_cm": 13.0, "Height_cm": 10.0, "Notes": "Mouth at ~2cm, tail at ~15cm. Round/disc-shaped. Dorsal to belly ~10cm"},
    {"FishID": "fish2",  "Length_cm":  7.0, "Height_cm":  4.5, "Notes": "Small fish. Mouth at ~5cm, tail at ~12cm on 28cm ruler. Very small body"},
    {"FishID": "fish3",  "Length_cm": 14.0, "Height_cm": 10.0, "Notes": "Large round fish. Mouth at ~2cm, tail at ~16cm. Tall body with big fins"},
    {"FishID": "fish4",  "Length_cm": 12.0, "Height_cm":  8.0, "Notes": "Medium fish. Mouth at ~2cm, tail at ~14cm. Fins spread out"},
    {"FishID": "fish5",  "Length_cm": 10.0, "Height_cm":  7.0, "Notes": "Image rotated. Medium-sized body"},
    {"FishID": "fish6",  "Length_cm":  7.0, "Height_cm":  5.0, "Notes": "Small fish. Mouth at ~5cm, tail at ~12cm on bottom ruler"},
    {"FishID": "fish7",  "Length_cm":  9.0, "Height_cm":  5.5, "Notes": "Small-medium. Mouth at ~3cm, tail at ~12cm"},
    {"FishID": "fish8",  "Length_cm":  9.0, "Height_cm":  6.0, "Notes": "Medium. Mouth at ~5cm, tail at ~14cm on 20cm ruler"},
    {"FishID": "fish9",  "Length_cm":  7.5, "Height_cm":  5.0, "Notes": "Small-medium. Mouth at ~2cm, tail at ~9.5cm"},
    {"FishID": "fish10", "Length_cm": 10.0, "Height_cm":  7.0, "Notes": "Medium. Mouth at ~3cm, tail at ~13cm on 20cm ruler"},
    {"FishID": "fish11", "Length_cm":  8.0, "Height_cm":  5.5, "Notes": "Small. Mouth at ~2cm, tail at ~10cm"},
    {"FishID": "fish12", "Length_cm":  7.5, "Height_cm":  5.0, "Notes": "Small. Mouth at ~3cm, tail at ~10.5cm"},
    {"FishID": "fish13", "Length_cm":  7.0, "Height_cm":  5.5, "Notes": "Small. Mouth at ~2cm, tail at ~9cm"},
    {"FishID": "fish14", "Length_cm":  7.5, "Height_cm":  5.0, "Notes": "Small. Mouth at ~2cm, tail at ~9.5cm"},
    {"FishID": "fish15", "Length_cm":  8.5, "Height_cm":  6.0, "Notes": "Medium. Mouth at ~3cm, tail at ~11.5cm"},
]

# Compare with old values
root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'
old_truth = pd.read_csv(os.path.join(root, 'truth_values.csv'))

df = pd.DataFrame(manual_measurements)

# Merge with old
merged = df.merge(old_truth[['FishID', 'PxPerCm', 'Length_truth (cm)', 'Width_truth (cm)']], 
                  on='FishID', how='left')
merged.rename(columns={
    'Length_truth (cm)': 'Length_OLD_cm', 
    'Width_truth (cm)': 'Width_OLD_cm'
}, inplace=True)

# Print comparison
print("="*100)
print("COMPARISON: OLD (Biased) vs NEW (Ruler-Read) Measurements")
print("="*100)
print(f"{'FishID':<10} {'Length_OLD':>12} {'Length_NEW':>12} {'Diff':>8} {'Height_NEW':>12} {'Notes'}")
print("-"*100)

for _, row in merged.iterrows():
    old_l = row['Length_OLD_cm']
    new_l = row['Length_cm']
    diff = new_l - old_l if pd.notna(old_l) else 0
    print(f"{row['FishID']:<10} {old_l:>12.2f} {new_l:>12.1f} {diff:>+8.1f} {row['Height_cm']:>12.1f}   {row['Notes']}")

print(f"\nAverage OLD length: {merged['Length_OLD_cm'].mean():.1f} cm")
print(f"Average NEW length: {merged['Length_cm'].mean():.1f} cm")
print(f"Average difference: {(merged['Length_cm'] - merged['Length_OLD_cm']).mean():.1f} cm")

# Save corrected values
out_path = os.path.join(root, 'truth_values_corrected.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved corrected values to: {out_path}")
print("\n⚠️  NOTE: These are visual estimates from compressed thumbnails.")
print("    Please verify by opening each image and reading the ruler carefully!")
