import pandas as pd
import numpy as np
import os

root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'

# Load the enhanced frames csv
frames_path = os.path.join(root, 'fish_frames_200_ENHANCED_clean_unique_no_repeat.csv')
frames_df = pd.read_csv(frames_path)

# Load the corrected truth values
truth_path = os.path.join(root, 'truth_values_corrected.csv')
truth_df = pd.read_csv(truth_path)

# We need the old truth values to compute the ratio, or we can just use the manual lengths directly?
# Wait, the regression uses the extracted video dimensions.
# The video dimensions (Length (cm) etc) were calculated as: Length_px / PxPerCm
# Since the OLD PxPerCm was biased, the extracted video Length (cm) is also biased.
# The extracted video dimensions = pixel_dimension / old_px_per_cm
# We should correct them by multiplying by (old_px_per_cm / new_px_per_cm)
# BUT wait! We don't have new_px_per_cm in truth_values_corrected.csv, we just have NEW Length.
# old_length = old_length_px / old_px_per_cm
# new_length = old_length_px / new_px_per_cm
# So ratio = new_length / old_length

old_truth = pd.read_csv(os.path.join(root, 'truth_values.csv'))

# Create a mapping of FishID to length scaling ratio
ratios = {}
for _, row in truth_df.iterrows():
    fish = row['FishID']
    new_len = row['Length_cm']
    old_row = old_truth[old_truth['FishID'] == fish]
    if not old_row.empty:
        old_len = old_row['Length_truth (cm)'].values[0]
        ratios[fish] = new_len / old_len

print("Scaling ratios per fish:")
for k, v in ratios.items():
    print(f"  {k}: {v:.3f}")

# Apply corrections
corrected_df = frames_df.copy()

for fish, ratio in ratios.items():
    mask = corrected_df['FishID'] == fish
    
    # Linear dimensions
    for col in ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Perimeter (cm)', 'Equivalent Diameter (cm)']:
        if col in corrected_df.columns:
            corrected_df.loc[mask, col] *= ratio
            
    # Area dimensions
    for col in ['Area (cm²)', 'Surface Area (cm²)']:
        if col in corrected_df.columns:
            corrected_df.loc[mask, col] *= (ratio ** 2)
            
    # Volume dimensions
    if 'Volume (cm³)' in corrected_df.columns:
        corrected_df.loc[mask, 'Volume (cm³)'] *= (ratio ** 3)

out_path = os.path.join(root, 'fish_frames_corrected.csv')
corrected_df.to_csv(out_path, index=False)
print(f"\nSaved corrected frames to: {out_path}")
