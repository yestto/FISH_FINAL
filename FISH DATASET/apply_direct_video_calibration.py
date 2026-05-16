import pandas as pd
import os

root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'

# Load the original ENHANCED frames (with the old bad calibration)
frames_path = os.path.join(root, 'fish_frames_200_ENHANCED_clean_unique_no_repeat.csv')
frames_df = pd.read_csv(frames_path)

# The user's requested true physical lengths for each fish.
# We will force the median extracted video length to match these exactly,
# which effectively calibrates the video's camera distance!
target_lengths = {
    "fish01": 12.5,
    "fish2":  8.5,
    "fish3":  13.0,  # "12.96 is fine for this fish 3"
    "fish4":  11.5,
    "fish5":  10.5,
    "fish6":  7.0,
    "fish7":  7.5,
    "fish8":  7.5,
    "fish9":  6.5,
    "fish10": 7.5,
    "fish11": 6.0,
    "fish12": 6.0,
    "fish13": 6.0,
    "fish14": 5.0,   # "fish 5cm lenght approx"
    "fish15": 6.0,   # "fish 15 is 6 cm approx"
}

# Calculate the current median length for each fish in the raw video data
medians = frames_df.groupby('FishID')['Length (cm)'].median()

ratios = {}
for fish, target in target_lengths.items():
    if fish in medians:
        current_median = medians[fish]
        ratios[fish] = target / current_median

print("Calibration ratios to force video lengths to match true physical lengths:")
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
print(f"\nSaved directly calibrated frames to: {out_path}")
