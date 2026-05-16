import pandas as pd

input_file = "fish_frames_production.csv"
output_file = "fish_frames_production_step1_filtered.csv"

# Load the raw production dataset
df = pd.read_csv(input_file)
original_count = len(df)
print(f"Original dataset rows: {original_count}")

# 1. Physical Constraint: Height cannot be greater than or equal to Length
# A fish's height is typically 20-50% of its length. Let's allow up to 80% to be safe.
df_filtered = df[df["Height (cm)"] < (df["Length (cm)"] * 0.8)].copy()

# 2. Aspect Ratio Constraint: (Length / Width)
# Fish are generally long and thin from the top. 
# Typical aspect ratio is between 2.5 and 7.0
df_filtered = df_filtered[(df_filtered["Aspect Ratio"] > 2.0) & (df_filtered["Aspect Ratio"] < 8.0)]

# 3. YOLO Confidence Score
# Only keep high-confidence detections
df_filtered = df_filtered[df_filtered["Score"] > 0.80]

# 4. Remove negative or zero dimensions (just in case)
df_filtered = df_filtered[(df_filtered["Length (cm)"] > 0) & 
                          (df_filtered["Width (cm)"] > 0) & 
                          (df_filtered["Height (cm)"] > 0)]

final_count = len(df_filtered)
dropped_count = original_count - final_count

print(f"\nApplied Physical Filters:")
print(f" - Removed {dropped_count} frames with perspective distortion or poor detections.")
print(f" - Remaining clean frames: {final_count} ({(final_count/original_count)*100:.1f}%)")

# Save the filtered dataset
df_filtered.to_csv(output_file, index=False)
print(f"\nSaved Step 1 dataset to: {output_file}")
