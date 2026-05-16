import pandas as pd

input_file = "fish_frames_production_step1_filtered.csv"
output_file = "fish_frames_production_step2_features.csv"

df = pd.read_csv(input_file)

# Select only the features we need for regression + the FishID (for splitting) and Weight (target)
columns_to_keep = [
    "FishID", 
    "Weight (g)", 
    "Length (cm)", 
    "Width (cm)", 
    "Height (cm)", 
    "Area (cm\u00b2)", 
    "Perimeter (cm)", 
    "Volume (cm\u00b3)", 
    "Surface Area (cm\u00b2)", 
    "Aspect Ratio", 
    "Elongation", 
    "Compactness", 
    "Condition Factor (K)", 
    "Rectangularity", 
    "Equivalent Diameter (cm)"
]

df_features = df[columns_to_keep].copy()

print(f"Step 2: Feature Selection")
print(f" - Reduced columns from {len(df.columns)} to {len(df_features.columns)}")
print(" - Kept purely biometric features + Target Weight.")

df_features.to_csv(output_file, index=False)
print(f"Saved Step 2 dataset to: {output_file}")
