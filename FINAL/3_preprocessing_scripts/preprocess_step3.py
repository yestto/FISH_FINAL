import pandas as pd
from sklearn.preprocessing import StandardScaler

input_file = "fish_frames_production_step2_features.csv"
output_file = "fish_frames_production_FINAL.csv"

df = pd.read_csv(input_file)

# We want to scale all the physical features, but we MUST NOT scale:
# 1. FishID (it's a string/label)
# 2. Weight (g) (we want the model to predict actual grams, not scaled decimals)
features_to_scale = [
    "Length (cm)", "Width (cm)", "Height (cm)", "Area (cm\u00b2)", 
    "Perimeter (cm)", "Volume (cm\u00b3)", "Surface Area (cm\u00b2)", 
    "Aspect Ratio", "Elongation", "Compactness", 
    "Condition Factor (K)", "Rectangularity", "Equivalent Diameter (cm)"
]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print(f"Step 3: Feature Scaling")
print(f" - Applied StandardScaler to 13 biometric features.")
print(f" - Mean of Length (cm) before scaling: {df['Length (cm)'].mean():.2f}")
print(f" - Mean of Length (cm) after scaling:  {df_scaled['Length (cm)'].mean():.2f} (Expected: ~0.0)")

df_scaled.to_csv(output_file, index=False)
print(f"\nSaved FINAL preprocessed dataset to: {output_file}")
