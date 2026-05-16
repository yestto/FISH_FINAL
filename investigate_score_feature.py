"""
INVESTIGATE SCORE FEATURE - POTENTIAL CONTAMINATION
Check if Score feature is causing the high accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("="*80)
print("🔍 INVESTIGATING SCORE FEATURE - POTENTIAL CONTAMINATION")
print("="*80)

# Load the clean dataset
df = pd.read_csv('fish_frames_CLEAN.csv')

# Check Score feature correlation with weight
score_weight_corr = df['Score'].corr(df['Weight (g)'])
print(f"Score vs Weight correlation: {score_weight_corr:.6f}")

# Check Score distribution
print(f"\nScore statistics:")
print(f"  Range: {df['Score'].min():.6f} - {df['Score'].max():.6f}")
print(f"  Mean: {df['Score'].mean():.6f} ± {df['Score'].std():.6f}")
print(f"  Unique values: {df['Score'].nunique()}")

# Check if Score is actually a detection confidence or something else
print(f"\nScore value counts (top 10):")
print(df['Score'].value_counts().head(10))

# Test model WITHOUT Score feature
legitimate_features_no_score = [
    'Length (cm)',
    'Width (cm)', 
    'Height (cm)',
    'Area (cm²)',
    'Perimeter (cm)',
    'BlurTop',      # Image quality
    'BlurFront'     # Image quality
]

# Remove missing values
df_clean = df.dropna(subset=legitimate_features_no_score + ['Weight (g)'])

X_no_score = df_clean[legitimate_features_no_score]
y = df_clean['Weight (g)']

print(f"\nTesting model WITHOUT Score feature:")
print(f"Features: {len(legitimate_features_no_score)}")
for feature in legitimate_features_no_score:
    print(f"  ✅ {feature}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X_no_score, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
pred = rf.predict(X_test_scaled)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"\n📊 RESULTS WITHOUT SCORE FEATURE:")
print(f"  MAE: {mae:.3f} g")
print(f"  R²: {r2:.3f}")
print(f"  Accuracy: {r2*100:.1f}%")

# Feature importance
feature_importance = rf.feature_importances_
print(f"\n🔍 FEATURE IMPORTANCE (without Score):")
for name, importance in zip(legitimate_features_no_score, feature_importance):
    print(f"{name:20s}: {importance:.3f}")

print(f"\n" + "="*80)
print("🎯 CONCLUSION:")
if score_weight_corr > 0.8:
    print("❌ Score feature is HIGHLY CORRELATED with weight - CONTAMINATED!")
elif score_weight_corr > 0.5:
    print("⚠️  Score feature is moderately correlated with weight - investigate further")
else:
    print("✅ Score feature appears legitimate")

print(f"Model without Score: {r2*100:.1f}% accuracy - this is your legitimate result!")