"""
VERIFICATION SCRIPT - CHECKING FOR DATA LEAKAGE AND MANIPULATION
This script provides complete transparency about the 99.98% accuracy results
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*80)
print("🔍 TRANSPARENCY ANALYSIS - CHECKING FOR DATA LEAKAGE")
print("="*80)

# Load and analyze the dataset
print("Loading fish_frames.csv...")
df = pd.read_csv('fish_frames.csv')

print(f"Original dataset: {len(df)} samples, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Check for potential data leakage
print("\n🔍 CHECKING FOR DATA LEAKAGE INDICATORS:")
print("-" * 50)

# Use only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
print(f"Numeric columns: {len(numeric_df.columns)} out of {len(df.columns)} total")

# Check correlations with weight
if 'Weight (g)' in numeric_df.columns:
    correlation_with_weight = numeric_df.corr()['Weight (g)'].abs().sort_values(ascending=False)
    print("\nCorrelation with Weight (g):")
    for col, corr in correlation_with_weight.items():
        if col != 'Weight (g)':
            print(f"  {col}: {corr:.3f}")
    
    # Check for columns that might be derived from weight
    print("\n⚠️  HIGH CORRELATION WARNING (>0.95):")
    high_corr_cols = correlation_with_weight[correlation_with_weight > 0.95].index.tolist()
    if 'Weight (g)' in high_corr_cols:
        high_corr_cols.remove('Weight (g)')
    
    if high_corr_cols:
        print("🚨 CRITICAL: Found columns highly correlated with weight:")
        for col in high_corr_cols:
            print(f"  - {col}: {correlation_with_weight[col]:.3f}")
        print("❌ These columns likely contain weight-derived features = DATA LEAKAGE")
    else:
        print("✅ No extremely high correlations found")
else:
    print("❌ Weight (g) column not found in numeric data")

# Check for columns that might contain weight information
print("\n🔍 CHECKING FOR SUSPICIOUS COLUMN NAMES:")
suspicious_patterns = ['truth', 'error', 'scale', 'weight', 'composite', 'selection']
for col in df.columns:
    col_lower = col.lower()
    for pattern in suspicious_patterns:
        if pattern in col_lower:
            print(f"⚠️  Suspicious column name: {col}")

# Check for duplicate or near-duplicate rows
print("\n🔍 CHECKING FOR DUPLICATES:")
duplicates = df.duplicated().sum()
print(f"Exact duplicates: {duplicates}")

# Now run the model with full transparency
print("\n" + "="*80)
print("🧪 RUNNING MODEL WITH TRANSPARENCY")
print("="*80)

# Use only numeric columns (excluding string columns like FishID)
numeric_cols = numeric_df.columns.tolist()
if 'Weight (g)' in numeric_cols:
    numeric_cols.remove('Weight (g)')

print(f"\nFeatures available: {len(numeric_cols)}")
print(f"Features: {numeric_cols}")

# Check if we have the core morphometric features
core_features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']
available_core = [f for f in core_features if f in numeric_cols]
print(f"\nCore morphometric features available: {len(available_core)}/{len(core_features)}")
print(f"Available: {available_core}")

# Create clean dataset
df_clean = numeric_df.dropna()
print(f"\nClean dataset: {len(df_clean)} samples after removing NaN")

# Separate features and target
X = df_clean[numeric_cols]
y = df_clean['Weight (g)']

print(f"\nTarget statistics:")
print(f"  Min: {y.min():.1f}g")
print(f"  Max: {y.max():.1f}g")
print(f"  Mean: {y.mean():.1f}g")
print(f"  Std: {y.std():.1f}g")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

print("Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
gb.fit(X_train_scaled, y_train)

# Make predictions
rf_pred = rf.predict(X_test_scaled)
gb_pred = gb.predict(X_test_scaled)
ensemble_pred = (rf_pred + gb_pred) / 2

# Calculate metrics
mae = mean_absolute_error(y_test, ensemble_pred)
rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
r2 = r2_score(y_test, ensemble_pred)
mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
accuracy = 100 - mape

within_10 = np.mean(np.abs(y_test - ensemble_pred) / y_test <= 0.1) * 100
within_5 = np.mean(np.abs(y_test - ensemble_pred) / y_test <= 0.05) * 100

print("\n" + "="*80)
print("📊 FINAL RESULTS WITH TRANSPARENCY ANALYSIS")
print("="*80)
print(f"MAE: {mae:.3f}g")
print(f"RMSE: {rmse:.3f}g") 
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Within 10%: {within_10:.1f}%")
print(f"Within 5%: {within_5:.1f}%")

# Analyze prediction distribution
print("\n🔍 PREDICTION ANALYSIS:")
print(f"Prediction range: {ensemble_pred.min():.1f}g - {ensemble_pred.max():.1f}g")
print(f"Actual range: {y_test.min():.1f}g - {y_test.max():.1f}g")

# Check for perfect predictions (suspicious)
perfect_predictions = np.sum(np.abs(y_test - ensemble_pred) < 0.001)
print(f"Near-perfect predictions (<0.001g error): {perfect_predictions}/{len(y_test)} ({perfect_predictions/len(y_test)*100:.1f}%)")

# Check prediction variance
pred_std = np.std(ensemble_pred)
actual_std = np.std(y_test)
print(f"Prediction std: {pred_std:.3f}g")
print(f"Actual std: {actual_std:.3f}g")

# Feature importance analysis
print("\n🔍 FEATURE IMPORTANCE (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': numeric_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")

print("\n" + "="*80)
print("🚨 TRANSPARENCY CONCLUSION")
print("="*80)

if accuracy > 99.0:
    print("❌ SUSPICIOUS: Accuracy >99% suggests potential data leakage")
    print("❌ This level of accuracy is unrealistic for biological prediction")
    print("❌ Publication reviewers will reject these results")
    print("\n🔧 RECOMMENDATIONS:")
    print("1. Remove columns highly correlated with weight (truth/error columns)")
    print("2. Use only pure morphometric features (Length, Width, Height, Area)")
    print("3. Ensure no weight-derived features are used")
    print("4. Consider the biological reality - fish weight prediction from images")
    print("5. Aim for realistic accuracy (80-95% is excellent for this domain)")
elif accuracy >= 90.0:
    print("✅ EXCELLENT: 90%+ accuracy achieved legitimately")
    print("✅ Results are suitable for publication")
else:
    print(f"📊 RESULT: {accuracy:.1f}% accuracy - acceptable for publication")

# Save analysis
with open('transparency_analysis.txt', 'w') as f:
    f.write("TRANSPARENCY ANALYSIS REPORT\n")
    f.write("="*50 + "\n")
    f.write(f"Dataset: {len(df)} samples\n")
    f.write(f"Features used: {len(numeric_cols)}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"MAE: {mae:.3f}g\n")
    f.write(f"Near-perfect predictions: {perfect_predictions}/{len(y_test)}\n")
    f.write(f"Prediction std: {pred_std:.3f}g\n")
    f.write(f"Actual std: {actual_std:.3f}g\n")
    
    if high_corr_cols:
        f.write(f"\nHIGH CORRELATION WARNINGS:\n")
        for col in high_corr_cols:
            f.write(f"- {col}: {correlation_with_weight[col]:.3f}\n")
    
    f.write(f"\nTOP FEATURES:\n")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
        f.write(f"{i+1}. {row['feature']}: {row['importance']:.3f}\n")

print(f"\n📄 Full analysis saved to: transparency_analysis.txt")