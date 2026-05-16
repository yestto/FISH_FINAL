import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading fish_frames.csv...")
df = pd.read_csv('fish_frames.csv')
print(f"Original: {len(df)} samples, {len(df.columns)} columns")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_cols].drop_duplicates()
print(f"After cleaning: {len(df)} samples, {len(df.columns)} numeric columns")

feature_cols = [col for col in df.columns if col != 'Weight (g)']
X = df[feature_cols]
y = df['Weight (g)']

print(f"Features: {len(feature_cols)}")
print(f"Target range: {y.min():.1f}g - {y.max():.1f}g")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training ensemble...")

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)

rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
gb_pred = gb.predict(X_test_scaled)

ensemble_pred = (rf_pred + gb_pred) / 2

mae = mean_absolute_error(y_test, ensemble_pred)
mse = mean_squared_error(y_test, ensemble_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, ensemble_pred)

mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
accuracy = 100 - mape

within_10 = np.mean(np.abs(y_test - ensemble_pred) / y_test <= 0.1) * 100
within_5 = np.mean(np.abs(y_test - ensemble_pred) / y_test <= 0.05) * 100

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"MAE: {mae:.3f}g")
print(f"RMSE: {rmse:.3f}g") 
print(f"R²: {r2:.3f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Within 10%: {within_10:.1f}%")
print(f"Within 5%: {within_5:.1f}%")

if accuracy >= 90.0:
    print("\n✅ SUCCESS! 90%+ ACCURACY ACHIEVED!")
elif accuracy >= 85.0:
    print("\n🎉 EXCELLENT! 85%+ ACCURACY ACHIEVED!")
elif accuracy >= 80.0:
    print("\n👍 GOOD! 80%+ ACCURACY ACHIEVED!")
else:
    print(f"\n📊 RESULT: {accuracy:.1f}% accuracy achieved")