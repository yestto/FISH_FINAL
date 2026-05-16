"""
PUBLICATION-READY FISH WEIGHT PREDICTION
Corrected methodology with proper scaling, validation, and statistical rigor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load data with proper preprocessing"""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv('fish_frames.csv')
    print(f"Original dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape[0]} samples")
    
    # Create scientifically meaningful features
    df['Length_Width_Ratio'] = df['Length (cm)'] / (df['Width (cm)'] + 1e-6)
    df['Area_Perimeter_Ratio'] = df['Area (cm²)'] / (df['Perimeter (cm)'] + 1e-6)
    df['Volume_Proxy'] = df['Length (cm)'] * df['Width (cm)'] * df['Height (cm)']
    df['Mask_Density'] = (df['TopMaskPixels'] + df['FrontMaskPixels']) / 2
    
    return df

def select_features_correlation(df, threshold=0.8):
    """Select features based on correlation analysis"""
    print("\nPerforming feature selection...")
    
    # Initial feature pool
    features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)',
                'TopMaskPixels', 'FrontMaskPixels', 'Length_Width_Ratio', 
                'Area_Perimeter_Ratio', 'Volume_Proxy', 'Mask_Density']
    
    # Calculate correlation matrix
    X_temp = df[features]
    corr_matrix = X_temp.corr().abs()
    
    # Find highly correlated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    selected_features = [f for f in features if f not in to_drop]
    
    print(f"Selected {len(selected_features)} features after correlation analysis:")
    for feat in selected_features:
        print(f"  - {feat}")
    
    return selected_features

def robust_cross_validation(df, features, target_col='Weight (g)', fish_id_col='FishID'):
    """Perform robust Leave-One-Fish-Out cross-validation"""
    print(f"\nPerforming Leave-One-Fish-Out cross-validation...")
    
    X = df[features]
    y = df[target_col]
    groups = df[fish_id_col]
    
    # Initialize scalers
    scaler_X = RobustScaler()  # More robust to outliers
    scaler_y = RobustScaler()
    
    logo = LeaveOneGroupOut()
    
    # Define models with regularization
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'SVR': SVR(kernel='rbf', C=10, epsilon=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        maes = []
        r2s = []
        predictions = []
        actuals = []
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            # Scale target (optional, but often helps)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            # Fit model
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predict
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            maes.append(mae)
            r2s.append(r2)
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            
            test_fish = groups.iloc[test_idx].iloc[0]
            print(f"  Fold {fold+1} (Fish {test_fish}): MAE={mae:.3f}g, R²={r2:.3f}")
        
        # Calculate overall statistics
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        mean_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)
        
        # Calculate confidence intervals
        mae_ci = stats.t.interval(0.95, len(maes)-1, loc=mean_mae, scale=stats.sem(maes))
        r2_ci = stats.t.interval(0.95, len(r2s)-1, loc=mean_r2, scale=stats.sem(r2s))
        
        results[name] = {
            'mae_mean': mean_mae,
            'mae_std': std_mae,
            'mae_ci': mae_ci,
            'r2_mean': mean_r2,
            'r2_std': std_r2,
            'r2_ci': r2_ci,
            'maes': maes,
            'r2s': r2s,
            'predictions': predictions,
            'actuals': actuals
        }
        
        print(f"  Overall: MAE={mean_mae:.3f}±{std_mae:.3f}g, R²={mean_r2:.3f}±{std_r2:.3f}")
        print(f"  95% CI: MAE [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}], R² [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
    
    return results

def perform_statistical_tests(results):
    """Perform statistical significance tests"""
    print("\n=== STATISTICAL SIGNIFICANCE TESTS ===")
    
    # Compare best models using paired t-test
    model_names = list(results.keys())
    
    print("Paired t-tests for MAE differences:")
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            mae1, mae2 = results[model1]['maes'], results[model2]['maes']
            
            t_stat, p_value = stats.ttest_rel(mae1, mae2)
            print(f"  {model1} vs {model2}: t={t_stat:.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print(f"    → Significant difference (p<0.05)")
            else:
                print(f"    → No significant difference (p≥0.05)")

def generate_publication_report(results):
    """Generate publication-ready report"""
    print("\n" + "="*80)
    print("PUBLICATION-READY RESULTS")
    print("="*80)
    
    # Find best model
    best_model = min(results.keys(), key=lambda x: results[x]['mae_mean'])
    best_results = results[best_model]
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"Mean Absolute Error: {best_results['mae_mean']:.3f} ± {best_results['mae_std']:.3f} g")
    print(f"R² Score: {best_results['r2_mean']:.3f} ± {best_results['r2_std']:.3f}")
    print(f"95% Confidence Interval:")
    print(f"  MAE: [{best_results['mae_ci'][0]:.3f}, {best_results['mae_ci'][1]:.3f}] g")
    print(f"  R²:  [{best_results['r2_ci'][0]:.3f}, {best_results['r2_ci'][1]:.3f}]")
    
    print(f"\n📊 ALL MODEL RESULTS:")
    print(f"{'Model':<20} {'MAE (g)':<12} {'R²':<10} {'Std (MAE)':<12}")
    print("-" * 60)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['mae_mean']):
        print(f"{name:<20} {result['mae_mean']:<12.3f} {result['r2_mean']:<10.3f} {result['mae_std']:<12.3f}")
    
    print(f"\n🔬 METHODOLOGY VALIDATION:")
    print(f"✅ Leave-One-Fish-Out Cross-Validation (n={len(best_results['maes'])} folds)")
    print(f"✅ Robust feature scaling applied")
    print(f"✅ Feature selection based on correlation analysis")
    print(f"✅ Statistical significance testing performed")
    print(f"✅ Confidence intervals calculated")
    
    # Check publication readiness
    if best_results['mae_mean'] < 5.0 and best_results['r2_mean'] > 0.7:
        print(f"\n✅ PUBLICATION READY - Results meet scientific standards")
    else:
        print(f"\n⚠️  CAUTION - Results may need additional validation")

def main():
    """Main execution"""
    print("PUBLICATION-READY FISH WEIGHT PREDICTION")
    print("="*80)
    print("Corrected methodology with proper scaling and validation")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Select features
    selected_features = select_features_correlation(df)
    
    # Perform robust cross-validation
    results = robust_cross_validation(df, selected_features)
    
    # Statistical tests
    perform_statistical_tests(results)
    
    # Generate publication report
    generate_publication_report(results)

if __name__ == "__main__":
    main()