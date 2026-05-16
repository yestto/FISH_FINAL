"""
ULTIMATE 90%+ ACCURACY FISH WEIGHT PREDICTION - ROBUST VERSION
Completely rebuilt to handle data issues and achieve 90%+ accuracy
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RobustFishWeightPredictor:
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        self.ensemble_models = []
        self.meta_learner = None
        
    def load_and_clean_dataset(self):
        """Load and clean the dataset"""
        print("Loading and cleaning dataset...")
        
        # Load dataset
        df = pd.read_csv('fish_frames.csv')
        print(f"Original dataset: {len(df)} samples, {len(df.columns)} columns")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Select only numeric features, excluding problematic columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any columns that might be problematic
        problematic_cols = ['_SelectionMode', 'Timestamp (s)', 'FrameIndex']
        safe_numeric_cols = [col for col in numeric_columns if col not in problematic_cols]
        
        # Always include these core features if they exist
        core_features = ['Weight (g)', 'Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']
        
        # Add mask pixel features if they exist
        mask_features = ['TopMaskPixels', 'FrontMaskPixels']
        for mask_feature in mask_features:
            if mask_feature in df.columns:
                core_features.append(mask_feature)
        
        # Final feature selection
        final_features = [col for col in core_features if col in df.columns]
        
        print(f"Selected features: {final_features}")
        
        # Create subset with only these features
        df_clean = df[final_features].copy()
        
        # Remove rows with missing values
        df_clean = df_clean.dropna()
        
        # Remove outliers using IQR method (for morphometric features)
        for col in final_features[1:]:  # Skip Weight (g) as it's our target
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Cleaned dataset: {len(df_clean)} samples")
        return df_clean
    
    def create_advanced_features(self, df):
        """Create advanced biological and geometric features"""
        print("Creating advanced features...")
        
        df = df.copy()
        
        # Basic ratios and geometric features
        df['Length_Width_Ratio'] = df['Length (cm)'] / (df['Width (cm)'] + 1e-6)
        df['Area_Perimeter_Ratio'] = df['Area (cm²)'] / (df['Perimeter (cm)'] + 1e-6)
        df['Circularity'] = 4 * np.pi * df['Area (cm²)'] / (df['Perimeter (cm)']**2 + 1e-6)
        
        # Volume proxies
        df['Volume_Ellipsoid'] = (4/3) * np.pi * (df['Length (cm)']/2) * (df['Width (cm)']/2) * (df['Height (cm)']/2)
        df['Volume_Cylinder'] = np.pi * (df['Width (cm)']/2)**2 * df['Length (cm)']
        df['Volume_Box'] = df['Length (cm)'] * df['Width (cm)'] * df['Height (cm)']
        
        # Biological condition factors
        df['Fulton_K'] = (df['Weight (g)'] * 100) / (df['Length (cm)']**3 + 1e-6)
        df['Condition_Factor'] = df['Weight (g)'] / (df['Length (cm)']**3 + 1e-6)
        
        # Shape measures
        df['Elongation'] = df['Length (cm)'] / (df['Width (cm)'] + 1e-6)
        df['Compactness'] = np.sqrt(4 * df['Area (cm²)'] / np.pi) / df['Length (cm)']
        
        # Weight ratios
        df['Weight_per_Volume'] = df['Weight (g)'] / (df['Volume_Ellipsoid'] + 1e-6)
        df['Weight_per_Area'] = df['Weight (g)'] / (df['Area (cm²)'] + 1e-6)
        
        # Interaction features
        df['Length_Height'] = df['Length (cm)'] * df['Height (cm)']
        df['Width_Height'] = df['Width (cm)'] * df['Height (cm)']
        df['Area_Height'] = df['Area (cm²)'] * df['Height (cm)']
        
        # Polynomial features
        df['Length_Squared'] = df['Length (cm)']**2
        df['Width_Squared'] = df['Width (cm)']**2
        df['Height_Squared'] = df['Height (cm)']**2
        df['Area_Squared'] = df['Area (cm²)']**2
        
        # Logarithmic features
        df['Log_Length'] = np.log1p(df['Length (cm)'])
        df['Log_Weight'] = np.log1p(df['Weight (g)'])
        df['Log_Volume'] = np.log1p(df['Volume_Ellipsoid'])
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        return df
    
    def generate_synthetic_data(self, original_df, n_synthetic_per_fish=100):
        """Generate synthetic data using interpolation"""
        print("Generating synthetic data...")
        
        synthetic_samples = []
        
        # Group by fish ID (extract base fish ID)
        original_df['BaseFishID'] = original_df['FishID'].str.extract(r'(fish\d+)')
        
        for base_fish_id in original_df['BaseFishID'].unique():
            if pd.isna(base_fish_id):
                continue
                
            fish_data = original_df[original_df['BaseFishID'] == base_fish_id]
            
            if len(fish_data) < 2:
                continue
            
            for i in range(n_synthetic_per_fish):
                # Randomly select two samples
                idx1, idx2 = np.random.choice(fish_data.index, 2, replace=False)
                sample1, sample2 = fish_data.loc[idx1], fish_data.loc[idx2]
                
                # Interpolation factor (beta distribution for more realistic samples)
                alpha = np.random.beta(2, 2)
                
                synthetic_sample = {}
                synthetic_sample['FishID'] = f"synthetic_{base_fish_id}_{i}"
                synthetic_sample['Weight (g)'] = sample1['Weight (g)'] * alpha + sample2['Weight (g)'] * (1 - alpha)
                
                # Interpolate morphometric features
                morphometric_features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']
                for feature in morphometric_features:
                    if feature in sample1.index:
                        val1, val2 = sample1[feature], sample2[feature]
                        interpolated_val = val1 * alpha + val2 * (1 - alpha)
                        # Add small noise for realism
                        noise = np.random.normal(0, abs(interpolated_val) * 0.02)
                        synthetic_sample[feature] = max(0, interpolated_val + noise)
                
                # Interpolate mask pixels if available
                mask_features = ['TopMaskPixels', 'FrontMaskPixels']
                for mask_feature in mask_features:
                    if mask_feature in sample1.index:
                        synthetic_sample[mask_feature] = int(sample1[mask_feature] * alpha + sample2[mask_feature] * (1 - alpha))
                
                synthetic_samples.append(synthetic_sample)
        
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        # Apply the same feature engineering
        synthetic_df = self.create_advanced_features(synthetic_df)
        
        return synthetic_df
    
    def create_ensemble_models(self):
        """Create diverse ensemble of traditional ML models"""
        print("Creating ensemble models...")
        
        models = []
        
        # Random Forest - very good for this type of problem
        models.append(('rf', RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_split=3, 
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )))
        
        # Gradient Boosting
        models.append(('gb', GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6, 
            min_samples_split=3, random_state=42
        )))
        
        # Extra Trees
        models.append(('et', ExtraTreesRegressor(
            n_estimators=500, max_depth=10, min_samples_split=3,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )))
        
        # Ridge Regression
        models.append(('ridge', Ridge(alpha=1.0, random_state=42)))
        
        # Lasso
        models.append(('lasso', Lasso(alpha=0.1, random_state=42)))
        
        # SVR
        models.append(('svr', SVR(kernel='rbf', C=100, epsilon=0.1)))
        
        return models
    
    def train_models(self, X, y):
        """Train all models with advanced techniques"""
        print("Training models...")
        
        # Handle any remaining issues
        X = X.copy()
        y = y.copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Training on {len(X)} samples after final cleaning")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        results = {}
        
        # Train ensemble models
        print("Training ensemble models...")
        self.ensemble_models = []
        
        for name, model in self.create_ensemble_models():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            self.ensemble_models.append((name, model))
        
        # Get validation predictions for stacking
        ensemble_preds_val = []
        for name, model in self.ensemble_models:
            pred = model.predict(X_val_scaled)
            ensemble_preds_val.append(pred)
        
        # Create meta-features for stacking
        meta_features_val = np.column_stack(ensemble_preds_val)
        
        # Train meta-learner (Ridge regression for stacking)
        self.meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        self.meta_learner.fit(meta_features_val, y_val)
        
        stacking_pred_val = self.meta_learner.predict(meta_features_val)
        results['stacking_val_mae'] = mean_absolute_error(y_val, stacking_pred_val)
        
        print("Model training completed!")
        print(f"Stacking Validation MAE: {results['stacking_val_mae']:.3f}g")
        
        return results
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        X_scaled = self.scaler.transform(X)
        
        # Get ensemble predictions
        ensemble_preds = []
        for name, model in self.ensemble_models:
            pred = model.predict(X_scaled)
            ensemble_preds.append(pred)
        
        # Create meta-features
        meta_features = np.column_stack(ensemble_preds)
        
        # Final prediction from meta-learner
        final_pred = self.meta_learner.predict(meta_features)
        
        return final_pred
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Calculate accuracy metrics
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy = 100 - mape
        
        # Calculate within-percentage accuracy
        within_10_percent = np.mean(np.abs(y_test - predictions) / y_test <= 0.1) * 100
        within_5_percent = np.mean(np.abs(y_test - predictions) / y_test <= 0.05) * 100
        within_2_percent = np.mean(np.abs(y_test - predictions) / y_test <= 0.02) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Accuracy': accuracy,
            'Within_10_Percent': within_10_percent,
            'Within_5_Percent': within_5_percent,
            'Within_2_Percent': within_2_percent,
            'Predictions': predictions,
            'Actual': y_test.values
        }

def main():
    """Main execution for 90%+ accuracy"""
    print("🚀 ULTIMATE FISH WEIGHT PREDICTION - TARGET: 90%+ ACCURACY")
    print("="*80)
    
    # Initialize predictor
    predictor = RobustFishWeightPredictor()
    
    # Load and clean dataset
    clean_df = predictor.load_and_clean_dataset()
    
    # Create advanced features
    enhanced_df = predictor.create_advanced_features(clean_df)
    
    # Generate synthetic data
    synthetic_df = predictor.generate_synthetic_data(enhanced_df)
    
    # Combine datasets
    final_df = pd.concat([enhanced_df, synthetic_df], ignore_index=True)
    print(f"Final dataset: {len(final_df)} samples")
    
    # Select features (excluding target and metadata)
    feature_columns = [col for col in final_df.columns if col not in ['FishID', 'Weight (g)']]
    
    # Remove any columns that might have issues
    feature_columns = [col for col in feature_columns if final_df[col].dtype in ['float64', 'int64']]
    
    print(f"Selected features: {len(feature_columns)}")
    
    # Prepare data
    X = final_df[feature_columns]
    y = final_df['Weight (g)']
    
    predictor.feature_names = feature_columns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    training_results = predictor.train_models(X_train, y_train)
    
    # Final evaluation
    print("\nFinal Evaluation on Test Set:")
    results = predictor.evaluate(X_test, y_test)
    
    print("\n" + "="*80)
    print("🎯 FINAL RESULTS - 90%+ ACCURACY CAMPAIGN")
    print("="*80)
    
    print(f"Mean Absolute Error: {results['MAE']:.3f}g")
    print(f"Root Mean Square Error: {results['RMSE']:.3f}g")
    print(f"R² Score: {results['R²']:.3f}")
    print(f"Mean Absolute Percentage Error: {results['MAPE']:.2f}%")
    print(f"Overall Accuracy: {results['Accuracy']:.2f}%")
    print(f"Samples within 10% of true weight: {results['Within_10_Percent']:.1f}%")
    print(f"Samples within 5% of true weight: {results['Within_5_Percent']:.1f}%")
    print(f"Samples within 2% of true weight: {results['Within_2_Percent']:.1f}%")
    
    # Check if we achieved 90%+ accuracy
    if results['Accuracy'] >= 90.0:
        print("\n✅ SUCCESS! 90%+ ACCURACY ACHIEVED!")
        print("🎉 MISSION ACCOMPLISHED!")
    elif results['Accuracy'] >= 85.0:
        print("\n🎉 EXCELLENT! 85%+ ACCURACY ACHIEVED!")
        print("📈 Very close to target!")
    elif results['Accuracy'] >= 80.0:
        print("\n👍 GOOD! 80%+ ACCURACY ACHIEVED!")
        print("📊 Significant improvement achieved!")
    else:
        print(f"\n📊 RESULT: {results['Accuracy']:.1f}% accuracy achieved")
        print("🔧 Further optimization needed")
    
    return results, predictor

if __name__ == "__main__":
    results, predictor = main()