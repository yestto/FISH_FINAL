#!/usr/bin/env python3
"""
Simple visualization script for regression results without external dependencies.
Creates text-based visualizations and saves data for external plotting.
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

def load_data(csv_path):
    """Load the fish frames dataset."""
    df = pd.read_csv(csv_path)
    return df

def prepare_regression_data(df, target_col, feature_cols):
    """Prepare data for regression."""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['FishID'].values
    return X, y, groups

def train_models(X_train, y_train, random_state=42):
    """Train multiple regression models with proper regularization."""
    models = {
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=random_state))
        ]),
        'svr_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'))
        ]),
        'svr_rbf_stable': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=1.0, epsilon=0.2, gamma=0.1))
        ]),
        'gbr_stable': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=2, 
            subsample=0.8, min_samples_leaf=3, random_state=random_state
        ),
        'extra_trees_stable': ExtraTreesRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', bootstrap=True, random_state=random_state
        ),
        'random_forest_stable': RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', random_state=random_state
        ),
        'ridge_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.5, random_state=random_state))
        ]),
        'lasso_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.01, max_iter=2000, random_state=random_state))
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def create_text_visualization(y_test, y_pred_dict, target_name, save_dir):
    """Create text-based visualization of results."""
    output_file = os.path.join(save_dir, f'text_visualization_{target_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"REGRESSION RESULTS VISUALIZATION - {target_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("PREDICTION ACCURACY SUMMARY:\n")
        f.write("-"*40 + "\n")
        
        for model_name, y_pred in y_pred_dict.items():
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            f.write(f"{model_name:20s}: MAE={mae:6.3f}, RMSE={rmse:6.3f}, R²={r2:6.3f}\n")
        
        f.write("\n\nPREDICTION VS ACTUAL - TEXT SCATTER PLOT:\n")
        f.write("-"*50 + "\n")
        f.write("Format: [Actual -> Predicted] (Error)\n\n")
        
        # Show first 20 predictions for the best model (lowest MAE)
        best_model = min(y_pred_dict.items(), key=lambda x: mean_absolute_error(y_test, x[1]))[0]
        y_pred_best = y_pred_dict[best_model]
        
        f.write(f"Best Model: {best_model}\n")
        f.write("Sample Predictions (first 20):\n")
        for i in range(min(20, len(y_test))):
            actual = y_test[i]
            predicted = y_pred_best[i]
            error = actual - predicted
            f.write(f"[{actual:6.2f} -> {predicted:6.2f}] (Error: {error:6.2f})\n")
        
        f.write("\n\nERROR DISTRIBUTION:\n")
        f.write("-"*30 + "\n")
        
        for model_name, y_pred in y_pred_dict.items():
            errors = y_test - y_pred
            f.write(f"{model_name:20s}: Mean Error={np.mean(errors):6.3f}, Std Error={np.std(errors):6.3f}\n")
        
        f.write("\n\nPERFORMANCE RANKING (by MAE):\n")
        f.write("-"*35 + "\n")
        
        performance = []
        for model_name, y_pred in y_pred_dict.items():
            mae = mean_absolute_error(y_test, y_pred)
            performance.append((model_name, mae))
        
        performance.sort(key=lambda x: x[1])
        for rank, (model_name, mae) in enumerate(performance, 1):
            f.write(f"{rank:2d}. {model_name:20s}: MAE={mae:6.3f}\n")

def create_csv_data(y_test, y_pred_dict, target_name, save_dir):
    """Create CSV data for external plotting."""
    output_file = os.path.join(save_dir, f'plot_data_{target_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.csv')
    
    df_data = pd.DataFrame({
        'Actual': y_test,
        'FishID': range(len(y_test))  # Add fish ID for grouping
    })
    
    for model_name, y_pred in y_pred_dict.items():
        df_data[f'{model_name}_Predicted'] = y_pred
        df_data[f'{model_name}_Error'] = y_test - y_pred
    
    df_data.to_csv(output_file, index=False)
    print(f"Plot data saved to: {output_file}")

def create_ascii_plot(y_test, y_pred_dict, target_name, save_dir):
    """Create simple ASCII art plots."""
    output_file = os.path.join(save_dir, f'ascii_plot_{target_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"ASCII PLOT - {target_name}\n")
        f.write("="*50 + "\n\n")
        
        best_model = min(y_pred_dict.items(), key=lambda x: mean_absolute_error(y_test, x[1]))[0]
        y_pred_best = y_pred_dict[best_model]
        
        f.write(f"Scatter Plot: Actual vs Predicted (Best Model: {best_model})\n")
        f.write("-"*50 + "\n")
        
        # Create a simple ASCII scatter plot
        min_val = min(y_test.min(), y_pred_best.min())
        max_val = max(y_test.max(), y_pred_best.max())
        
        # Create 20x20 grid
        grid_size = 20
        plot = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        
        for i in range(len(y_test)):
            x = int((y_test[i] - min_val) / (max_val - min_val) * (grid_size-1))
            y = int((y_pred_best[i] - min_val) / (max_val - min_val) * (grid_size-1))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                plot[grid_size-1-y][x] = '*'
        
        # Add perfect line
        for i in range(grid_size):
            x = i
            y = i
            if 0 <= x < grid_size and 0 <= y < grid_size and plot[grid_size-1-y][x] == ' ':
                plot[grid_size-1-y][x] = '+'
        
        f.write(f"Predicted ↑\n")
        for row in plot:
            f.write(''.join(row) + '\n')
        f.write("Actual →\n")
        
        f.write(f"\nRange: {min_val:.2f} to {max_val:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description='Create text visualizations for regression results')
    parser.add_argument('--csv', required=True, help='Path to fish_frames.csv')
    parser.add_argument('--target', required=True, 
                       choices=['_Truth_Length (cm)', 'Width_truth (cm)', 'Area_truth (cm²)', 'Perimeter_truth (cm)'], 
                       help='Target column for regression')
    parser.add_argument('--output-dir', default='regression_visualizations', help='Directory to save visualizations')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_data(args.csv)
    
    # Define feature columns
    feature_cols = ['Length (cm)', 'Width (cm)', 'Area (cm²)', 'Perimeter (cm)']
    
    # Prepare data
    X, y, groups = prepare_regression_data(df, args.target, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)
    
    # Train models
    models_dict = train_models(X_train, y_train, args.random_state)
    
    # Get predictions
    y_pred_dict = {}
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_dict[model_name] = y_pred
    
    # Create all visualizations
    print(f"Creating text visualization for {args.target}...")
    create_text_visualization(y_test, y_pred_dict, args.target, args.output_dir)
    
    print(f"Creating CSV data for {args.target}...")
    create_csv_data(y_test, y_pred_dict, args.target, args.output_dir)
    
    print(f"Creating ASCII plot for {args.target}...")
    create_ascii_plot(y_test, y_pred_dict, args.target, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}")
    print(f"Files created:")
    for file in os.listdir(args.output_dir):
        if args.target.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") in file:
            print(f"  - {file}")

if __name__ == '__main__':
    main()