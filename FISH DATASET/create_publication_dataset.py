#!/usr/bin/env python3
"""
Create a publication-ready dataset by aggregating training frames
with one sample per fish to avoid data leakage.
"""

import pandas as pd
import numpy as np

def main():
    # Load the training frames dataset
    df = pd.read_csv('fish_frame_measurements_training_recreated.csv')
    
    print(f"Original training dataset: {len(df)} rows, {df['FishID'].nunique()} unique fish")
    
    # Group by FishID and calculate median values for each measurement
    # This gives us one representative sample per fish
    aggregated = df.groupby('FishID').agg({
        'Weight (g)': 'first',  # Weight should be the same for all frames of same fish
        'Length (cm)': 'median',
        'Width (cm)': 'median', 
        'Height (cm)': 'median',
        'Area (cm²)': 'median',
        'Perimeter (cm)': 'median',
        '_Truth_Length (cm)': 'first',  # Truth measurements should be consistent
        'Width_truth (cm)': 'first',
        'Area_truth (cm²)': 'first', 
        'Perimeter_truth (cm)': 'first'
    }).reset_index()
    
    # Calculate quality metrics
    aggregated['Length_Error_cm'] = abs(aggregated['Length (cm)'] - aggregated['_Truth_Length (cm)'])
    aggregated['Length_Error_pct'] = (aggregated['Length_Error_cm'] / aggregated['_Truth_Length (cm)']) * 100
    
    # Add frame count and quality score
    frame_counts = df.groupby('FishID').size().reset_index(name='Frame_Count')
    aggregated = aggregated.merge(frame_counts, on='FishID')
    
    # Calculate a composite quality score (lower is better)
    aggregated['Quality_Score'] = (
        aggregated['Length_Error_pct'] + 
        (100 / aggregated['Frame_Count'])  # Prefer fish with more frames
    )
    
    # Reorder columns for better readability
    column_order = [
        'FishID', 'Weight (g)', 'Length (cm)', 'Width (cm)', 'Height (cm)', 
        'Area (cm²)', 'Perimeter (cm)', '_Truth_Length (cm)', 'Width_truth (cm)',
        'Area_truth (cm²)', 'Perimeter_truth (cm)', 'Length_Error_cm', 
        'Length_Error_pct', 'Frame_Count', 'Quality_Score'
    ]
    
    final_df = aggregated[column_order]
    
    # Save the publication-ready dataset
    final_df.to_csv('fish_measurements_publication_ready.csv', index=False)
    
    print(f"\nPublication-ready dataset created:")
    print(f"- {len(final_df)} unique fish")
    print(f"- Mean length error: {final_df['Length_Error_cm'].mean():.2f} cm ({final_df['Length_Error_pct'].mean():.1f}%)")
    print(f"- Mean frame count per fish: {final_df['Frame_Count'].mean():.1f}")
    print(f"- Quality score range: {final_df['Quality_Score'].min():.1f} - {final_df['Quality_Score'].max():.1f}")
    
    # Show summary statistics
    print(f"\nDataset Summary:")
    print(f"Weight range: {final_df['Weight (g)'].min():.1f} - {final_df['Weight (g)'].max():.1f} g")
    print(f"Length range: {final_df['Length (cm)'].min():.1f} - {final_df['Length (cm)'].max():.1f} cm")
    print(f"Width range: {final_df['Width (cm)'].min():.1f} - {final_df['Width (cm)'].max():.1f} cm")
    print(f"Height range: {final_df['Height (cm)'].min():.1f} - {final_df['Height (cm)'].max():.1f} cm")
    
    return final_df

if __name__ == "__main__":
    main()