import pandas as pd
import numpy as np

df_old = pd.read_csv('fish_frames.csv')
df_new = pd.read_csv('fish_frame_measurements_enhanced.csv')

def get_col(df, cols):
    for c in cols:
        if c in df.columns: return c
    return None

def analyze(df, name):
    print(f"\n--- {name} ---")
    print(f"Total Rows (Frames): {len(df)}")
    print(f"Number of Fish Detected: {df['FishID'].nunique()}")
    
    length_col = get_col(df, ['Length_cm', 'Length (cm)', 'TopLength(cm)'])
    width_col = get_col(df, ['Width_cm', 'Width (cm)', 'TopWidth(cm)'])
    height_col = get_col(df, ['Height_cm', 'Height (cm)', 'FrontHeight(cm)'])
    
    cvs = []
    for fish_id, grp in df.groupby('FishID'):
        for col in [length_col, width_col, height_col]:
            if col and grp[col].mean() > 0:
                cv = (grp[col].std() / grp[col].mean()) * 100
                if not np.isnan(cv):
                    cvs.append(cv)
    if cvs:
        print(f"Median Measurement CV (lower is better): {np.nanmedian(cvs):.2f}%")
        print(f"Lowest (Best) CV: {np.nanmin(cvs):.2f}%")

analyze(df_old, 'Baseline (Before Enhancements)')
analyze(df_new, 'Enhanced Pipeline (Our Techniques)')
