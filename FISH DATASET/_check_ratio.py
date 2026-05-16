import pandas as pd
df = pd.read_csv('fish_frame_measurements_groups_relaxed.csv')
for f in ['fish9', 'fish10', 'fish12']:
    g = df[df['FishID'] == f].copy()
    g['L_W_Ratio'] = g['Length (cm)'] / g['Width (cm)']
    print(f'{f}: Max ratio = {g["L_W_Ratio"].max():.2f}')
    print(f'{f}: frames > 1.8 = {len(g[g["L_W_Ratio"] >= 1.8])}')
    print(f'{f}: frames > 2.0 = {len(g[g["L_W_Ratio"] >= 2.0])}')
