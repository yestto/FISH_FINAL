import pandas as pd
df = pd.read_csv('fish_frame_measurements_groups_relaxed.csv')

def get_stats(f_id):
    g = df[df['FishID'] == f_id].copy()
    g['Width (cm)'] = pd.to_numeric(g['Width (cm)'], errors='coerce')
    g['Length (cm)'] = pd.to_numeric(g['Length (cm)'], errors='coerce')
    g = g.dropna(subset=['Width (cm)', 'Length (cm)'])
    print(f"\n{f_id} (n={len(g)}) Raw Quantiles:")
    for metric in ['Length (cm)', 'Width (cm)']:
        print(f"  {metric}: ", end='')
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
            print(f"{g[metric].quantile(q):.1f} ", end='')
        print()

for f in ['fish01', 'fish2', 'fish11', 'fish13', 'fish6']:
    get_stats(f)
