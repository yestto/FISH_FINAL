import pandas as pd
df = pd.read_csv('fish_frames.csv')
for f in ['fish6', 'fish7', 'fish8', 'fish9']:
    g = df[df['FishID'] == f]
    print(f'\n{f}: n={len(g)}')
    if len(g) > 0:
        print(f"  Length: med={g['Length (cm)'].median():.2f} min={g['Length (cm)'].min():.2f} max={g['Length (cm)'].max():.2f}")
        print(f"  Width : med={g['Width (cm)'].median():.2f} min={g['Width (cm)'].min():.2f} max={g['Width (cm)'].max():.2f}")
        print(f"  L/W Ratio: {(g['Length (cm)'].median() / g['Width (cm)'].median()):.2f}")
