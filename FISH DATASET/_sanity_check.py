import pandas as pd
import numpy as np

df = pd.read_csv('fish_frames_200_clean_unique_no_repeat.csv')
wdf = pd.read_csv('weights.csv')
weights = dict(zip(wdf['FishID'].astype(str), wdf['Weight']))

lines = []
lines.append("FISH  Wt(g) Rows  medL   medW   medH   L/W")
lines.append("-" * 65)

for fish in sorted(df['FishID'].unique()):
    g = df[df['FishID'] == fish]
    wt = weights.get(fish, 0)
    ml = g['Length (cm)'].median()
    mw = g['Width (cm)'].median()
    mh = g['Height (cm)'].median()
    lw = ml / mw if mw > 0 else 0
    lines.append(f"{fish:7s} {wt:5.1f} {len(g):4d}  {ml:5.1f}  {mw:5.1f}  {mh:5.1f}  {lw:4.1f}")

with open('_out_utf8_aggro.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
