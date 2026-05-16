import os
import pandas as pd

df = pd.read_csv('fish_frames_200_clean_unique_no_repeat.csv')
print(f'Current CSV total rows: {len(df)}')
print(f'Current CSV unique fish: {df["FishID"].nunique()}')

dir_path = 'final stages clean no repeat'
if os.path.exists(dir_path):
    dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    print(f'\nFolders in {dir_path}: {len(dirs)}')
    for d in dirs:
        imgs = [f for f in os.listdir(os.path.join(dir_path, d)) if f.endswith('.jpg')]
        csv_count = len(df[df['FishID']==d])
        print(f'  {d}: {len(imgs)} images (CSV says it should have {csv_count})')
