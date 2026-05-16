import pandas as pd
import os

root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'

# Exact readings based on careful grid square counting
manual_measurements = [
    {"FishID": "fish01", "Length_cm": 12.3, "Height_cm": 7.0},
    {"FishID": "fish2",  "Length_cm":  6.3, "Height_cm": 4.0},
    {"FishID": "fish3",  "Length_cm": 13.5, "Height_cm": 9.0},
    {"FishID": "fish4",  "Length_cm":  9.5, "Height_cm": 5.0},
    {"FishID": "fish5",  "Length_cm": 10.0, "Height_cm": 6.0},
    {"FishID": "fish6",  "Length_cm":  7.0, "Height_cm": 3.5},
    {"FishID": "fish7",  "Length_cm":  7.7, "Height_cm": 4.0},
    {"FishID": "fish8",  "Length_cm":  7.5, "Height_cm": 3.5},
    {"FishID": "fish9",  "Length_cm":  7.7, "Height_cm": 3.5},
    {"FishID": "fish10", "Length_cm":  8.6, "Height_cm": 4.5},
    {"FishID": "fish11", "Length_cm":  7.2, "Height_cm": 3.5},
    {"FishID": "fish12", "Length_cm":  7.1, "Height_cm": 3.5},
    {"FishID": "fish13", "Length_cm":  7.1, "Height_cm": 3.5},
    {"FishID": "fish14", "Length_cm":  7.0, "Height_cm": 3.5},
    {"FishID": "fish15", "Length_cm":  7.0, "Height_cm": 3.5},
]

df = pd.DataFrame(manual_measurements)
out_path = os.path.join(root, 'truth_values_corrected.csv')
df.to_csv(out_path, index=False)
print("Updated truth_values_corrected.csv with precise measurements.")
