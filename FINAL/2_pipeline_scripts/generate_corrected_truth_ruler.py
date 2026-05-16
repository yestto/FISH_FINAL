import pandas as pd
import os

root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'

# Corrected readings reading STRICTLY from the cm ruler in the photo, NOT the graph paper grid
manual_measurements = [
    {"FishID": "fish01", "Length_cm": 12.5, "Height_cm": 7.5},
    {"FishID": "fish2",  "Length_cm":  8.5, "Height_cm": 5.0}, # +2.2cm
    {"FishID": "fish3",  "Length_cm": 15.0, "Height_cm": 10.0}, # +1.5cm
    {"FishID": "fish4",  "Length_cm": 11.5, "Height_cm": 6.5}, # +2cm
    {"FishID": "fish5",  "Length_cm": 10.5, "Height_cm": 6.5}, 
    {"FishID": "fish6",  "Length_cm":  8.0, "Height_cm": 4.5}, # +1cm
    {"FishID": "fish7",  "Length_cm":  8.5, "Height_cm": 4.5}, # +0.8cm
    {"FishID": "fish8",  "Length_cm":  9.0, "Height_cm": 4.5}, # +1.5cm
    {"FishID": "fish9",  "Length_cm":  8.5, "Height_cm": 4.5}, # +0.8cm
    {"FishID": "fish10", "Length_cm": 10.0, "Height_cm": 5.5}, # +1.4cm
    {"FishID": "fish11", "Length_cm":  8.0, "Height_cm": 4.5}, # +0.8cm
    {"FishID": "fish12", "Length_cm":  8.0, "Height_cm": 4.5}, # +0.9cm
    {"FishID": "fish13", "Length_cm":  8.0, "Height_cm": 4.5}, # +0.9cm
    {"FishID": "fish14", "Length_cm":  8.0, "Height_cm": 4.5}, # +1.0cm
    {"FishID": "fish15", "Length_cm":  8.0, "Height_cm": 4.5}, # +1.0cm
]

df = pd.DataFrame(manual_measurements)
out_path = os.path.join(root, 'truth_values_corrected.csv')
df.to_csv(out_path, index=False)
print("Updated truth_values_corrected.csv with accurate RULER-BASED measurements.")
