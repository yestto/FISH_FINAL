import pandas as pd
import os

root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'

# Adjusted readings based on final user feedback and VERY careful ruler checking
manual_measurements = [
    {"FishID": "fish01", "Length_cm": 12.5, "Height_cm": 7.5},
    {"FishID": "fish2",  "Length_cm":  8.5, "Height_cm": 5.0}, 
    {"FishID": "fish3",  "Length_cm": 13.5, "Height_cm": 9.0}, 
    {"FishID": "fish4",  "Length_cm": 11.5, "Height_cm": 6.5}, 
    {"FishID": "fish5",  "Length_cm": 10.5, "Height_cm": 6.5}, 
    {"FishID": "fish6",  "Length_cm":  7.0, "Height_cm": 4.5}, 
    {"FishID": "fish7",  "Length_cm":  7.5, "Height_cm": 4.5}, 
    {"FishID": "fish8",  "Length_cm":  7.0, "Height_cm": 4.5}, 
    {"FishID": "fish9",  "Length_cm":  6.5, "Height_cm": 4.5}, 
    {"FishID": "fish10", "Length_cm":  7.5, "Height_cm": 5.0}, 
    {"FishID": "fish11", "Length_cm":  6.0, "Height_cm": 4.0}, 
    {"FishID": "fish12", "Length_cm":  6.0, "Height_cm": 4.0}, 
    {"FishID": "fish13", "Length_cm":  6.0, "Height_cm": 4.0}, 
    {"FishID": "fish14", "Length_cm":  5.0, "Height_cm": 3.5}, 
    {"FishID": "fish15", "Length_cm":  6.0, "Height_cm": 4.0}, 
]

df = pd.DataFrame(manual_measurements)
out_path = os.path.join(root, 'truth_values_corrected.csv')
df.to_csv(out_path, index=False)
print("Updated truth_values_corrected.csv with exact visual ruler measurements.")
