# combining_data.py
# This script combines multiple gesture CSV files into a single dataset CSV file.
import os
import pandas as pd

# Path where your CSV files are stored
DATA_DIR = "gesture_data"

# All gesture CSV files
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

all_data = []

for file in csv_files:
    label = file.replace(".csv", "")   # label is file name
    df = pd.read_csv(os.path.join(DATA_DIR, file))
    df["label"] = label                # add label column
    all_data.append(df)

# Combine all gesture data
dataset = pd.concat(all_data, ignore_index=True)

# Save combined dataset
dataset.to_csv("gesture_dataset.csv", index=False)

print("✅ Combined dataset saved as gesture_dataset.csv")
print("Classes:", dataset['label'].value_counts())

