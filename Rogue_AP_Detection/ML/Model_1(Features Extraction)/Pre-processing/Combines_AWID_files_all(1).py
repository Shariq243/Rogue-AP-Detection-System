# =============================================================================
# Research Evil Twin Detection - Converted from Jupyter Notebook
# =============================================================================
# Workspace: Project Data Directory



import csv
import shutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# =============================================================================
# 1. INFORMATION GATHERING
# =============================================================================

# =============================================================================
# 1.1 AWID Dataset (ATK-F-Trn) - Full Training packets of different attacks
# =============================================================================

# Define paths for training data
from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Define paths relative to project root
trn_path = BASE_DIR / "data" / "AWID-CLS-F-Trn"
tst_path = BASE_DIR / "data" / "AWID-CLS-F-Tst"

print("\n" + "=" * 80)
print("1.3 Evil Twin Dataset Creation")
print("=" * 80)

workspace_path = BASE_DIR / "data"
dataset_path = workspace_path


# List of CSV files to combine (from AWID-CLS-F-Trn directory)
csv_files = ['9.csv', '10.csv', '72.csv', '96.csv']
csv_full_paths = [os.path.join(trn_path, file) for file in csv_files]

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Iterate through each file
for i, file in enumerate(csv_full_paths):
    try:
        if i == 0:
            # Read the first file with the header
            data = pd.read_csv(file)
        else:
            # Read subsequent files without the header
            data = pd.read_csv(file, skiprows=1, header=None)
            # Assign the columns from the first file to the new data
            data.columns = combined_data.columns
        
        # Filter rows with labels 'normal' and 'impersonation' (evil_twin)
        filtered_data = data[data['class'].isin(['normal', 'impersonation'])]
        
        print(f"File {csv_files[i]}: Class distribution = {filtered_data['class'].value_counts().to_dict()}")
        
        # Append the filtered data to the combined DataFrame
        combined_data = pd.concat([combined_data, filtered_data], ignore_index=True)
    except FileNotFoundError:
        print(f'File not found: {file}')

# Rename 'impersonation' to 'evil_twin' for consistency
combined_data['class'] = combined_data['class'].replace('impersonation', 'evil_twin')

# Save the combined dataset to a new file
evil_twin_path = os.path.join(dataset_path, 'Evil_Twin-Dataset.csv')
combined_data.to_csv(evil_twin_path, index=False)
print(f"\nEvil_Twin-Dataset.csv has been created successfully at {evil_twin_path}!")

# Check class distribution
print("\n--- Class distribution in Evil_Twin Dataset ---")
counts = {}
with open(evil_twin_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        last_col = row[-1]
        if last_col not in counts:
            counts[last_col] = 0
        counts[last_col] += 1

unique_values = sorted(counts.keys())
for value in unique_values:
    print(f'{value}: {counts[value]}')

# Display head
print("\n--- Head of Evil_Twin-Dataset.csv ---")
df = pd.read_csv(evil_twin_path)
print(df.head())
