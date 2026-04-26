"""
Domain Shift Analysis: Create Domain-Invariant Datasets
========================================================
Load training and testing datasets, drop domain-specific columns,
and save cleaned versions for domain shift analysis.
"""

import pandas as pd
import os

# Define paths
datasets_folder = r"C:\Users\hamza\OneDrive\Desktop\Machine Learning Part\99_Main_Work\1_datasets\processed"
output_folder = r"C:\Users\hamza\OneDrive\Desktop\Machine Learning Part\99_Main_Work\2_Stuff\domain shift"

# Input files
trn_path = os.path.join(datasets_folder, "Evil_Twin-Dataset.csv")
tst_path = os.path.join(datasets_folder, "Evil_Twin-Dataset-Tst-Preprocessed.csv")

# Output files
trn_output = os.path.join(output_folder, "Evil_Twin-Dataset-Domain-Invariant.csv")
tst_output = os.path.join(output_folder, "Evil_Twin-Dataset-Tst-Domain-Invariant.csv")

# Columns to drop (domain-specific features)
columns_to_drop = [
    'frame.len',
    'frame.cap_len',
    'radiotap.dbm_antsignal',
    'wlan.duration',
    'wlan.frag',
    'wlan.seq',
    'data.len'
]

print("=" * 70)
print("CREATING DOMAIN-INVARIANT DATASETS")
print("=" * 70)

# Load training dataset
print(f"\nLoading training data from: {trn_path}")
df_train = pd.read_csv(trn_path)
print(f"Original shape: {df_train.shape}")
print(f"Columns: {df_train.columns.tolist()}")

# Load testing dataset
print(f"\nLoading testing data from: {tst_path}")
df_test = pd.read_csv(tst_path)
print(f"Original shape: {df_test.shape}")

# Drop columns from training data
print(f"\nDropping {len(columns_to_drop)} columns from training data...")
print(f"Columns to drop: {columns_to_drop}")
df_train_clean = df_train.drop(columns=columns_to_drop, errors='ignore')
print(f"New shape: {df_train_clean.shape}")
print(f"Remaining columns: {df_train_clean.columns.tolist()}")

# Drop columns from testing data
print(f"\nDropping {len(columns_to_drop)} columns from testing data...")
df_test_clean = df_test.drop(columns=columns_to_drop, errors='ignore')
print(f"New shape: {df_test_clean.shape}")

# Save cleaned training data
print(f"\nSaving training data to: {trn_output}")
df_train_clean.to_csv(trn_output, index=False)
print(f"✓ Saved successfully")

# Save cleaned testing data
print(f"Saving testing data to: {tst_output}")
df_test_clean.to_csv(tst_output, index=False)
print(f"✓ Saved successfully")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Training data: {df_train.shape[0]:,} rows × {df_train_clean.shape[1]} columns")
print(f"Testing data:  {df_test.shape[0]:,} rows × {df_test_clean.shape[1]} columns")
print(f"Columns removed: {len(columns_to_drop)}")
print(f"Domain-invariant datasets created successfully!")
print("=" * 70)
