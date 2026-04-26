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
# DEFINE PATHS
# =============================================================================

# Define base paths
workspace_path = r"c:\Users\hamza\OneDrive\Desktop\Machine Learning Part\99_Main_Work"
evil_twin_path = os.path.join(workspace_path, "1_datasets", "processed", "Evil_Twin-Dataset.csv")

# =============================================================================
# 1.3 Evil Twin Dataset (Loading)
# =============================================================================

print("\n" + "=" * 80)
print("1.3 Evil Twin Dataset")
print("=" * 80)

# Load the existing combined dataset
print(f"\nLoading Evil_Twin-Dataset.csv from {evil_twin_path}")
df = pd.read_csv(evil_twin_path)
combined_data = df

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

# =============================================================================
# 2. PREPROCESSING & MODEL TRAINING
# =============================================================================

# =============================================================================
# 2.1 INITIAL PREPROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("2.1 Initial Preprocessing")
print("=" * 80)

# Read the evil twin dataset
df = pd.read_csv(evil_twin_path)
print(f"\nInitial dataset shape: {df.shape}")
print(df.head())

# Get unique class counts
def unique_class_counts(df, target_column):
    class_counts = df[target_column].value_counts().to_dict()
    return class_counts

class_counts = unique_class_counts(df, 'class')
print(f"\nClass counts in the DataFrame: {class_counts}")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Find columns with all equal values
equal_columns = []
for column in df.columns:
    if all(df[column] == df[column].iloc[0]):
        equal_columns.append(column)
print(f"\nColumns with equal values in all rows: {equal_columns}")

# Columns to remove (from notebook)
columns_to_remove = [
    'wlan.fcs_good', 'radiotap.channel.freq', 'radiotap.channel.type.2ghz', 'radiotap.channel.type.half', 'wlan.fc.order', 
    'radiotap.channel.type.passive', 'radiotap.channel.type.gsm', 'wlan_mgt.tcprep.link_mrg', 'radiotap.present.db_antsignal', 
    'wlan.qos.buf_state_indicated', 'radiotap.present.tsft', 'radiotap.present.db_antnoise', 'radiotap.channel.type.sturbo', 
    'radiotap.present.rxflags', 'radiotap.flags.fcs', 'radiotap.channel.type.quarter', 'radiotap.channel.type.5ghz', 
    'radiotap.present.db_tx_attenuation', 'radiotap.flags.frag', 'radiotap.present.dbm_tx_power', 'radiotap.present.xchannel', 
    'radiotap.present.dbm_antnoise', 'radiotap.present.tx_attenuation', 'radiotap.present.vht', 'radiotap.channel.type.gfsk', 
    'frame.marked', 'radiotap.channel.type.turbo', 'radiotap.flags.datapad', 'wlan_mgt.fixed.chanwidth', 'radiotap.present.lock_quality', 
    'frame.interface_id', 'radiotap.present.reserved', 'radiotap.flags.wep', 'radiotap.flags.badfcs', 'radiotap.present.fhss', 
    'radiotap.version', 'wlan.fc.version', 'wlan_mgt.tcprep.trsmt_pow', 'radiotap.pad', 'frame.offset_shift', 'wlan_mgt.fixed.htact', 
    'radiotap.present.channel', 'radiotap.rxflags.badplcp', 'radiotap.present.vendor_ns', 'radiotap.present.flags', 'wlan.fc.frag', 
    'frame.dlt', 'frame.ignored', 'radiotap.flags.cfp', 'radiotap.flags.shortgi', 'radiotap.channel.type.cck'
]

# Drop the specified columns that exist in the dataframe
columns_to_remove = [col for col in columns_to_remove if col in df.columns]
df.drop(columns=columns_to_remove, inplace=True)

print(f"\nShape after dropping columns: {df.shape}")
print(df.head())

# Replace ? to NaN
pre_num_cols = len(df.columns)
df.replace('?', np.nan, inplace=True)

# Remove Columns (Features) which has more than 90% missing values
df.dropna(axis='columns', thresh=len(df.index)*0.10, inplace=True)

post_num_cols = len(df.columns)
print(f"\nNumber of columns dropped: {pre_num_cols - post_num_cols}")

print(df.head())

column_names_list = df.columns.tolist()
print(f"\nNumber of columns: {len(column_names_list)}")
print(f"Column names: {column_names_list}")

# Get updated class counts
class_counts = unique_class_counts(df, 'class')
print(f"\nUpdated Class counts in the DataFrame: {class_counts}")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# List of columns to drop
columns_to_drop = [
    'frame.time_epoch', 'radiotap.length', 'radiotap.present.mcs',
    'radiotap.present.ampdu', 'radiotap.flags.preamble', 'radiotap.channel.type.ofdm',
    'radiotap.channel.type.dynamic', 'radiotap.antenna', 'wlan.fc.retry',
    'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected', 'wlan.ba.control.multitid',
    'wlan.ba.control.cbitmap', 'wlan_mgt.fixed.fragment', 'wlan_mgt.fixed.sequence',
    'wlan.qos.ack', 'wlan.qos.amsdupresent', 'wlan.qos.bit4', 'wlan.qos.txop_dur_req',
    'wlan.wep.iv', 'wlan.wep.key', 'wlan.wep.icv','wlan.sa','wlan.ta', 'wlan.da','wlan.ra','wlan.bssid','wlan.ba.bm','radiotap.mactime'
]

# Drop only existing columns
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)

print(f"\nShape after dropping more columns: {df.shape}")
print(df.head())

column_names_list = df.columns.tolist()
print(f"\nNumber of columns: {len(column_names_list)}")
print(f"Column names: {column_names_list}")

# Replace NaN to 0
df.replace(np.nan, 0, inplace=True)

print(df.head())

# Drop columns that exist only in training but not in test data (to align datasets)
training_only_cols = ['wlan.qos.buf_state_indicated.1', 'wlan.qos.eosp']
training_only_cols = [col for col in training_only_cols if col in df.columns]
if training_only_cols:
    print(f"\nDropping columns that don't exist in test data: {training_only_cols}")
    df.drop(columns=training_only_cols, inplace=True)

print(f"Final shape after aligning with test data: {df.shape}")

# Get class counts before downsampling
class_counts = unique_class_counts(df, 'class')
print(f"\nClass counts before downsampling: {class_counts}")

# Downsample normal to be slightly above evil_twin count
evil_twin_count = len(df[df['class'] == 'evil_twin'])
normal_desired = int(evil_twin_count * 1.05)  # 5% above evil_twin count

df_evil_twin = df[df['class'] == 'evil_twin']
df_normal = df[df['class'] == 'normal'].sample(n=normal_desired, replace=False, random_state=42)
df = pd.concat([df_evil_twin, df_normal], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

class_counts = unique_class_counts(df, 'class')
print(f"\nClass counts after downsampling:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
class_counts = unique_class_counts(df, 'class')
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Save the downsampled dataset back to Evil_Twin-Dataset.csv
df.to_csv(evil_twin_path, index=False)
print(f"\n[SUCCESS] Updated Evil_Twin-Dataset.csv with downsampled data at {evil_twin_path}")
print(f"New dataset shape: {df.shape}")
