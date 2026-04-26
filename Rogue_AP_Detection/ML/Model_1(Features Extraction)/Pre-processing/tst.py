# =============================================================================
# Test Dataset Preprocessing - tst.py
# Loads existing Evil_Twin-Dataset-Tst.csv and preprocesses it
# =============================================================================

import os
import numpy as np
import pandas as pd

# =============================================================================
# DEFINE PATHS
# =============================================================================

from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

workspace_path = BASE_DIR / "data"
evil_twin_tst_path = os.path.join(workspace_path, "Evil_Twin-Dataset-Tst.csv")


print("=" * 80)
print("TEST DATASET PREPROCESSING - tst.py")
print("=" * 80)

# =============================================================================
# LOAD AND PREPROCESS TEST DATA
# =============================================================================

print("\n" + "=" * 80)
print("Loading and Preprocessing Test Data")
print("=" * 80)

# Load the combined test dataset
print(f"\nLoading dataset from {evil_twin_tst_path}")
if not os.path.exists(evil_twin_tst_path):
    print(f"[ERROR] File not found: {evil_twin_tst_path}")
    import sys
    sys.exit(1)

print("This may take a few minutes (8.29M rows)...")
import sys
sys.stdout.flush()

# Use optimized pandas settings for faster loading
df = pd.read_csv(
    evil_twin_tst_path, 
    engine='c',  # C engine is faster than Python
    dtype={'class': 'category'},  # Use category for class column to save memory
    memory_map=True  # Memory map the file
)
print(f"\n[SUCCESS] Loaded shape: {df.shape}")
print(f"Loading complete!")

# Class distribution before preprocessing
print("\n--- Class distribution before preprocessing ---")
df['class'] = df['class'].astype(str)  # Convert category back to string
class_counts = df['class'].value_counts().to_dict()
for class_name in sorted(class_counts.keys()):
    print(f"{class_name}: {class_counts[class_name]}")

# =============================================================================
# PREPROCESSING
# =============================================================================

# Columns to remove (same as training data)
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
print(f"\nDropping first set of {len(columns_to_remove)} columns...")
df.drop(columns=columns_to_remove, inplace=True)

print(f"Shape after dropping first set: {df.shape}")

# Replace ? to NaN
print("\nReplacing '?' with NaN...")
pre_num_cols = len(df.columns)
df.replace('?', np.nan, inplace=True)

# Remove Columns (Features) which has more than 90% missing values
print("Dropping columns with >90% missing values...")
df.dropna(axis='columns', thresh=len(df.index)*0.10, inplace=True)

post_num_cols = len(df.columns)
print(f"Columns dropped (>90% missing): {pre_num_cols - post_num_cols}")
print(f"Shape: {df.shape}")

# List of columns to drop (same as training)
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
print(f"\nDropping second set of {len(columns_to_drop)} columns...")
df.drop(columns=columns_to_drop, inplace=True)

print(f"Shape: {df.shape}")

# Replace NaN to 0
print("\nReplacing NaN with 0...")
df.replace(np.nan, 0, inplace=True)

# Drop columns that exist only in test but not in training data (to align datasets)
test_only_cols = ['wlan.ba.control.ackpolicy']
test_only_cols = [col for col in test_only_cols if col in df.columns]
if test_only_cols:
    print(f"Dropping columns that don't exist in training data: {test_only_cols}")
    df.drop(columns=test_only_cols, inplace=True)

print(f"Final shape before downsampling: {df.shape}")

# Get class counts before downsampling
print("\n--- Class counts before downsampling ---")
class_counts = df['class'].value_counts().to_dict()
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Downsample normal to be slightly above evil_twin count
print("\n--- Downsampling normal to match evil_twin count ---")
evil_twin_count = len(df[df['class'] == 'evil_twin'])
normal_desired = int(evil_twin_count * 1.05)  # 5% above evil_twin count
print(f"evil_twin count: {evil_twin_count}")
print(f"normal target: {normal_desired}")

df_evil_twin = df[df['class'] == 'evil_twin']
df_normal = df[df['class'] == 'normal'].sample(n=normal_desired, replace=False, random_state=42)
df = pd.concat([df_evil_twin, df_normal], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

class_counts = df['class'].value_counts().to_dict()
print(f"\nClass counts after downsampling:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

class_counts = df['class'].value_counts().to_dict()
print(f"\nClass counts after downsampling:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Save the preprocessed test dataset
print("\n--- Saving preprocessed test dataset ---")
print("This may take a few minutes...")
preprocessed_tst_path = os.path.join(workspace_path, "1_datasets", "processed", "Evil_Twin-Dataset-Tst-Preprocessed.csv")
df.to_csv(preprocessed_tst_path, index=False)
print(f"\n[SUCCESS] Preprocessed test dataset saved!")
print(f"Location: {preprocessed_tst_path}")
print(f"Final dataset shape: {df.shape}")

print("\n" + "=" * 80)
print("Test Dataset Preprocessing Completed!")
print("=" * 80)
