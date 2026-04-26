# =============================================================================
# FYP Phase 2: Domain-Invariant Feature Selection via KS-Test
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("=" * 60)
print("PHASE 1: LOADING DATA")
print("=" * 60)

from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

evil_twin_trn_path = BASE_DIR / "data" / "Evil_Twin-Dataset.csv"
evil_twin_tst_path = BASE_DIR / "data" / "Evil_Twin-Dataset-Tst-Preprocessed.csv"


df_train = pd.read_csv(evil_twin_trn_path)
df_test = pd.read_csv(evil_twin_tst_path)

print("Converting hexadecimal features...")
for df in [df_train, df_test]:
    df['wlan.fc.type_subtype'] = df['wlan.fc.type_subtype'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else x)
    df['wlan.fc.ds'] = df['wlan.fc.ds'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else x)

X_train = df_train.drop(columns=['class'])
y_train = df_train['class']
X_test = df_test.drop(columns=['class'])
y_test = df_test['class']

print("\n" + "=" * 60)
print("PHASE 2: STATISTICAL DRIFT ANALYSIS (KS-TEST)")
print("=" * 60)

# We will drop any feature where the distribution shifted by more than 15%
DRIFT_THRESHOLD = 0.15 
drifting_features = []
stable_features = []

print(f"Running Two-Sample Kolmogorov-Smirnov Test (Threshold: {DRIFT_THRESHOLD})...")

for col in X_train.columns:
    # Compare the column in Train vs the column in Test
    ks_stat, p_value = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
    
    if ks_stat > DRIFT_THRESHOLD:
        drifting_features.append(col)
        print(f" DROPPED: {col} (Drift Score: {ks_stat:.4f})")
    else:
        stable_features.append(col)
        print(f" KEPT:    {col} (Drift Score: {ks_stat:.4f})")

print(f"\nTotal Features Kept: {len(stable_features)} out of {len(X_train.columns)}")

# Filter the datasets to ONLY include stable, domain-invariant features
X_train_stable = X_train[stable_features].to_numpy().astype('float32')
X_test_stable = X_test[stable_features].to_numpy().astype('float32')
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

print("\n" + "=" * 60)
print("PHASE 3: TRAINING & BLIND TEST ON STABLE FEATURES")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=4,
    min_samples_leaf=2,
    criterion='entropy',
    max_features='sqrt',
    n_jobs=-1, 
    random_state=42
)

print(f"Training model on Trn using only the {len(stable_features)} stable features...")
rf_model.fit(X_train_stable, y_train_np)

print("Predicting on Tst dataset...")
y_test_pred = rf_model.predict(X_test_stable)

print("\n" + "=" * 50)
print(" DOMAIN-INVARIANT BLIND TEST RESULTS ")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test_np, y_test_pred) * 100:.4f}%")
print(f"F1-Score: {f1_score(y_test_np, y_test_pred, average='weighted') * 100:.4f}%")
print("\nClassification Report:")
print(classification_report(y_test_np, y_test_pred))