import csv
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report

# =============================================================================
# DEFINE PATHS AND LOAD DATA
# =============================================================================

from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

workspace_path = BASE_DIR / "data"
evil_twin_trn_path = os.path.join(workspace_path, "Evil_Twin-Dataset.csv")
evil_twin_tst_path = os.path.join(workspace_path, "Evil_Twin-Dataset-Tst-Preprocessed.csv")


print("\n" + "=" * 80)
print("3.2 TRAINING AND EVALUATION WITH BEST HYPERPARAMETERS")
print("=" * 80)

# Load preprocessed training dataset
print(f"\nLoading preprocessed training dataset from {evil_twin_trn_path}")
df_train = pd.read_csv(evil_twin_trn_path)
print(f"Training dataset shape: {df_train.shape}")
print(f"Training class distribution: {df_train['class'].value_counts().to_dict()}")

# Load preprocessed test dataset
print(f"\nLoading preprocessed test dataset from {evil_twin_tst_path}")
df_test = pd.read_csv(evil_twin_tst_path)
print(f"Test dataset shape: {df_test.shape}")
print(f"Test class distribution: {df_test['class'].value_counts().to_dict()}")

# =============================================================================
# PREPROCESSING
# =============================================================================

print("\n--- Preprocessing ---")

# Preprocessing for training data
df_train_rf = df_train.copy()

print(f"\nTraining data - Class counts: {df_train_rf['class'].value_counts().to_dict()}")

# Convert hexadecimal strings to numeric (integer) values
df_train_rf['wlan.fc.type_subtype'] = df_train_rf['wlan.fc.type_subtype'].apply(lambda x: int(x, 16))
df_train_rf['wlan.fc.ds'] = df_train_rf['wlan.fc.ds'].apply(lambda x: int(x, 16))

# Create normalized training dataframe
df_train_normalized = df_train_rf

# Filter evil_twin instances (keep all normal instances)
df_train_evil_twin = df_train_normalized[df_train_normalized['class'] == 'evil_twin']
df_train_normal = df_train_normalized[df_train_normalized['class'] == 'normal']

# Concatenate and shuffle training data
df_train_normalized = pd.concat([df_train_evil_twin, df_train_normal], ignore_index=True)
df_train_normalized = df_train_normalized.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Training data after balancing: {df_train_normalized['class'].value_counts().to_dict()}")

df_train_normalized = shuffle(df_train_normalized)

# Preprocessing for test data
df_test_rf = df_test.copy()

print(f"\nTest data - Class counts: {df_test_rf['class'].value_counts().to_dict()}")

# Convert hexadecimal strings to numeric (integer) values
df_test_rf['wlan.fc.type_subtype'] = df_test_rf['wlan.fc.type_subtype'].apply(lambda x: int(x, 16))
df_test_rf['wlan.fc.ds'] = df_test_rf['wlan.fc.ds'].apply(lambda x: int(x, 16))

print(f"Test data after preprocessing: {df_test_rf['class'].value_counts().to_dict()}")

# =============================================================================
# LOAD BEST HYPERPARAMETERS
# =============================================================================

print("\n--- Loading Best Hyperparameters ---")

best_params_path = os.path.join(workspace_path, "0_research", "best_params.json")

if os.path.exists(best_params_path):
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    print(f"\n✓ Loaded best parameters from: {best_params_path}")
    print(f"\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
else:
    print(f"\n✗ Best parameters file not found!")
    print(f"Please run 3_1_hyperparameter_tuning.py first to generate best_params.json")
    exit()

# =============================================================================
# PREPARE DATA AND TRAIN MODEL
# =============================================================================

print("\n--- Training and Evaluation ---")

# Prepare training data
X_train = df_train_normalized.drop(columns=['class'])
y_train = df_train_normalized['class']

# Prepare test data
X_test = df_test_rf.drop(columns=['class'])
y_test = df_test_rf['class']

# Train with best hyperparameters
print("\nTraining Random Forest with best hyperparameters...")
classifier = RandomForestClassifier(random_state=42, **best_params)
classifier.fit(X_train, y_train)

print("✓ Model training completed!")

# Make predictions
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

# =============================================================================
# EVALUATE MODEL
# =============================================================================

print("\n--- Model Evaluation ---")

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
train_precision = precision_score(y_train, y_train_pred, average='weighted')

test_accuracy = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')

# Display training metrics
print("\nRandom Forest Model Metrics - Training Set:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-score:  {train_f1:.4f}")
print(f"  Precision: {train_precision:.4f}")

# Display testing metrics
print("\nRandom Forest Model Metrics - Testing Set:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-score:  {test_f1:.4f}")
print(f"  Precision: {test_precision:.4f}")

# Classification report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_test_pred))

# Cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean():.4f}")
print(f"Std Dev: {cv_scores.std():.4f}")

# Feature importances
print("\n--- Feature Importances ---")
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
importance_percentages = (importances / importances.sum()) * 100

feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns[indices],
    'Importance Percentage': importance_percentages[indices]
})

print(feature_importance_df.to_string(index=False))

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("RANDOM FOREST MODEL RESULTS SUMMARY")
print("=" * 80)

print("\nTraining Set Performance:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-score:  {train_f1:.4f}")
print(f"  Precision: {train_precision:.4f}")

print("\nTesting Set Performance:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-score:  {test_f1:.4f}")
print(f"  Precision: {test_precision:.4f}")

print("\n" + "=" * 80)
print("Script Execution Completed!")
print("=" * 80)
