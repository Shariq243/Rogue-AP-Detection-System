# =============================================================================
# FYP PHASE 2 FINAL: LOGISTIC REGRESSION CHAMPION MODEL
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

print("=" * 60)
print("PHASE 1: LOADING & CLEANING DATA")
print("=" * 60)

from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

train_path = BASE_DIR / "data" / "Evil_Twin-Dataset-Domain-Invariant.csv"
test_path = BASE_DIR / "data" / "Evil_Twin-Dataset-Tst-Domain-Invariant.csv"



print("Loading Data...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# --- THE FIX: HEX TO INT CONVERSION ---
def hex_to_int(x):
    if isinstance(x, str) and x.startswith('0x'):
        return int(x, 16)
    try:
        return float(x)
    except:
        return 0

print("Cleaning hexadecimal strings...")
for col in df_train.columns:
    if col != 'class':
        df_train[col] = df_train[col].apply(hex_to_int)
        df_test[col] = df_test[col].apply(hex_to_int)

# Map targets
target_map = {'normal': 0, 'evil_twin': 1}
df_train['class'] = df_train['class'].map(target_map)
df_test['class'] = df_test['class'].map(target_map)

# Get feature names for the app bundle
feature_names = df_train.drop(columns=['class']).columns.tolist()

X_train = df_train.drop(columns=['class']).to_numpy().astype('float32')
y_train = df_train['class'].to_numpy().astype('int8')
X_test = df_test.drop(columns=['class']).to_numpy().astype('float32')
y_test = df_test['class'].to_numpy().astype('int8')

# Preprocessing (CRITICAL FOR LOGISTIC REGRESSION)
print("Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("PHASE 2: TRAINING ON TRN DATASET")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
start_train = time.time()
lr_model.fit(X_train_scaled, y_train)
print(f" Training Complete in {round(time.time() - start_train, 2)} seconds.")

print("\n" + "=" * 60)
print("PHASE 3: BLIND TESTING ON UNSEEN TST DATASET")
print("=" * 60)

# THIS IS THE TESTING PHASE ON THE UNSEEN DATA
start_test = time.time()
y_pred = lr_model.predict(X_test_scaled)
print(f" Inference Complete in {round(time.time() - start_test, 2)} seconds.")

print("\n" + "-"*30 + "\nFINAL TEST RESULTS\n" + "-"*30)
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Evil Twin']))

print("\n" + "=" * 60)
print("PHASE 4: EXPORTING ASSETS")
print("=" * 60)

# 1. CONFUSION MATRIX
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal', 'Evil Twin'], 
            yticklabels=['Normal', 'Evil Twin'])
plt.title('Final Logistic Regression Confusion Matrix (Test Data)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('logistic_regression_cm.png', dpi=300)
print(" Confusion Matrix saved as: logistic_regression_cm.png")

# 2. EXPORT BUNDLE FOR APP
app_bundle = {
    'model': lr_model,
    'scaler': scaler,
    'features': feature_names
}

joblib.dump(app_bundle, 'evil_twin_champion_model.pkl')
print(" Champion Bundle saved as: evil_twin_champion_model.pkl")