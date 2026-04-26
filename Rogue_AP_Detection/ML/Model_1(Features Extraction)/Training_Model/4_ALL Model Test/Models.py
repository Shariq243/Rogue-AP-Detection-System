# =============================================================================
# FYP ULTIMATE COMPARISON: 8 ML MODELS + 3 DEEP LEARNING MODELS
# Optimized for Google Colab T4 GPU (WITH HEX FIX & 60 EPOCHS)
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

import time, gc, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, LSTM, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("PHASE 1: LOADING & CLEANING DATA")
print("=" * 60)

# 1. DATA LOADING
train_path = "/content/drive/My Drive/Machine_Learning1/Evil_Twin-Dataset-Domain-Invariant.csv"
test_path = "/content/drive/My Drive/Machine_Learning1/Evil_Twin-Dataset-Tst-Domain-Invariant.csv"

print("Loading datasets...")
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

print("Converting to tensors...")
X_train = df_train.drop(columns=['class']).to_numpy().astype('float32')
y_train = df_train['class'].to_numpy().astype('int8')
X_test = df_test.drop(columns=['class']).to_numpy().astype('float32')
y_test = df_test['class'].to_numpy().astype('int8')

# Scaling is mandatory for DL and Linear Models
print("Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

print("\n" + "=" * 60)
print("PHASE 2: STANDARD ML MODELS")
print("=" * 60)

ml_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gaussian NB": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "LinearSVC": LinearSVC(dual=False, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost (GPU)": xgb.XGBClassifier(n_estimators=100, tree_method='hist', device='cuda', random_state=42)
}

for name, model in ml_models.items():
    print(f"Training {name}...")
    st = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - st

    st_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - st_test

    results.append({
        "Model": name, "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds), "Recall": recall_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds), "Train Time": train_time, "Test Time": test_time
    })
    gc.collect()

# KNN Bypass (100k)
print("Training KNN (100k Stratified Sample)...")
idx = np.random.choice(len(y_train), 100000, replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
st = time.time()
knn.fit(X_train_scaled[idx], y_train[idx])
train_time = time.time() - st
st_test = time.time()
knn_preds = knn.predict(X_test_scaled)
results.append({
    "Model": "KNN (100k)", "Accuracy": accuracy_score(y_test, knn_preds),
    "Precision": precision_score(y_test, knn_preds), "Recall": recall_score(y_test, knn_preds),
    "F1-Score": f1_score(y_test, knn_preds), "Train Time": train_time, "Test Time": time.time() - st_test
})
del knn, knn_preds
gc.collect()

print("\n" + "=" * 60)
print("PHASE 3: DEEP LEARNING ARCHITECTURES (GPU)")
print("=" * 60)

# Reshape for DL: (Samples, Timesteps=1, Features=15)
X_train_dl = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_dl = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
in_shape = (1, X_train_scaled.shape[1])

def run_dl(name, model):
    print(f"\nTraining {name}...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st = time.time()

    # UPGRADED TO 60 EPOCHS
    # patience=5 means if the model doesn't improve for 5 straight epochs, it aborts to save time.
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train_dl, y_train, epochs=60, batch_size=2048, validation_split=0.1, callbacks=[early_stop], verbose=1)
    tr_t = time.time() - st

    st_test = time.time()
    p = (model.predict(X_test_dl, batch_size=2048) > 0.5).astype(int).flatten()
    test_time = time.time() - st_test

    results.append({
        "Model": name, "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p), "Recall": recall_score(y_test, p),
        "F1-Score": f1_score(y_test, p), "Train Time": tr_t, "Test Time": test_time
    })
    tf.keras.backend.clear_session()
    gc.collect()

# 1. 1D-CNN
run_dl("1D-CNN", Sequential([
    Input(shape=in_shape), Conv1D(32, 1, activation='relu'),
    Flatten(), Dense(1, activation='sigmoid')
]))

# 2. LSTM
run_dl("LSTM", Sequential([
    Input(shape=in_shape), LSTM(32), Dense(1, activation='sigmoid')
]))

# 3. Transformer (Attention)
inputs = Input(shape=in_shape)
x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
x = LayerNormalization()(x + inputs)
x = GlobalAveragePooling1D()(x)
transformer_model = Model(inputs, Dense(1, activation='sigmoid')(x))
run_dl("Transformer", transformer_model)

print("\n" + "=" * 60)
print("FINAL 11-MODEL COMPARISON RESULTS")
print("=" * 60)

final_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

# Format for display
display_df = final_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    display_df[col] = (display_df[col] * 100).round(4).astype(str) + "%"
for col in ["Train Time", "Test Time"]:
    display_df[col] = display_df[col].round(2).astype(str) + "s"

print(display_df.to_markdown())

# Save to Drive
csv_path = "/content/drive/My Drive/Machine_Learning1/FULL_11_MODEL_COMPARISON.csv"
final_df.to_csv(csv_path, index=False)
print(f"\n Results saved successfully to: {csv_path}")