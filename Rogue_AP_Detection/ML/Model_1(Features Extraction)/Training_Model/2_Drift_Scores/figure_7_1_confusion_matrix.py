# =============================================================================
# FIGURE 7.1: CONFUSION MATRIX VISUALIZATION
# Logistic Regression on 15 Domain-Invariant AWID Features
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

print("=" * 60)
print("LOADING & PREPARING DATA")
print("=" * 60)

train_path = r"C:\Users\hamza\OneDrive\Desktop\Machine Learning Part\99_Main_Work\2_Stuff\domain shift\Evil_Twin-Dataset-Domain-Invariant.csv"
test_path = r"C:\Users\hamza\OneDrive\Desktop\Machine Learning Part\99_Main_Work\2_Stuff\domain shift\Evil_Twin-Dataset-Tst-Domain-Invariant.csv"

print("Loading Data...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# --- HEX TO INT CONVERSION ---
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

# Prepare data
X_train = df_train.drop(columns=['class']).to_numpy().astype('float32')
y_train = df_train['class'].to_numpy().astype('int8')
X_test = df_test.drop(columns=['class']).to_numpy().astype('float32')
y_test = df_test['class'].to_numpy().astype('int8')

# Scale features
print("Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predict
print("Generating predictions...")
y_pred = lr_model.predict(X_test_scaled)

print("\n" + "=" * 60)
print("GENERATING FIGURE 7.1: CONFUSION MATRIX")
print("=" * 60)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extract TP, TN, FP, FN
tn, fp, fn, tp = cm.ravel()

# Create figure with professional styling
fig, ax = plt.subplots(figsize=(10, 8))

# Define custom labels for the heatmap
labels = np.array([[f'TN\n{tn:,}', f'FP\n{fp:,}'],
                    [f'FN\n{fn:,}', f'TP\n{tp:,}']])

# Create heatmap
sns.heatmap(cm, annot=labels, fmt='s', cmap='Blues', cbar=True,
            xticklabels=['Predicted Normal', 'Predicted Evil Twin'],
            yticklabels=['Actual Normal', 'Actual Evil Twin'],
            annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=2, linecolor='black',
            cbar_kws={'label': 'Sample Count'}, ax=ax)

# Enhance appearance
ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title('Figure 7.1: Brain 1 Confusion Matrix\nLogistic Regression on 15 Domain-Invariant AWID Features\n'
             f'(Test Set: {len(y_test):,} samples)', 
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure_7_1_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"[OK] Figure 7.1 saved as: figure_7_1_confusion_matrix.png")

# Print summary stats
print(f"\nConfusion Matrix Summary:")
print(f"  True Negatives (TN):  {tn:,}")
print(f"  False Positives (FP): {fp:,}")
print(f"  False Negatives (FN): {fn:,}")
print(f"  True Positives (TP):  {tp:,}")
print(f"  Total Samples: {len(y_test):,}")

plt.show()
