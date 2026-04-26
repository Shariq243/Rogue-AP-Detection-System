"""
Figure 7.4: KS-Test Domain Drift Scores per Feature
Bar chart showing KS statistic for each of the 25 features with 0.15 threshold line
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import ks_2samp

# =============================================================================
# LOAD DATA
# =============================================================================

from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

print("Loading training and test datasets...")
evil_twin_trn_path = BASE_DIR / "data" / "Evil_Twin-Dataset.csv"
evil_twin_tst_path = BASE_DIR / "data" / "Evil_Twin-Dataset-Tst-Preprocessed.csv"


df_train = pd.read_csv(evil_twin_trn_path)
df_test = pd.read_csv(evil_twin_tst_path)

print("Converting hexadecimal features...")
for df in [df_train, df_test]:
    df['wlan.fc.type_subtype'] = df['wlan.fc.type_subtype'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else x)
    df['wlan.fc.ds'] = df['wlan.fc.ds'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else x)

X_train = df_train.drop(columns=['class'])
X_test = df_test.drop(columns=['class'])

# =============================================================================
# CALCULATE KS STATISTICS FOR ALL 25 FEATURES
# =============================================================================

print(f"\nCalculating KS statistics for {len(X_train.columns)} features...")
print("="*70)

ks_results = {}
drifting_features = []
stable_features = []
DRIFT_THRESHOLD = 0.15

for col in X_train.columns:
    ks_stat, p_value = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
    ks_results[col] = {
        'ks_statistic': float(ks_stat),
        'p_value': float(p_value),
        'is_drifting': bool(ks_stat > DRIFT_THRESHOLD)
    }
    
    status = "DRIFT" if ks_stat > DRIFT_THRESHOLD else "STABLE"
    drifting_features.append(col) if ks_stat > DRIFT_THRESHOLD else stable_features.append(col)
    print(f"{status:8} | {col:40} | KS: {ks_stat:.4f}")

print("="*70)
print(f"Total Stable Features: {len(stable_features)} / {len(X_train.columns)}")
print(f"Total Drifting Features: {len(drifting_features)} / {len(X_train.columns)}")

# Save results
with open('ks_statistics.json', 'w') as f:
    json.dump(ks_results, f, indent=2)
print("\n[OK] KS statistics saved to: ks_statistics.json")

# =============================================================================
# CREATE BAR CHART
# =============================================================================

# Prepare data for plotting
features = list(ks_results.keys())
ks_stats = [ks_results[f]['ks_statistic'] for f in features]
colors = ['#e74c3c' if ks_results[f]['is_drifting'] else '#2ecc71' for f in features]

# Create figure
fig, ax = plt.subplots(figsize=(16, 8))

# Create bars
bars = ax.bar(range(len(features)), ks_stats, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Add threshold line
ax.axhline(y=DRIFT_THRESHOLD, color='orange', linestyle='--', linewidth=2.5, label=f'Drift Threshold ({DRIFT_THRESHOLD})')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, ks_stats)):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.003,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize chart
ax.set_xlabel('Features', fontsize=13, fontweight='bold')
ax.set_ylabel('KS Statistic (Domain Drift Score)', fontsize=13, fontweight='bold')
ax.set_title('Figure 7.4: KS-Test Domain Drift Scores per Feature\nEvil Twin Detection - Training vs Test Distribution Shift',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
ax.set_ylim([0, max(ks_stats) * 1.15])
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add legend with color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label=f'Stable Features ({len(stable_features)})'),
    Patch(facecolor='#e74c3c', edgecolor='black', label=f'Drifting Features ({len(drifting_features)})'),
    plt.Line2D([0], [0], color='orange', linewidth=2.5, linestyle='--', label=f'Threshold ({DRIFT_THRESHOLD})')
]
ax.legend(handles=legend_elements, fontsize=12, loc='upper right', framealpha=0.95)

plt.tight_layout()
plt.savefig('figure_7_4_ks_drift_scores.png', dpi=300, bbox_inches='tight')
print("[OK] Figure 7.4 saved as: figure_7_4_ks_drift_scores.png")

# Print summary
print("\n" + "="*70)
print("KS-TEST DOMAIN DRIFT ANALYSIS SUMMARY")
print("="*70)
print(f"\nSTABLE FEATURES (KS <= {DRIFT_THRESHOLD}):")
for i, feat in enumerate(stable_features, 1):
    ks_val = ks_results[feat]['ks_statistic']
    print(f"  {i:2}. {feat:40} | KS: {ks_val:.4f}")

print(f"\nDRIFTING FEATURES (KS > {DRIFT_THRESHOLD}):")
for i, feat in enumerate(drifting_features, 1):
    ks_val = ks_results[feat]['ks_statistic']
    print(f"  {i:2}. {feat:40} | KS: {ks_val:.4f}")

print("="*70)

plt.show()
