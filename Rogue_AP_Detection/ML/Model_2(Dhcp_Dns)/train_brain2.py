import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("[*] Loading Mininet Dataset...")
df = pd.read_csv("Final_10k_Dataset.csv")

# 1. Enforce the exact 9000 row limit
if len(df) > 9000:
    df = df.sample(n=9000, random_state=42)

active_features = ['dhcp_offer_count', 'dns_queries_total']

print(f"[*] Extracting active features from {len(df)} rows...")
X = df[active_features].fillna(0)
y = df['label']

# 2. Strict split: 7000 Train, 2000 Test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7000, test_size=2000, random_state=42)

print(f"[*] Training Brain 2 on {len(X_train)} rows. Testing on {len(X_test)} rows...")
rf_active = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_active.fit(X_train, y_train)

# 3. Test and Save
preds = rf_active.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"[+] Brain 2 Accuracy on Mininet Data: {acc * 100:.2f}%")

joblib.dump(rf_active, "dhcp_dns_rf.pkl")
print("[+] SUCCESS: 'dhcp_dns_rf.pkl' is ready for Phase 3!")