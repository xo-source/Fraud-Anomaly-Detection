# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# LOAD DATA
# =========================
creditfile = Path("creditcard.csv")
df = pd.read_csv(creditfile)

# =========================
# PREPROCESSING
# =========================
y = df["Class"]

df["Amount"] = np.log1p(df["Amount"])
df["Time"] = np.log1p(df["Time"])

X = df.drop(columns=["Class"])

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL 1: ISOLATION FOREST
# =========================
iso = IsolationForest(
    n_estimators=200,
    contamination=0.009,
    max_samples=256,
    random_state=42
)

iso.fit(X_train_scaled)

iso_pred = iso.predict(X_test_scaled)
iso_pred = np.where(iso_pred == -1, 1, 0)

# =========================
# EVALUATION (ISOLATION FOREST)
# =========================
print("=== Isolation Forest ===")
print(confusion_matrix(y_test, iso_pred))
print(classification_report(y_test, iso_pred))

# =========================
# MODEL 2: PCA ANOMALY SCORE (EXPERIMENT)
# =========================
pca = PCA(n_components=10)
pca.fit(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)
X_test_recon = pca.inverse_transform(X_test_pca)

pca_scores = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)

threshold = np.percentile(pca_scores, 99)
pca_pred = (pca_scores > threshold).astype(int)

# =========================
# EVALUATION (PCA)
# =========================
print("\n=== PCA Anomaly Detection ===")
print(confusion_matrix(y_test, pca_pred))
print(classification_report(y_test, pca_pred))