# Scikit-learn 1.7.x | Part 6: Scikit-learn
# Topics: Pipeline, ColumnTransformer, cross-validation, model evaluation
# Run: python 01_sklearn_core.py

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
import pandas as pd
import joblib, tempfile, os

print("Part 6: Scikit-learn Core Patterns")
print("=" * 60)

# ============================================================
# 1. THE GOLDEN RULE: SPLIT FIRST, FIT ONLY ON TRAINING DATA
# ============================================================
print("\n--- 1. Train/Test Split (Data Leakage Prevention) ---")

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names.tolist()

# ALWAYS split before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # preserve class balance in both sets
)
print(f"Train: {X_train.shape}, positive rate: {y_train.mean():.2%}")
print(f"Test:  {X_test.shape},  positive rate: {y_test.mean():.2%}")

# ============================================================
# 2. PIPELINE: THE CORRECT WAY
# ============================================================
print("\n--- 2. Pipeline (Prevents Leakage) ---")

# Without pipeline (WRONG pattern for illustration):
# scaler = StandardScaler()
# X_all_scaled = scaler.fit_transform(X)   # LEAKS test stats into training!
# X_tr, X_te = train_test_split(X_all_scaled)

# With pipeline (CORRECT):
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, C=1.0)),
])

# Cross-validate: each fold trains its own scaler — no leakage
cv_scores = cross_val_score(
    pipe, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
)
print(f"LR Pipeline CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit on full training set, evaluate on test
pipe.fit(X_train, y_train)
y_proba = pipe.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, y_proba):.3f}")

# ============================================================
# 3. COLUMN TRANSFORMER FOR MIXED-TYPE DATA
# ============================================================
print("\n--- 3. ColumnTransformer (Mixed Types) ---")

# Create a mixed-type dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "age":      np.random.uniform(18, 80, n),
    "income":   np.random.exponential(50000, n),
    "tenure":   np.random.randint(0, 120, n),
    "category": np.random.choice(["A", "B", "C", None], n),
    "channel":  np.random.choice(["web", "app", "store"], n),
})
y_mixed = (np.random.rand(n) > 0.4).astype(int)

num_cols = ["age", "income", "tenure"]
cat_cols = ["category", "channel"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols),
])

full_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
])

X_tr, X_te, y_tr, y_te = train_test_split(df, y_mixed, stratify=y_mixed, test_size=0.2)
full_pipe.fit(X_tr, y_tr)
auc = roc_auc_score(y_te, full_pipe.predict_proba(X_te)[:, 1])
print(f"Mixed-type pipeline AUC: {auc:.3f}")

# ============================================================
# 4. RANDOM FOREST: FEATURE IMPORTANCES
# ============================================================
print("\n--- 4. Random Forest and Feature Importances ---")

rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
])
rf_pipe.fit(X_train, y_train)
y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
print(f"Random Forest Test AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")

importances = rf_pipe.named_steps["model"].feature_importances_
top_k = 5
idx = np.argsort(importances)[::-1][:top_k]
print(f"Top {top_k} features:")
for rank, i in enumerate(idx):
    print(f"  {rank+1}. {feature_names[i]}: {importances[i]:.4f}")

# ============================================================
# 5. MODEL EVALUATION: BEYOND ACCURACY
# ============================================================
print("\n--- 5. Evaluation Metrics ---")

y_pred = rf_pipe.predict(X_test)
print(f"Test Report:\n{classification_report(y_test, y_pred, target_names=['malignant','benign'])}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

# ============================================================
# 6. SAVE AND LOAD
# ============================================================
print("\n--- 6. Save and Load Pipeline ---")

tmp = tempfile.mktemp(suffix=".pkl")
joblib.dump(rf_pipe, tmp)
loaded_pipe = joblib.load(tmp)
assert roc_auc_score(y_test, loaded_pipe.predict_proba(X_test)[:, 1]) == roc_auc_score(y_test, y_proba_rf)
print(f"Pipeline saved and loaded, AUC matches: True")
os.remove(tmp)

print("\n" + "=" * 60)
print("Scikit-learn demonstrations complete.")
