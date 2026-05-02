"""
Regenerate results.csv using all saved models.
Loads each model, runs on test set, computes metrics.
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from src.preprocess import clean_text

MODELS_DIR = "models"

# Load data
df = pd.read_csv("data/dataset.csv").dropna(subset=["text", "label"])
df["clean"] = df["text"].apply(clean_text)
df = df[df["clean"].str.strip() != ""]
X, y = df["clean"].values, df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[DATA] {len(df)} samples, {len(X_test)} test samples")

# Load vectorizer
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
X_test_tfidf = vectorizer.transform(X_test)
print("[OK] Vectorizer loaded")

# Base models (TF-IDF already applied)
base_models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Naive Bayes":         "naive_bayes.pkl",
    "Random Forest":       "random_forest.pkl",
    "Linear SVM":          "linear_svm.pkl",
    "Gradient Boosting":   "gradient_boosting.pkl",
    "MLP (Deep Learning)": "mlp_deep_learning.pkl",
}

results = []
for name, fname in base_models.items():
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        print(f"[SKIP] {name} not found")
        continue
    with open(path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test_tfidf)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        auc = round(roc_auc_score(y_test, y_prob), 4)
    else:
        auc = None
    row = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   auc,
    }
    results.append(row)
    print(f"[OK] {name}: acc={row['accuracy']}  f1={row['f1']}")

# Stacking ensemble (operates on raw text)
stack_path = os.path.join(MODELS_DIR, "stacked_ensemble.pkl")
if os.path.exists(stack_path):
    with open(stack_path, "rb") as f:
        stacking = pickle.load(f)
    y_pred_s = stacking.predict(X_test)
    y_prob_s = stacking.predict_proba(X_test)[:, 1]
    row_s = {
        "model":     "Stacked Ensemble",
        "accuracy":  round(accuracy_score(y_test, y_pred_s), 4),
        "precision": round(precision_score(y_test, y_pred_s, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred_s, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred_s, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob_s), 4),
    }
    results.append(row_s)
    print(f"[OK] Stacked Ensemble: acc={row_s['accuracy']}  f1={row_s['f1']}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODELS_DIR, "results.csv"), index=False)
print("\n[DONE] results.csv updated:")
print(results_df.to_string(index=False))
