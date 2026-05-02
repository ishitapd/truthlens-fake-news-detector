"""
Resume training: train only the MLP and Stacking Ensemble
(base models LR, NB, RF, SVM, GB are already saved).
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
from src.preprocess import clean_text

MODELS_DIR = "models"

# ── Load data ────────────────────────────────────────────
df = pd.read_csv("data/dataset.csv").dropna(subset=["text","label"])
df["clean"] = df["text"].apply(clean_text)
df = df[df["clean"].str.strip() != ""]
X, y = df["clean"].values, df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[DATA] {len(df)} samples loaded.")

# ── Load vectorizer ──────────────────────────────────────
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print("[OK] Vectorizer loaded.")

# ── Train MLP ────────────────────────────────────────────
print("\n[TRAIN] MLP (Deep Learning) ...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu", solver="adam",
    alpha=1e-4, batch_size=32,
    learning_rate="adaptive",
    max_iter=200, early_stopping=True,
    validation_fraction=0.1,
    random_state=42, verbose=False,
)
mlp.fit(X_train_tfidf, y_train)
y_pred = mlp.predict(X_test_tfidf)
y_prob = mlp.predict_proba(X_test_tfidf)[:,1]
print(f"  Accuracy : {accuracy_score(y_test,y_pred):.4f}  AUC: {roc_auc_score(y_test,y_prob):.4f}")
print(f"  F1       : {f1_score(y_test,y_pred):.4f}")
print(f"  CM:\n{confusion_matrix(y_test,y_pred)}")
with open(os.path.join(MODELS_DIR, "mlp_deep_learning.pkl"), "wb") as f:
    pickle.dump(mlp, f)
print("[SAVE] MLP saved.")

mlp_metrics = {
    "model":"MLP (Deep Learning)",
    "accuracy":  round(accuracy_score(y_test,y_pred),4),
    "precision": round(precision_score(y_test,y_pred),4),
    "recall":    round(recall_score(y_test,y_pred),4),
    "f1":        round(f1_score(y_test,y_pred),4),
    "auc_roc":   round(roc_auc_score(y_test,y_prob),4),
}

# ── Stacking Ensemble ────────────────────────────────────
print("\n[ENSEMBLE] Building Stacked Ensemble ...")

def make_vectorizer():
    return TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                           sublinear_tf=True, min_df=2, strip_accents="unicode")

stack_estimators = [
    ("lr",  Pipeline([("tfidf", make_vectorizer()), ("clf", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"))])),
    ("nb",  Pipeline([("tfidf", make_vectorizer()), ("clf", MultinomialNB(alpha=0.1))])),
    ("svm", Pipeline([("tfidf", make_vectorizer()), ("clf", CalibratedClassifierCV(LinearSVC(max_iter=1000, C=0.5)))])),
]
stacking = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=500, C=0.5),
    cv=5, n_jobs=-1, passthrough=False,
)
stacking.fit(X_train, y_train)

y_pred_s = stacking.predict(X_test)
y_prob_s = stacking.predict_proba(X_test)[:,1]
print(f"  Accuracy : {accuracy_score(y_test,y_pred_s):.4f}  AUC: {roc_auc_score(y_test,y_prob_s):.4f}")
print(f"  F1       : {f1_score(y_test,y_pred_s):.4f}")
print(f"  CM:\n{confusion_matrix(y_test,y_pred_s)}")

with open(os.path.join(MODELS_DIR, "stacked_ensemble.pkl"), "wb") as f:
    pickle.dump(stacking, f)
print("[SAVE] Stacked Ensemble saved.")

stack_metrics = {
    "model":"Stacked Ensemble",
    "accuracy":  round(accuracy_score(y_test,y_pred_s),4),
    "precision": round(precision_score(y_test,y_pred_s),4),
    "recall":    round(recall_score(y_test,y_pred_s),4),
    "f1":        round(f1_score(y_test,y_pred_s),4),
    "auc_roc":   round(roc_auc_score(y_test,y_prob_s),4),
}

# ── Merge with existing results ──────────────────────────
existing = pd.read_csv(os.path.join(MODELS_DIR,"results.csv")) if os.path.exists(os.path.join(MODELS_DIR,"results.csv")) else pd.DataFrame()
# Keep only base model rows (not MLP/Ensemble if they existed)
if not existing.empty:
    existing = existing[~existing["model"].isin(["MLP (Deep Learning)","Stacked Ensemble"])]
new_rows = pd.DataFrame([mlp_metrics, stack_metrics])
results_df = pd.concat([existing, new_rows], ignore_index=True)
results_df.to_csv(os.path.join(MODELS_DIR,"results.csv"), index=False)

print("\n[RESULTS] Final Summary:")
print(results_df.to_string(index=False))
print("\n[DONE] All models trained and saved.")
