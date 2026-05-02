"""
Model Training Pipeline
- TF-IDF vectorization
- Individual ML models: Logistic Regression, Naive Bayes, Random Forest, SVM, Gradient Boosting
- Neural MLP (deep learning analog to LSTM for text)
- Stacked Ensemble (meta-learner on top of base model predictions)
- Saves all models and vectorizer to models/
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import clean_text

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. Load & Preprocess ──────────────────────────────────────────────────────

def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.strip() != ""]
    print(f"[DATA] Loaded {len(df)} samples after preprocessing.")
    return df


# ── 2. TF-IDF Vectorization ───────────────────────────────────────────────────

def build_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log normalization
        min_df=2,
        strip_accents="unicode",
    )


# ── 3. Base Models ────────────────────────────────────────────────────────────

def get_base_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
        "Linear SVM":          CalibratedClassifierCV(LinearSVC(max_iter=2000, C=0.5)),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        "MLP (Deep Learning)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate="adaptive",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        ),
    }


# ── 4. Stacked Ensemble ───────────────────────────────────────────────────────

def build_stacking_ensemble(base_estimators_list):
    """
    Base estimators are (name, pipeline) tuples.
    Meta-learner: Logistic Regression
    """
    meta_learner = LogisticRegression(max_iter=1000, C=0.5)
    stacking = StackingClassifier(
        estimators=base_estimators_list,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stacking


# ── 5. Evaluate a Model ───────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float("nan")

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}   AUC-ROC : {auc:.4f}")
    print(f"  Precision: {prec:.4f}  Recall  : {rec:.4f}  F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
    }


# ── 6. Train & Save Everything ───────────────────────────────────────────────

def train():
    # Load data
    df = load_data()
    X = df["clean"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorizer
    print("\n[TFIDF] Fitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Save vectorizer
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print("[SAVE] Vectorizer saved.")

    results = []
    trained_pipelines = []  # for stacking

    # Train base models
    base_models = get_base_models()
    for name, clf in base_models.items():
        print(f"\n[TRAIN] Training: {name} ...")
        clf.fit(X_train_tfidf, y_train)
        metrics = evaluate(name, clf, X_test_tfidf, y_test)
        results.append(metrics)

        # Save individual model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        print(f"[SAVE] Saved -> {path}")

        # Keep for stacking (use Pipeline with vectorizer pre-applied)
        trained_pipelines.append((safe_name, clf))

    # ── Stacking Ensemble ──
    print("\n\n[ENSEMBLE] Building Stacked Ensemble...")
    # For stacking on raw text, build full pipelines
    stack_estimators = []
    for name_key, _ in [
        ("logistic_regression",  LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
        ("naive_bayes",          MultinomialNB(alpha=0.1)),
        ("linear_svm",           CalibratedClassifierCV(LinearSVC(max_iter=1000, C=0.5))),
    ]:
        clf_new = dict(get_base_models()).get(
            {
                "logistic_regression": "Logistic Regression",
                "naive_bayes":          "Naive Bayes",
                "linear_svm":           "Linear SVM",
            }[name_key]
        )
        pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", clf_new)])
        stack_estimators.append((name_key, pipe))

    stacking = build_stacking_ensemble(stack_estimators)
    stacking.fit(X_train, y_train)

    # Evaluate stacking (raw text → predict)
    y_pred_stack = stacking.predict(X_test)
    y_prob_stack = stacking.predict_proba(X_test)[:, 1]

    stack_metrics = {
        "model": "Stacked Ensemble",
        "accuracy":  round(accuracy_score(y_test, y_pred_stack), 4),
        "precision": round(precision_score(y_test, y_pred_stack), 4),
        "recall":    round(recall_score(y_test, y_pred_stack), 4),
        "f1":        round(f1_score(y_test, y_pred_stack), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob_stack), 4),
    }
    print(f"\n{'='*55}")
    print("  Stacked Ensemble")
    print(f"{'='*55}")
    print(f"  Accuracy : {stack_metrics['accuracy']}   AUC-ROC : {stack_metrics['auc_roc']}")
    print(f"  Precision: {stack_metrics['precision']}  Recall  : {stack_metrics['recall']}  F1: {stack_metrics['f1']}")
    results.append(stack_metrics)

    # Save stacking model
    with open(os.path.join(MODELS_DIR, "stacked_ensemble.pkl"), "wb") as f:
        pickle.dump(stacking, f)
    print("[SAVE] Stacked Ensemble saved.")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODELS_DIR, "results.csv"), index=False)
    print(f"\n[RESULTS] Summary:\n{results_df.to_string(index=False)}")
    print("\n[DONE] Training complete. All models saved to models/")
    return results_df


if __name__ == "__main__":
    train()
