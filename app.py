"""
Flask Web Application — Fake News Detection System
Real-time article classification with confidence scores and model comparison.
Auto-trains models on first startup if not found (supports cloud deployment).
"""

import os
import pickle
import csv
import subprocess
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
from src.preprocess import clean_text

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = "models"

MODEL_FILES = {
    "Logistic Regression":  "logistic_regression.pkl",
    "Naive Bayes":          "naive_bayes.pkl",
    "Random Forest":        "random_forest.pkl",
    "Linear SVM":           "linear_svm.pkl",
    "Gradient Boosting":    "gradient_boosting.pkl",
    "MLP (Deep Learning)":  "mlp_deep_learning.pkl",
}

vectorizer     = None
base_models    = {}
stacking_model = None
results_summary = []


# ── Auto-train if models are missing ──────────────────────────────────────────

def ensure_models_exist():
    """Train models on first deployment if pickle files are absent."""
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        return  # already trained

    print("[STARTUP] Model files not found — running training pipeline...")

    # 1. Generate dataset if missing
    if not os.path.exists(os.path.join("data", "dataset.csv")):
        print("[STARTUP] Generating dataset...")
        subprocess.run([sys.executable, "generate_data.py"], check=True)

    # 2. Train all models
    print("[STARTUP] Training models...")
    subprocess.run([sys.executable, "-m", "src.train"], check=True)
    print("[STARTUP] Training complete.")


# ── Load Models ───────────────────────────────────────────────────────────────

def load_models():
    global vectorizer, base_models, stacking_model, results_summary

    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                base_models[name] = pickle.load(f)

    stack_path = os.path.join(MODELS_DIR, "stacked_ensemble.pkl")
    if os.path.exists(stack_path):
        with open(stack_path, "rb") as f:
            stacking_model = pickle.load(f)

    results_path = os.path.join(MODELS_DIR, "results.csv")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            reader = csv.DictReader(f)
            results_summary = list(reader)

    print(f"[OK] Loaded {len(base_models)} base models, "
          f"ensemble={'yes' if stacking_model else 'no'}, "
          f"results={len(results_summary)} rows")


# Run on startup
ensure_models_exist()
load_models()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    models_loaded = bool(vectorizer and base_models)
    return render_template("index.html",
                           models_loaded=models_loaded,
                           results=results_summary)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    article_text = data.get("text", "").strip()

    if not article_text:
        return jsonify({"error": "No text provided."}), 400

    if not vectorizer or not base_models:
        return jsonify({"error": "Models not loaded. Please wait and retry."}), 503

    cleaned = clean_text(article_text)
    if not cleaned:
        return jsonify({"error": "Text too short or unprocessable after cleaning."}), 400

    tfidf_vec = vectorizer.transform([cleaned])
    word_count = len(article_text.split())

    predictions = {}
    for name, model in base_models.items():
        pred = int(model.predict(tfidf_vec)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(tfidf_vec)[0][pred])
        else:
            prob = 1.0
        predictions[name] = {"label": pred, "confidence": round(prob * 100, 1)}

    # Stacked ensemble (operates on raw cleaned text)
    ensemble_pred = None
    ensemble_conf = None
    if stacking_model:
        ep = int(stacking_model.predict([cleaned])[0])
        if hasattr(stacking_model, "predict_proba"):
            ep_prob = float(stacking_model.predict_proba([cleaned])[0][ep])
        else:
            ep_prob = 1.0
        ensemble_pred = ep
        ensemble_conf = round(ep_prob * 100, 1)

    # Majority vote from base models
    labels = [v["label"] for v in predictions.values()]
    majority = int(np.round(np.mean(labels)))
    vote_confidence = round((labels.count(majority) / len(labels)) * 100, 1)

    final_label = ensemble_pred if ensemble_pred is not None else majority
    final_conf  = ensemble_conf if ensemble_conf is not None else vote_confidence

    return jsonify({
        "final_label":        final_label,
        "final_label_text":   "FAKE" if final_label == 1 else "REAL",
        "final_confidence":   final_conf,
        "ensemble_label":     ensemble_pred,
        "ensemble_confidence": ensemble_conf,
        "majority_vote":      majority,
        "vote_confidence":    vote_confidence,
        "word_count":         word_count,
        "cleaned_length":     len(cleaned.split()),
        "model_predictions":  predictions,
    })


@app.route("/api/results")
def api_results():
    return jsonify(results_summary)


@app.route("/api/status")
def api_status():
    return jsonify({
        "vectorizer_loaded":   vectorizer is not None,
        "base_models_loaded":  list(base_models.keys()),
        "ensemble_loaded":     stacking_model is not None,
        "results_available":   len(results_summary) > 0,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
