# TruthLens — Fake News Detection System

> **End-to-end NLP pipeline** stacking 6 machine learning architectures into a unified ensemble for real-time news authenticity classification.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)

---

## Project Overview

**TruthLens** is an end-to-end fake news detection system built from scratch — covering the full machine learning lifecycle from raw text preprocessing to a live deployed web application. It demonstrates proficiency in NLP, ensemble machine learning, and full-stack Python deployment.

### Key Highlights

- **NLP Pipeline**: Text cleaning → URL/HTML removal → stopword filtering → Porter stemming (NLTK)
- **TF-IDF Vectorization**: 10,000-feature bag-of-words with unigram + bigram n-grams and sublinear TF scaling
- **6 Base ML Models**: Logistic Regression, Naïve Bayes, Random Forest, Linear SVM, Gradient Boosting, MLP (Neural Network)
- **Stacked Ensemble**: Meta-learner (Logistic Regression) trained on cross-validated predictions of 3 base pipelines
- **Live Flask Web App**: Real-time article classification with per-model confidence scores and visual breakdown

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| NLP | NLTK, TF-IDF (Scikit-learn) |
| ML Models | Scikit-learn (LR, NB, RF, SVM, GB, MLP) |
| Ensemble | `StackingClassifier` with meta-learner |
| Web Backend | Flask |
| Frontend | HTML5 · CSS3 · Vanilla JS (glassmorphism UI) |
| Serialization | Pickle |

---

## Architecture

```
Raw Text Input
     │
     ▼
NLP Preprocessing (NLTK)
  ├─ Lowercase
  ├─ Remove URLs / HTML
  ├─ Strip punctuation
  ├─ Remove stopwords
  └─ Porter Stemming
     │
     ▼
TF-IDF Vectorization
  └─ 10,000 features, n-gram(1,2), sublinear_tf
     │
     ▼
Base Models (6 architectures)
  ├─ Logistic Regression
  ├─ Naïve Bayes
  ├─ Random Forest (200 trees)
  ├─ Linear SVM (calibrated)
  ├─ Gradient Boosting
  └─ MLP Neural Network (128→64 hidden layers)
     │
     ▼
Stacking Meta-Learner
  └─ Logistic Regression on OOF predictions (5-fold CV)
     │
     ▼
Verdict: REAL / FAKE + Confidence Score
```

---

## Model Performance

All models were evaluated on a 20% held-out test set (400 samples):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Naïve Bayes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Linear SVM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gradient Boosting | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| MLP (Deep Learning) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Stacked Ensemble** ⭐ | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

> *Note: Perfect scores reflect the synthetic dataset used for demonstration. The full pipeline is designed to work with real-world datasets (e.g., LIAR, FakeNewsNet).*

---

## Project Structure

```
fake news/
├── app.py                  # Flask web application (REST API + frontend serving)
├── generate_data.py        # Synthetic dataset generator (2,000 samples)
├── resume_training.py      # Train MLP + Stacking Ensemble only
├── fix_results.py          # Regenerate results.csv from saved models
├── run_training.py         # One-shot: generate data → train → serve
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # NLP cleaning pipeline (NLTK)
│   └── train.py            # Full training pipeline (all 6 models + ensemble)
│
├── data/
│   └── dataset.csv         # 2,000-sample balanced fake/real dataset
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── linear_svm.pkl
│   ├── gradient_boosting.pkl
│   ├── mlp_deep_learning.pkl
│   ├── stacked_ensemble.pkl
│   └── results.csv
│
├── templates/
│   └── index.html          # Glassmorphism dark-mode UI
│
└── static/
    ├── css/style.css       # Full design system (animations, glassmorphism)
    └── js/app.js           # Frontend logic (fetch API, dynamic rendering)
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd "fake news"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Generate data + train models

```bash
python run_training.py
```

Or step by step:
```bash
python generate_data.py     # Creates data/dataset.csv
python -m src.train         # Trains all models → models/
```

### 3. Launch the web app

```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web interface |
| `/predict` | POST | Classify article text |
| `/api/status` | GET | Check loaded models |
| `/api/results` | GET | Get model performance metrics |

### `/predict` request/response

```json
// POST /predict
{ "text": "News article text here..." }

// Response
{
  "final_label": 1,
  "final_label_text": "FAKE",
  "final_confidence": 99.3,
  "ensemble_label": 1,
  "ensemble_confidence": 99.3,
  "majority_vote": 1,
  "vote_confidence": 100.0,
  "word_count": 77,
  "cleaned_length": 49,
  "model_predictions": {
    "Logistic Regression": { "label": 1, "confidence": 99.2 },
    "Naive Bayes":         { "label": 1, "confidence": 100.0 },
    ...
  }
}
```

---

## Extending with Real Data

To use a real dataset (e.g., [LIAR](https://huggingface.co/datasets/liar) or [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)):

1. Replace `data/dataset.csv` with your dataset (columns: `text`, `label` where `0=Real, 1=Fake`)
2. Run `python -m src.train` to retrain all models
3. Restart `python app.py`

---

## Skills Demonstrated

- **NLP Pipeline Engineering** — text normalization, tokenization, stemming
- **Feature Engineering** — TF-IDF with n-grams and sublinear scaling
- **Ensemble ML** — stacking classifier with cross-validated out-of-fold predictions
- **Model Evaluation** — accuracy, precision, recall, F1, AUC-ROC, confusion matrix
- **Full-Stack Deployment** — Flask REST API + responsive frontend
- **Software Engineering** — modular code structure, pickle serialization, error handling

---

*Built independently as a portfolio project demonstrating end-to-end ML system design.*
