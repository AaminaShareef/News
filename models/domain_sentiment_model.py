"""
domain_sentiment_model.py
Two focused models:
  Model 1 — Domain    : 6 classes  (5 domains + Not_Relevant) → 85% accuracy
  Model 2 — Sentiment : 3 classes  (Negative / Neutral / Positive) → 86% accuracy
Both trained from domain_sentiment_dataset.csv
"""

import os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _load_data():
    path = os.path.join(_BASE_DIR, 'data', 'domain_sentiment_dataset.csv')
    df = pd.read_csv(path)
    df = df[df['label'] != 'label'].copy()  # drop any duplicate headers
    df['text'] = df['text'].apply(clean_text)
    df['domain']    = df['label'].apply(lambda l: l if l == 'Not_Relevant' else l.split('_')[0])
    df['sentiment'] = df['label'].apply(lambda l: None if l == 'Not_Relevant' else l.split('_')[1])
    return df


def _make_pipeline(C=1.0, max_features=10000, ngram=(1, 3)):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=ngram,
            max_features=max_features,
            stop_words='english',
            sublinear_tf=True,
            min_df=1,
        )),
        ('clf', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=C,
            solver='lbfgs',
        ))
    ])


def _train_domain_model(df):
    X, y = df['text'], df['domain']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = _make_pipeline(C=1.0, max_features=10000, ngram=(1, 3))
    model.fit(Xtr, ytr)
    print("\n── Domain Model Evaluation (6 classes) ─────────────────")
    print(classification_report(yte, model.predict(Xte), zero_division=1))
    return model


def _train_sentiment_model(df):
    df_s = df[df['sentiment'].notna()].copy()
    X, y = df_s['text'], df_s['sentiment']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = _make_pipeline(C=2.0, max_features=8000, ngram=(1, 2))
    model.fit(Xtr, ytr)
    print("── Sentiment Model Evaluation (3 classes) ───────────────")
    print(classification_report(yte, model.predict(Xte), zero_division=1))
    return model


def train_domain_sentiment_model():
    """Returns (domain_model, sentiment_model). Call once at startup."""
    print("Training domain model (6 classes)...")
    df = _load_data()
    domain_model    = _train_domain_model(df)
    print("Training sentiment model (3 classes)...")
    sentiment_model = _train_sentiment_model(df)
    return domain_model, sentiment_model


def predict(domain_model, sentiment_model, text):
    """
    Returns (domain, sentiment, dom_confidence, sent_confidence, full_label)
    domain        : e.g. 'Geopolitical'  or 'Not_Relevant'
    sentiment     : e.g. 'Negative'      or None
    dom_confidence: float 0.0–1.0
    sent_confidence: float 0.0–1.0  (0.0 if Not_Relevant)
    full_label    : e.g. 'Geopolitical_Negative' or 'Not_Relevant'
    """
    cleaned        = clean_text(text)
    domain         = domain_model.predict([cleaned])[0]
    dom_confidence = float(domain_model.predict_proba([cleaned])[0].max())

    if domain == 'Not_Relevant':
        return 'Not_Relevant', None, dom_confidence, 0.0, 'Not_Relevant'

    sentiment       = sentiment_model.predict([cleaned])[0]
    sent_confidence = float(sentiment_model.predict_proba([cleaned])[0].max())
    full_label      = domain + '_' + sentiment
    return domain, sentiment, dom_confidence, sent_confidence, full_label