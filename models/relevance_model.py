import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def train_relevance_model():
    df = pd.read_csv(os.path.join(_BASE_DIR, "data", "relevance_dataset.csv"))

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred, zero_division=1))

    return model