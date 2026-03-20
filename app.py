import io
import csv
import re

import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

from utils.news_fetcher import fetch_news
from utils.relevance_filter import rule_based_filter, strong_keyword_check
from utils.geo_utils import detect_country_region
from models.relevance_model import train_relevance_model, clean_text
from models.event_model import train_event_model
from utils.risk_engine import calculate_risk

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print("Loading ML models...")
relevance_model = train_relevance_model()
event_model = train_event_model()
print("Models ready.")


def _normalise_risk_level(raw):
    """Extract clean risk label — strips emojis and whitespace reliably."""
    if not raw:
        return "LOW"
    # Remove anything that is not a letter, digit, hyphen or space
    clean = re.sub(r'[^A-Za-z0-9\-\s]', '', str(raw)).strip()
    valid = {"HIGH", "MEDIUM-HIGH", "MEDIUM", "LOW-MEDIUM", "LOW"}
    # Direct match
    if clean in valid:
        return clean
    # First word match  e.g. "HIGH " -> "HIGH"
    first = clean.split()[0] if clean else "LOW"
    return first if first in valid else "LOW"


def _run_pipeline(df):
    if df.empty:
        return df

    df["text"] = (
        df["title"].fillna("") + " " +
        df.get("description", pd.Series([""] * len(df))).fillna("") + " " +
        df.get("content", pd.Series([""] * len(df))).fillna("")
    )
    df["clean_text"] = df["text"].apply(clean_text)
    df["rule_pass"] = df["clean_text"].apply(rule_based_filter)
    df = df[df["rule_pass"] == True].copy()
    if df.empty:
        return df

    df["confidence"] = relevance_model.predict_proba(df["clean_text"])[:, 1]
    df["strong_signal"] = df["clean_text"].apply(strong_keyword_check)
    df = df[
        (df["confidence"] > 0.7) |
        ((df["confidence"] > 0.65) & (df["strong_signal"] == True))
    ]
    if df.empty:
        return df

    df["event_type"] = event_model.predict(df["clean_text"])

    risk_data = df["event_type"].apply(calculate_risk)
    df["risk_score"] = risk_data.apply(lambda x: x[0])
    df["risk_level_raw"] = risk_data.apply(lambda x: x[1])
    df["risk_level"] = df["risk_level_raw"].apply(_normalise_risk_level)

    df[["country", "region"]] = df["text"].apply(
        lambda t: pd.Series(detect_country_region(t))
    )

    if "title" in df.columns:
        df = df.drop_duplicates(subset="title")

    # Debug — visible in terminal
    print("\n=== Pipeline Sample ===")
    for _, row in df.head(3).iterrows():
        print("  event_type :", row.get("event_type"))
        print("  risk_level :", row.get("risk_level"))
        print("  risk_score :", row.get("risk_score"))
        print("  ---")
    print("======================\n")

    return df


def _df_to_records(df):
    if df.empty:
        return []
    keep = ["title", "source", "country", "region", "event_type",
            "risk_level", "risk_score", "published_at", "url"]
    out_cols = [c for c in keep if c in df.columns]
    records = df[out_cols].copy()
    records["risk_score"] = records["risk_score"].round(2)
    for col in ["source", "country", "region", "published_at", "url"]:
        if col not in records.columns:
            records[col] = ""
    records = records.fillna("")
    return records.to_dict(orient="records")


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    page_size = int(body.get("page_size", 20))
    if not query:
        return jsonify({"error": "Query is required"}), 400
    df = fetch_news(query, page_size=page_size)
    if df.empty:
        return jsonify({"results": [], "total": 0})
    enriched = _run_pipeline(df)
    records = _df_to_records(enriched)
    return jsonify({"results": records, "total": len(records)})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": "Could not parse CSV: " + str(e)}), 400
    if "title" not in df.columns:
        return jsonify({"error": "CSV must contain a 'title' column"}), 400
    for col in ["description", "content"]:
        if col not in df.columns:
            df[col] = ""
    enriched = _run_pipeline(df)
    records = _df_to_records(enriched)
    return jsonify({"results": records, "total": len(records)})


@app.route("/api/export", methods=["POST"])
def export_csv():
    body = request.get_json(silent=True) or {}
    results = body.get("results", [])
    if not results:
        return jsonify({"error": "No data to export"}), 400
    output = io.StringIO()
    fieldnames = ["title", "source", "country", "region",
                  "event_type", "risk_level", "risk_score", "published_at", "url"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    csv_bytes = output.getvalue().encode("utf-8")
    return Response(csv_bytes, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=risk_report.csv"})


if __name__ == "__main__":
    app.run(debug=True)