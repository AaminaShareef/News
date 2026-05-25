"""
app.py  —  ProcureGuard Flask application
==========================================

NEW LOGIC (per Supply Chain Disruption Prediction System docs)
--------------------------------------------------------------
Step 1  collect_training_data.py
        → Fetches news INDUSTRY-WISE using broad domain queries (no hardcoded keywords)
        → Assigns weak auto-labels (fetch-domain + heuristics)
        → Saves every record to  data/training/train.json  (verified=false)
        → Human reviewer sets verified=true for confirmed records

Step 2  models/domain_sentiment_model.py
        → Loads ONLY verified records from train.json
        → Embeds them with DistilBERT (all-MiniLM-L6-v2, 384-dim)
        → Fits BERTopic for multi-domain topic assignment (up to 3 domains/article)
        → Returns domain_model + sentiment_model used in the pipeline

Step 3  _run_pipeline()  (this file)
        → Embeds incoming articles with the same DistilBERT model
        → Assigns up to 3 supply-chain domains (primary + secondary)
        → Runs cosine-similarity semantic grouping (threshold 0.82)
        → Scores risk using the weighted formula from the documentation
        → Detects country/region via geo_utils
        → Returns enriched records for the dashboard
"""

import io, csv
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

from utils.news_fetcher  import fetch_news
from utils.geo_utils     import detect_country_region
from utils.risk_engine   import calculate_risk, get_all_industries
from models.domain_sentiment_model import (
    train_domain_sentiment_model,
    predict,
    predict_multi,
)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ── Step 2: Load/train models at startup ──────────────────────────────────────
# train_domain_sentiment_model() will:
#   1. Load verified records from data/training/train.json  (produced by Step 1)
#   2. Embed them with DistilBERT
#   3. Fit BERTopic for multi-domain classification
#   4. Return (domain_model, sentiment_model)
print("Step 2: Loading / training models from verified training data...")
domain_model, sentiment_model = train_domain_sentiment_model()
print("Models ready.\n")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(df: pd.DataFrame, industry: str = None) -> pd.DataFrame:
    """
    Enrich a DataFrame of raw news articles.

    For each article:
      - Combine title + description + content into one text field.
      - Use predict_multi() to get primary domain, secondary domains,
        domain scores, sentiment (all via DistilBERT + BERTopic).
      - Filter out articles where no domain passes the confidence threshold.
      - Score risk using the documented weighted formula.
      - Detect country / region with geo_utils.
      - Deduplicate by title.

    If `industry` is provided it OVERRIDES the primary domain label
    (the model still runs; the override only affects the 'industry' column).
    """
    if df.empty:
        return df

    # ── Build combined text (title + description + content[:300]) ─────────────
    df['text'] = (
        df['title'].fillna('') + ' ' +
        df.get('description', pd.Series(['']*len(df))).fillna('') + ' ' +
        df.get('content',     pd.Series(['']*len(df))).fillna('').str[:300]
    )

    # ── Multi-domain prediction (DistilBERT + BERTopic) ───────────────────────
    multi_results = df['text'].apply(
        lambda t: predict_multi(domain_model, sentiment_model, t)
    )

    df['primary_domain']    = multi_results.apply(lambda x: x['primary_domain'])
    df['secondary_domains'] = multi_results.apply(lambda x: x['secondary_domains'])
    df['domain_scores']     = multi_results.apply(lambda x: x['domain_scores'])
    df['domain_count']      = multi_results.apply(lambda x: x['domain_count'])
    df['sentiment']         = multi_results.apply(
        lambda x: x['sentiment'] if x['sentiment'] else 'Neutral'
    )
    df['dom_confidence']    = multi_results.apply(lambda x: x['dom_confidence'])
    df['sent_confidence']   = multi_results.apply(lambda x: x['sent_confidence'])

    # ── Filter irrelevant articles ─────────────────────────────────────────────
    df = df[df['primary_domain'] != 'Not_Relevant'].copy()
    if df.empty:
        return df

    # ── Industry label (caller override or primary domain) ────────────────────
    df['industry'] = industry if industry else df['primary_domain']

    # ── Risk scoring (Section 8.2 formula) ───────────────────────────────────
    risk = df.apply(
        lambda row: calculate_risk(row['industry'], row['sentiment'], row['dom_confidence']),
        axis=1
    )
    df['risk_score'] = risk.apply(lambda x: x[0])
    df['risk_level'] = risk.apply(lambda x: x[1])
    df['event_type'] = df['primary_domain']     # reflect actual ML domain

    # ── Geographic detection ──────────────────────────────────────────────────
    df[['country', 'region']] = df['text'].apply(
        lambda t: pd.Series(detect_country_region(t))
    )

    # ── Deduplication ─────────────────────────────────────────────────────────
    if 'title' in df.columns:
        df = df.drop_duplicates(subset='title')

    # ── Debug sample ──────────────────────────────────────────────────────────
    print(f"\n=== PIPELINE SAMPLE (industry='{industry}', rows={len(df)}) ===")
    for _, row in df.head(3).iterrows():
        print(f"  primary_domain     : '{row.get('primary_domain')}'")
        print(f"  secondary_domains  : {row.get('secondary_domains')}")
        print(f"  domain_scores      : {row.get('domain_scores')}")
        print(f"  industry (label)   : '{row.get('industry')}'")
        print(f"  risk_level         : '{row.get('risk_level')}'")
        print(f"  sentiment          : '{row.get('sentiment')}'")
        print(f"  risk_score         : {row.get('risk_score')}")
        print(f"  country            : '{row.get('country')}'")
        print("  ---")
    print("=" * 50 + "\n")

    return df


def _df_to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    keep = [
        'title', 'source', 'country', 'region',
        'industry', 'event_type',
        'primary_domain', 'secondary_domains', 'domain_scores', 'domain_count',
        'sentiment', 'risk_level', 'risk_score', 'dom_confidence',
        'published_at', 'url',
    ]
    out_cols = [c for c in keep if c in df.columns]
    records  = df[out_cols].copy()
    records['risk_score']     = records['risk_score'].round(3)
    records['dom_confidence'] = records['dom_confidence'].round(2)

    # Ensure all expected columns exist
    for col in ['source', 'country', 'region', 'published_at', 'url', 'sentiment', 'industry']:
        if col not in records.columns:
            records[col] = ''

    # Serialise list/dict columns to JSON-safe types
    for col in ['secondary_domains', 'domain_scores']:
        if col in records.columns:
            records[col] = records[col].apply(
                lambda v: v if isinstance(v, (list, dict)) else []
            )

    records = records.fillna('')
    return records.to_dict(orient='records')


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/industries', methods=['GET'])
def industries():
    """Return supported supply-chain domain labels for the frontend dropdown."""
    return jsonify({'industries': get_all_industries()})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    body      = request.get_json(silent=True) or {}
    query     = body.get('query', '').strip()
    industry  = body.get('industry', '').strip() or None
    page_size = int(body.get('page_size', 30))

    if not query and not industry:
        return jsonify({'error': 'Query or industry is required'}), 400

    # If only industry is given, use it as the broad search query
    if not query and industry:
        query = industry

    df = fetch_news(query, page_size=page_size, industry=industry)
    if df.empty:
        return jsonify({'results': [], 'total': 0})

    enriched = _run_pipeline(df, industry=industry)
    records  = _df_to_records(enriched)
    return jsonify({'results': records, 'total': len(records)})


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    try:
        df = pd.read_csv(f)
    except Exception as exc:
        return jsonify({'error': f'Could not parse CSV: {exc}'}), 400
    if 'title' not in df.columns:
        return jsonify({'error': "CSV must contain a 'title' column"}), 400
    for col in ['description', 'content']:
        if col not in df.columns:
            df[col] = ''
    industry = request.form.get('industry', '').strip() or None
    enriched = _run_pipeline(df, industry=industry)
    records  = _df_to_records(enriched)
    return jsonify({'results': records, 'total': len(records)})


@app.route('/api/export', methods=['POST'])
def export_csv():
    body    = request.get_json(silent=True) or {}
    results = body.get('results', [])
    if not results:
        return jsonify({'error': 'No data to export'}), 400
    output     = io.StringIO()
    fieldnames = [
        'title', 'source', 'country', 'region',
        'industry', 'event_type', 'primary_domain',
        'sentiment', 'risk_level', 'risk_score', 'dom_confidence',
        'published_at', 'url',
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)
    return Response(
        output.getvalue().encode('utf-8'), mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=risk_report.csv'}
    )


@app.route('/api/collect-training', methods=['POST'])
def collect_training():
    """
    STEP 1 — Trigger a training-data collection run.

    POST body (all optional):
    {
      "domains":    ["pandemic", "labour"],  // null = all 6 domains
      "page_size":  30,
      "from_date":  "2025-01-01"
    }

    Fetches news INDUSTRY-WISE using broad queries, assigns weak labels,
    saves to data/training/train.json with verified=false.
    Human reviewer must set verified=true before /api/retrain is called.

    Returns the run report as JSON.
    """
    try:
        from collect_training_data import run as collect_run
    except ImportError:
        return jsonify({'error': 'collect_training_data.py not found'}), 500

    body      = request.get_json(silent=True) or {}
    domains   = body.get('domains')   or None
    page_size = int(body.get('page_size', 30))
    from_date = body.get('from_date') or None

    try:
        report = collect_run(domains=domains, page_size=page_size, from_date=from_date)
        return jsonify(report)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """
    STEP 2 — Retrain the domain model using verified records in train.json.

    Called after a human reviewer has set verified=true on collected records.
    Reloads domain_model and sentiment_model in place — no restart needed.

    Returns: { "status": "ok", "verified_records": N }
    """
    global domain_model, sentiment_model
    try:
        print("Retraining models from verified training data...")
        domain_model, sentiment_model = train_domain_sentiment_model()
        print("Retraining complete.")

        # Report how many verified records were used
        import json
        from pathlib import Path
        train_path = Path("data/training/train.json")
        verified_count = 0
        if train_path.exists():
            records        = json.loads(train_path.read_text())
            verified_count = sum(1 for r in records if r.get("verified") is True)

        return jsonify({
            "status":           "ok",
            "verified_records": verified_count,
            "message":          "Models retrained successfully.",
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=True)