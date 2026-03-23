"""
app.py - ProcureGuard Flask application
"""
import io, csv
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from utils.news_fetcher import fetch_news
from utils.geo_utils import detect_country_region
from models.domain_sentiment_model import train_domain_sentiment_model, clean_text, predict
from utils.risk_engine import calculate_risk

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print("Loading models...")
domain_model, sentiment_model = train_domain_sentiment_model()
print("Models ready.\n")


def _run_pipeline(df):
    if df.empty:
        return df

    df['text'] = (
        df['title'].fillna('') + ' ' +
        df.get('description', pd.Series(['']*len(df))).fillna('') + ' ' +
        df.get('content', pd.Series(['']*len(df))).fillna('')
    )

    results = df['text'].apply(lambda t: predict(domain_model, sentiment_model, t))
    df['domain']          = results.apply(lambda x: x[0])
    df['sentiment']       = results.apply(lambda x: x[1] if x[1] else 'Neutral')
    df['dom_confidence']  = results.apply(lambda x: x[2])
    df['sent_confidence'] = results.apply(lambda x: x[3])

    df = df[df['domain'] != 'Not_Relevant'].copy()
    if df.empty:
        return df

    risk = df.apply(lambda row: calculate_risk(row['domain'], row['sentiment'], row['dom_confidence']), axis=1)
    df['risk_score'] = risk.apply(lambda x: x[0])
    df['risk_level'] = risk.apply(lambda x: x[1])  # clean string: 'HIGH', 'MEDIUM-HIGH', etc

    df['event_type'] = df['domain']

    df[['country', 'region']] = df['text'].apply(lambda t: pd.Series(detect_country_region(t)))

    if 'title' in df.columns:
        df = df.drop_duplicates(subset='title')

    # Debug print — shows exact values going to frontend
    print("\n=== DATA SAMPLE (first 3 rows) ===")
    for _, row in df.head(3).iterrows():
        print(f"  event_type : '{row.get('event_type')}'")
        print(f"  risk_level : '{row.get('risk_level')}'")
        print(f"  sentiment  : '{row.get('sentiment')}'")
        print(f"  risk_score : {row.get('risk_score')}")
        print(f"  country    : '{row.get('country')}'")
        print("  ---")
    print(f"  Total: {len(df)}")
    print("==================================\n")

    return df


def _df_to_records(df):
    if df.empty:
        return []
    keep = ['title', 'source', 'country', 'region', 'event_type',
            'sentiment', 'risk_level', 'risk_score', 'dom_confidence', 'published_at', 'url']
    out_cols = [c for c in keep if c in df.columns]
    records = df[out_cols].copy()
    records['risk_score']     = records['risk_score'].round(3)
    records['dom_confidence'] = records['dom_confidence'].round(2)
    for col in ['source', 'country', 'region', 'published_at', 'url', 'sentiment']:
        if col not in records.columns:
            records[col] = ''
    records = records.fillna('')
    return records.to_dict(orient='records')


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    body      = request.get_json(silent=True) or {}
    query     = body.get('query', '').strip()
    page_size = int(body.get('page_size', 20))
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    df = fetch_news(query, page_size=page_size)
    if df.empty:
        return jsonify({'results': [], 'total': 0})
    enriched = _run_pipeline(df)
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
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {e}'}), 400
    if 'title' not in df.columns:
        return jsonify({'error': "CSV must contain a 'title' column"}), 400
    for col in ['description', 'content']:
        if col not in df.columns:
            df[col] = ''
    enriched = _run_pipeline(df)
    records  = _df_to_records(enriched)
    return jsonify({'results': records, 'total': len(records)})


@app.route('/api/export', methods=['POST'])
def export_csv():
    body    = request.get_json(silent=True) or {}
    results = body.get('results', [])
    if not results:
        return jsonify({'error': 'No data to export'}), 400
    output     = io.StringIO()
    fieldnames = ['title', 'source', 'country', 'region', 'event_type',
                  'sentiment', 'risk_level', 'risk_score', 'dom_confidence', 'published_at', 'url']
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)
    return Response(output.getvalue().encode('utf-8'), mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=risk_report.csv'})


if __name__ == '__main__':
    app.run(debug=True)