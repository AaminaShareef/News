"""
utils/collect_trainingdata.py
------------------------------
TRACK A — Training Data Generation Pipeline.
Exactly as specified in doc section 2 (Track A) and section 6.

Pipeline:
  news.json (raw)
    → clean          (cleaner.py)
    → embed          (embedder.py)
    → semantic group (semantic_grouper.py)
    → weighted merge (weighted_merger.py)
    → weak labels    (topic_modeller.py zero-shot)
    → risk score     (risk_aggregator.py)
    → train.json     (ready for human review → fine-tuning)

Doc section 6.1: Weak auto-labels, ~70-75% accuracy.
Doc section 6.2: Human review flags written to train.json (verified=false initially).
Doc section 6.3: train.json ≠ events_master.json — never mix.

Output: data/training/train.json
"""

import json
import logging
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load raw news.json
# ---------------------------------------------------------------------------

def load_news_json(path: Path) -> List[Dict]:
    log.info("Loading raw articles from %s ...", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    log.info("Loaded %d raw articles.", len(articles))
    return articles


# ---------------------------------------------------------------------------
# Weak labeling (doc section 6.1)
# Zero-shot classification = ~70-75% accuracy, good enough to start
# ---------------------------------------------------------------------------

def generate_weak_labels(events: List[Dict]) -> List[Dict]:
    """
    Step 1 of weak labeling: zero-shot industry assignment (topic_modeller.py).
    Step 2: disruption type assignment (what caused it).
    Both done in topic_modeller — called from pipeline below.
    Already applied in run_pipeline(). This function adds the
    'weak_label_method' audit field required by train.json schema.
    """
    for event in events:
        event["weak_label_method"] = "zero_shot_bart_mnli"
        event["weak_label_confidence"] = event.get("industry_scores", {})
    return events


# ---------------------------------------------------------------------------
# Build train.json records (doc section 7.2)
# ---------------------------------------------------------------------------

def _assign_split(idx: int, total: int) -> str:
    """
    Assign train/val/test split automatically.
    Doc spec: 80% train, 10% val, 10% test.
    """
    r = (idx * 2654435761) % total  # deterministic shuffle by index
    pct = r / total
    if pct < 0.80:
        return "train"
    elif pct < 0.90:
        return "val"
    else:
        return "test"


def events_to_training_records(events: List[Dict]) -> List[Dict]:
    """
    Convert merged event records into training records.
    Schema exactly as doc section 7.2:

    {
        "id":          "tr_00142",
        "text":        "...",
        "labels":      ["semiconductor_electronics", "automotive"],
        "severity":    3.8,
        "source":      "Reuters",
        "date":        "2024-03-12",
        "split":       "train",
        "verified":    false,          ← human review sets this to true
        "verified_by": null,
        "risk_level":  "HIGH",
        "disruption_types": ["labour_strike"],
        "industry_scores":  {"semiconductor_electronics": 0.81},
        "weak_label_method": "zero_shot_bart_mnli",
        "event_id":    "a3f9c2d1",
        "geo":         ["Taiwan"],
        "article_count": 7,
    }
    """
    records = []
    total = len(events)

    for idx, event in enumerate(events):
        primary = event.get("primary_industry")
        secondary = event.get("secondary_industries", [])

        labels = []
        if primary and primary != "unclassified":
            labels.append(primary)
        labels.extend(secondary)

        if not labels:
            continue  # skip unclassified events from training data

        # Build text: title + merged body (truncated)
        title = event.get("event_title", "")
        body  = event.get("merged_body", "")[:500]
        text  = f"{title}. {body}".strip()

        if len(text) < 30:
            continue

        # Best source name
        sources = event.get("sources", [])
        source  = sources[0] if sources else "Unknown"

        # Date from first_seen
        date_str = (event.get("first_seen") or "")[:10] or \
                   datetime.now(timezone.utc).strftime("%Y-%m-%d")

        record_id = f"tr_{idx:05d}"

        record = {
            "id":                 record_id,
            "text":               text,
            "labels":             labels,
            "severity":           event.get("severity"),
            "source":             source,
            "date":               date_str,
            "split":              _assign_split(idx, total),
            "verified":           False,        # human review sets True
            "verified_by":        None,
            "risk_level":         event.get("risk_level"),
            "risk_score":         event.get("risk_score"),
            "disruption_types":   event.get("disruption_types", []),
            "industry_scores":    event.get("industry_scores", {}),
            "weak_label_method":  event.get("weak_label_method", "zero_shot_bart_mnli"),
            "event_id":           event.get("id"),
            "geo":                event.get("geo", []),
            "article_count":      event.get("article_count", 1),
            "urls":               event.get("urls", []),
        }
        records.append(record)

    log.info(
        "Training records: %d total | train=%d val=%d test=%d",
        len(records),
        sum(1 for r in records if r["split"] == "train"),
        sum(1 for r in records if r["split"] == "val"),
        sum(1 for r in records if r["split"] == "test"),
    )
    return records


# ---------------------------------------------------------------------------
# Save train.json
# ---------------------------------------------------------------------------

def save_train_json(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated_at":    datetime.now(timezone.utc).isoformat() + "Z",
            "total_records":   len(records),
            "verified_count":  sum(1 for r in records if r["verified"]),
            "split_counts": {
                "train": sum(1 for r in records if r["split"] == "train"),
                "val":   sum(1 for r in records if r["split"] == "val"),
                "test":  sum(1 for r in records if r["split"] == "test"),
            },
            "label_counts":    _count_labels(records),
            "note": (
                "Weak auto-labels. verified=false records need human review "
                "before fine-tuning. See doc section 6.2."
            ),
        },
        "records": records,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("Saved %d training records → %s", len(records), path)


def _count_labels(records: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        for label in r.get("labels", []):
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Full Track A pipeline
# ---------------------------------------------------------------------------

def run_pipeline(news_json_path: Path, train_json_path: Path) -> None:
    """
    Full Track A pipeline as per doc section 2.
    """
    from processing.cleaner          import clean_articles
    from processing.embedder         import embed_articles
    from processing.semantic_grouper import group_articles
    from processing.weighted_merger  import merge_all_groups
    from processing.topic_modeller   import assign_industries_zero_shot, assign_disruption_types
    from modelling.risk_aggregator   import score_all_events
    from config.settings             import SIMILARITY_THRESHOLD, DEDUP_THRESHOLD

    log.info("=" * 60)
    log.info("TRACK A — Training Data Generation Pipeline")
    log.info("=" * 60)

    # Step 1: Load
    raw_articles = load_news_json(news_json_path)

    # Step 2: Clean (cleaner.py)
    log.info("[1/7] Cleaning articles ...")
    clean = clean_articles(raw_articles)

    # Step 3: Embed (embedder.py)
    log.info("[2/7] Embedding with DistilBERT ...")
    embedded = embed_articles(clean)

    # Step 4: Semantic grouping (semantic_grouper.py)
    log.info("[3/7] Grouping by semantic similarity ...")
    groups = group_articles(embedded, SIMILARITY_THRESHOLD, DEDUP_THRESHOLD)

    # Step 5: Weighted merge (weighted_merger.py)
    log.info("[4/7] Merging groups into event records ...")
    events = merge_all_groups(groups)

    # Step 6: Weak labels — industry assignment (topic_modeller.py)
    log.info("[5/7] Assigning industry labels (zero-shot) ...")
    events = assign_industries_zero_shot(events)

    # Step 6b: Disruption type assignment
    log.info("[6/7] Assigning disruption types ...")
    events = assign_disruption_types(events)
    events = generate_weak_labels(events)

    # Step 7: Risk scoring (risk_aggregator.py)
    log.info("[7/7] Computing risk scores ...")
    events = score_all_events(events)

    # Build training records and save
    log.info("Building training records ...")
    records = events_to_training_records(events)
    save_train_json(records, train_json_path)

    log.info("=" * 60)
    log.info("Track A complete.")
    log.info("  Events processed : %d", len(events))
    log.info("  Training records : %d", len(records))
    log.info("  Output           : %s", train_json_path)
    log.info("  Next step        : Human review (set verified=true)")
    log.info("  Then             : python train.py")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Programmatic wrapper used by the Flask app
def run(domains: List[str] = None, page_size: int = 30, from_date: str = None) -> Dict[str, Any]:
    """
    Run the Track A pipeline using configured NEWS_JSON / TRAIN_JSON paths.

    Parameters `domains`, `page_size`, `from_date` are accepted for API
    compatibility but currently not used by the offline pipeline.

    Returns a JSON-serialisable report dict.
    """
    from config.settings import NEWS_JSON, TRAIN_JSON

    try:
        run_pipeline(NEWS_JSON, TRAIN_JSON)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    try:
        with open(TRAIN_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "ok", "report": data.get("metadata", {})}
    except Exception as exc:
        return {"status": "ok", "report": {}, "warning": f"could not read {TRAIN_JSON}: {exc}"}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from config.settings import NEWS_JSON, TRAIN_JSON

    if not NEWS_JSON.exists():
        log.error("news.json not found at %s — run fetch_news.py first.", NEWS_JSON)
        sys.exit(1)

    run_pipeline(NEWS_JSON, TRAIN_JSON)