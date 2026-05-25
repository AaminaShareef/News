"""
config/settings.py
------------------
All configuration loaded from environment / config.py (project root).
Mirrors the .env.example from the doc exactly, with industries replacing domains.
"""

import os
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Import API keys from project-root config.py
# ---------------------------------------------------------------------------
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
        NEWS_API_KEY,
        # Add these to your config.py if using additional sources:
        # TWITTER_BEARER_TOKEN,
        # SERPAPI_KEY,
        # MEDIASTACK_KEY,
    )
except ImportError as e:
    raise ImportError(f"config.py missing or incomplete: {e}")

# Optional keys — default to None if not set
TWITTER_BEARER_TOKEN = getattr(__import__('config'), 'TWITTER_BEARER_TOKEN', None)
SERPAPI_KEY          = getattr(__import__('config'), 'SERPAPI_KEY', None)
MEDIASTACK_KEY       = getattr(__import__('config'), 'MEDIASTACK_KEY', None)

# ---------------------------------------------------------------------------
# ML settings (from doc section 10 env vars)
# ---------------------------------------------------------------------------
DISTILBERT_MODEL            = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD        = 0.82    # grouping cutoff
DEDUP_THRESHOLD             = 0.95    # near-duplicate removal
DOMAIN_CONFIDENCE_THRESHOLD = 0.40    # BERTopic min score to assign industry
MAX_INDUSTRIES_PER_ARTICLE  = 3       # hard cap (doc calls this MAX_DOMAINS)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parents[1]
DATA_DIR        = BASE_DIR / "data"
TRAINING_DIR    = DATA_DIR / "training"
EVENTS_DIR      = DATA_DIR / "events"
MODELS_DIR      = DATA_DIR / "models"
NEWS_JSON       = DATA_DIR / "news.json"
TRAIN_JSON      = TRAINING_DIR / "train.json"
EVENTS_MASTER   = EVENTS_DIR / "events_master.json"

# Ensure directories exist
for d in [TRAINING_DIR, EVENTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scheduler (from doc section 4.2)
# ---------------------------------------------------------------------------
DEFAULT_POLL_INTERVAL_SECONDS = 1800   # 30 min
ALERT_POLL_INTERVAL_SECONDS   = 300    # 5 min during alert
ALERT_DURATION_HOURS          = 6

# ---------------------------------------------------------------------------
# Risk score weights (from doc section 8.2)
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "severity":          0.40,
    "article_volume":    0.25,
    "source_diversity":  0.20,
    "recency":           0.15,
}

RISK_THRESHOLDS = {
    "LOW":      2.0,
    "MEDIUM":   3.4,
    "HIGH":     4.4,
    "CRITICAL": float("inf"),
}

# Source trust scores (from doc section 5.3)
SOURCE_TRUST = {
    "reuters":   1.0,
    "ap":        0.9,
    "bbc":       0.85,
    "bloomberg": 0.85,
    "ft":        0.80,
    "wsj":       0.80,
    "default":   0.5,
    "blog":      0.3,
}

# ---------------------------------------------------------------------------
# Load industry map YAML
# ---------------------------------------------------------------------------
_yaml_path = Path(__file__).parent / "industry_map.yaml"
with open(_yaml_path) as f:
    _cfg = yaml.safe_load(f)

INDUSTRIES        = _cfg["industries"]
DISRUPTION_TYPES  = _cfg["disruption_types"]
INDUSTRY_IDS      = [i["id"] for i in INDUSTRIES]
INDUSTRY_LABELS   = {i["id"]: i["label"] for i in INDUSTRIES}