"""
processing/weighted_merger.py
------------------------------
Stage 4 of the ML pipeline.
Merges a group of articles about the same event into one enriched record.
Exactly as specified in doc section 5.3.

Merge weight formula (from doc):
  merge_weight = (novelty_score * 0.50)
               + (source_trust_score * 0.30)
               + (recency_score * 0.20)

novelty_score:      how many sentences contain info not in other articles
source_trust_score: Reuters=1.0, AP=0.9, BBC=0.85, Twitter=0.5, blog=0.3
recency_score:      later-published articles score higher
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _source_trust(article: dict) -> float:
    from config.settings import SOURCE_TRUST
    source = (article.get("source") or {}).get("name", "").lower()
    for key, score in SOURCE_TRUST.items():
        if key in source:
            return score
    return SOURCE_TRUST["default"]


def _recency_score(article: dict, newest_ts: float, oldest_ts: float) -> float:
    """Normalise publishedAt to 0–1 scale within the group."""
    try:
        pub = article.get("publishedAt") or ""
        ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.5
    if newest_ts == oldest_ts:
        return 1.0
    return (ts - oldest_ts) / (newest_ts - oldest_ts)


def _novelty_score(article: dict, all_sentences: set) -> float:
    """
    Fraction of sentences in this article not seen in previously processed articles.
    High novelty = brings new information to the merged record.
    """
    body = article.get("body") or article.get("clean_text") or ""
    sentences = set(s.strip() for s in body.split(".") if len(s.strip()) > 20)
    if not sentences:
        return 0.0
    new = sentences - all_sentences
    return len(new) / len(sentences)


def _parse_ts(article: dict) -> float:
    try:
        pub = article.get("publishedAt") or ""
        return datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main merger
# ---------------------------------------------------------------------------

def merge_group(group: List[dict]) -> Dict[str, Any]:
    """
    Merge a list of articles (same event cluster) into one enriched event record.

    Returns an event dict ready for BERTopic topic assignment.
    The _industry field from fetch_news.py is carried over as primary_industry_hint.
    """
    if len(group) == 1:
        return _single_to_event(group[0])

    # Compute timestamps for recency scoring
    timestamps = [_parse_ts(a) for a in group]
    newest_ts  = max(timestamps)
    oldest_ts  = min(timestamps)

    # Score each article
    all_sentences: set = set()
    scored = []
    for article in group:
        novelty  = _novelty_score(article, all_sentences)
        trust    = _source_trust(article)
        recency  = _recency_score(article, newest_ts, oldest_ts)
        weight   = (novelty * 0.50) + (trust * 0.30) + (recency * 0.20)

        scored.append((weight, article))

        # Update seen sentences so next article's novelty is relative
        body = article.get("body") or ""
        all_sentences.update(s.strip() for s in body.split(".") if len(s.strip()) > 20)

    scored.sort(key=lambda x: x[0], reverse=True)

    # Build merged body from top-weighted articles, highest novelty first
    merged_sentences: list = []
    seen: set = set()
    for _, article in scored:
        body = article.get("body") or ""
        for sent in body.split("."):
            sent = sent.strip()
            if len(sent) > 20 and sent not in seen:
                merged_sentences.append(sent)
                seen.add(sent)

    merged_body = ". ".join(merged_sentences[:20])  # cap at 20 sentences

    # Best article = highest weight → used for title, url, source
    best_weight, best = scored[0]

    # Collect all source names and URLs (doc: never discard URLs)
    sources = list({
        (article.get("source") or {}).get("name", "Unknown")
        for article in group
    })
    urls = [a.get("url") for a in group if a.get("url")]

    # Collect geo hints (basic: look for country names in title/body)
    geo = _extract_geo_hints(group)

    # industry hints from _industry tag (set by fetch_news.py)
    industry_hints = list({a.get("_industry", "") for a in group if a.get("_industry")})

    # Stable event ID from best article URL
    raw_id = best.get("url") or best.get("title") or str(newest_ts)
    event_id = hashlib.md5(raw_id.encode()).hexdigest()[:8]

    # Composite embedding = mean of all article embeddings
    import numpy as np
    embeddings = [a.get("embedding") for a in group if a.get("embedding")]
    mean_embedding = np.mean(embeddings, axis=0).tolist() if embeddings else []

    return {
        "id":                  event_id,
        "event_title":         best.get("title", ""),
        "merged_body":         merged_body,
        "embedding":           mean_embedding,
        "article_count":       len(group),
        "sources":             sources,
        "urls":                urls,
        "geo":                 geo,
        "industry_hints":      industry_hints,   # from fetch query tags
        "first_seen":          _ts_to_iso(oldest_ts),
        "last_seen":           _ts_to_iso(newest_ts),
        # These are filled by topic_modeller.py:
        "primary_industry":    None,
        "secondary_industries": [],
        "industry_scores":     {},
        "industry_count":      0,
        "disruption_types":    [],
        "risk_level":          None,
        "severity":            None,
    }


def _single_to_event(article: dict) -> Dict[str, Any]:
    """Wrap a solo article as an event record."""
    import hashlib, numpy as np
    raw_id = article.get("url") or article.get("title") or ""
    event_id = hashlib.md5(raw_id.encode()).hexdigest()[:8]
    return {
        "id":                   event_id,
        "event_title":          article.get("title", ""),
        "merged_body":          article.get("body") or article.get("clean_text") or "",
        "embedding":            article.get("embedding", []),
        "article_count":        1,
        "sources":              [(article.get("source") or {}).get("name", "Unknown")],
        "urls":                 [article.get("url")] if article.get("url") else [],
        "geo":                  _extract_geo_hints([article]),
        "industry_hints":       [article.get("_industry", "")] if article.get("_industry") else [],
        "first_seen":           article.get("publishedAt"),
        "last_seen":            article.get("publishedAt"),
        "primary_industry":     None,
        "secondary_industries": [],
        "industry_scores":      {},
        "industry_count":       0,
        "disruption_types":     [],
        "risk_level":           None,
        "severity":             None,
    }


def merge_all_groups(groups: List[List[dict]]) -> List[Dict[str, Any]]:
    """Merge all event clusters into event records."""
    events = [merge_group(g) for g in groups]
    log.info("Merger: %d clusters → %d event records", len(groups), len(events))
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_iso(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# Simple geo extraction — country/city names most common in supply chain news
_GEO_TERMS = [
    "China", "India", "USA", "United States", "Taiwan", "Japan", "South Korea",
    "Germany", "Vietnam", "Malaysia", "Indonesia", "Thailand", "Philippines",
    "Mexico", "Brazil", "Australia", "Canada", "UK", "Europe", "Middle East",
    "Ukraine", "Russia", "Chile", "Singapore", "Hong Kong", "Dubai", "Rotterdam",
    "Shanghai", "Shenzhen", "Chennai", "Mumbai", "Los Angeles", "Long Beach",
]

def _extract_geo_hints(articles: list) -> list:
    found = set()
    for a in articles:
        text = f"{a.get('title','')} {a.get('body','')}"
        for term in _GEO_TERMS:
            if term.lower() in text.lower():
                found.add(term)
    return sorted(found)