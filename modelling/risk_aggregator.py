"""
modelling/risk_aggregator.py
-----------------------------
Computes risk scores per industry from event records.
Exactly as specified in doc section 8.2.

Risk formula (from doc):
  risk_score = (severity_score        * 0.40)
             + (article_volume_norm   * 0.25)
             + (source_diversity_norm * 0.20)
             + (recency_boost         * 0.15)

Risk levels:
  LOW      < 2.0
  MEDIUM   2.0 – 3.4
  HIGH     3.5 – 4.4
  CRITICAL 4.5+

Recency rule (from doc):
  Events from last 24 hrs count 2x vs last week.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-event severity estimation
# ---------------------------------------------------------------------------

# Keywords that push severity up — found in merged_body or title
_HIGH_SEVERITY_SIGNALS = [
    "halt", "shutdown", "closure", "bankrupt", "collapse", "crisis",
    "critical", "severe", "major", "emergency", "catastrophic", "devastat",
    "blockade", "sanction", "war", "conflict", "outbreak", "explosion",
]
_MEDIUM_SEVERITY_SIGNALS = [
    "delay", "shortage", "disruption", "strike", "protest", "shortage",
    "surge", "rise", "increase", "shortage", "congestion", "restrict",
]


def estimate_severity(event: Dict) -> float:
    """
    Estimate event severity on a 1–5 scale based on:
    - article_count (more coverage = more serious)
    - source diversity (Reuters > blog)
    - language signals in title/body
    """
    from config.settings import SOURCE_TRUST

    text = f"{event.get('event_title','')} {event.get('merged_body','')}".lower()
    article_count = event.get("article_count", 1)
    sources = event.get("sources", [])

    # Base: article volume (1–2 points)
    volume_score = min(2.0, article_count / 5.0)

    # Language signals (0–2 points)
    lang_score = 0.0
    for kw in _HIGH_SEVERITY_SIGNALS:
        if kw in text:
            lang_score += 0.3
    for kw in _MEDIUM_SEVERITY_SIGNALS:
        if kw in text:
            lang_score += 0.1
    lang_score = min(2.0, lang_score)

    # Source quality (0–1 point)
    trust_scores = []
    for src in sources:
        src_lower = src.lower()
        for key, score in SOURCE_TRUST.items():
            if key in src_lower:
                trust_scores.append(score)
                break
        else:
            trust_scores.append(SOURCE_TRUST["default"])
    source_score = max(trust_scores) if trust_scores else 0.5

    severity = volume_score + lang_score + source_score
    return round(min(5.0, max(1.0, severity)), 2)


# ---------------------------------------------------------------------------
# Risk score per event
# ---------------------------------------------------------------------------

def _recency_multiplier(event: Dict) -> float:
    """
    Events from last 24 hrs → multiplier 2.0
    Last week → multiplier 1.0
    Older → multiplier 0.5
    (doc: events from last 24 hrs count 2x vs last week)
    """
    try:
        last_seen = event.get("last_seen") or event.get("first_seen") or ""
        ts = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - ts
        if age < timedelta(hours=24):
            return 2.0
        elif age < timedelta(days=7):
            return 1.0
        else:
            return 0.5
    except Exception:
        return 1.0


def compute_event_risk(event: Dict, max_article_count: int = 50) -> Dict:
    """
    Compute risk_score and risk_level for a single event.
    Adds 'severity', 'risk_score', 'risk_level' to the event dict.
    """
    from config.settings import RISK_WEIGHTS, RISK_THRESHOLDS, SOURCE_TRUST

    severity = estimate_severity(event)
    event["severity"] = severity

    # severity_score: normalise 1–5 to 0–1, then scale to 0–5
    severity_norm = (severity - 1.0) / 4.0

    # article_volume_norm: normalise by expected max
    vol_norm = min(1.0, event.get("article_count", 1) / max_article_count)

    # source_diversity_norm: how many distinct trusted sources
    sources = event.get("sources", [])
    trusted = sum(
        1 for s in sources
        if any(k in s.lower() for k in ["reuters", "bbc", "bloomberg", "ap", "ft", "wsj"])
    )
    diversity_norm = min(1.0, trusted / 4.0)

    # recency_boost
    recency = _recency_multiplier(event)
    recency_norm = recency / 2.0  # normalise 0.5–2.0 → 0.25–1.0

    raw_score = (
        severity_norm  * RISK_WEIGHTS["severity"]         * 5 +
        vol_norm       * RISK_WEIGHTS["article_volume"]   * 5 +
        diversity_norm * RISK_WEIGHTS["source_diversity"] * 5 +
        recency_norm   * RISK_WEIGHTS["recency"]          * 5
    )

    risk_score = round(min(5.0, raw_score), 3)

    # Assign level
    if risk_score < RISK_THRESHOLDS["LOW"]:
        risk_level = "LOW"
    elif risk_score < RISK_THRESHOLDS["MEDIUM"]:
        risk_level = "MEDIUM"
    elif risk_score < RISK_THRESHOLDS["HIGH"]:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    event["risk_score"] = risk_score
    event["risk_level"] = risk_level
    return event


# ---------------------------------------------------------------------------
# Aggregate risk per industry
# ---------------------------------------------------------------------------

def aggregate_industry_risk(events: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate risk scores across all events grouped by industry.
    Returns a dict keyed by industry_id with summary stats.
    Used by the dashboard to show industry-level risk overview.
    """
    from config.settings import INDUSTRY_LABELS, RISK_THRESHOLDS

    industry_events: Dict[str, List[Dict]] = {}

    for event in events:
        primary = event.get("primary_industry")
        secondaries = event.get("secondary_industries", [])
        all_industries = ([primary] if primary else []) + secondaries

        for ind_id in all_industries:
            if ind_id and ind_id != "unclassified":
                industry_events.setdefault(ind_id, []).append(event)

    summary = {}
    for ind_id, ind_events in industry_events.items():
        risk_scores = [e.get("risk_score", 0) for e in ind_events]
        severities  = [e.get("severity", 1) for e in ind_events]
        article_counts = [e.get("article_count", 1) for e in ind_events]

        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)

        if avg_risk < RISK_THRESHOLDS["LOW"]:
            level = "LOW"
        elif avg_risk < RISK_THRESHOLDS["MEDIUM"]:
            level = "MEDIUM"
        elif avg_risk < RISK_THRESHOLDS["HIGH"]:
            level = "HIGH"
        else:
            level = "CRITICAL"

        summary[ind_id] = {
            "industry_id":     ind_id,
            "industry_label":  INDUSTRY_LABELS.get(ind_id, ind_id),
            "event_count":     len(ind_events),
            "total_articles":  sum(article_counts),
            "avg_risk_score":  round(avg_risk, 3),
            "max_risk_score":  round(max_risk, 3),
            "avg_severity":    round(sum(severities) / len(severities), 2),
            "risk_level":      level,
            "top_events":      sorted(
                ind_events, key=lambda e: e.get("risk_score", 0), reverse=True
            )[:5],
        }

    log.info("Risk aggregated across %d industries.", len(summary))
    return summary


def score_all_events(events: List[Dict]) -> List[Dict]:
    """Apply risk scoring to every event in the list."""
    max_count = max((e.get("article_count", 1) for e in events), default=50)
    for event in events:
        compute_event_risk(event, max_article_count=max_count)
    log.info("Risk scores computed for %d events.", len(events))
    return events