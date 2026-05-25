"""
processing/topic_modeller.py
-----------------------------
Stage 5 of the ML pipeline.
Assigns up to 3 industries per event record using:
  1. Primary label  — from _industry tag set by fetch_news.py (NewsAPI query axis)
  2. Secondary labels — cosine similarity of embedding against industry keyword centroids
  3. Disruption types — keyword scoring against disruption signal vocabulary

No external heavy models (no bart-large-mnli). All inference uses the
same DistilBERT embeddings already computed by embedder.py.

Doc sections 5.4 and 5.5 compliance:
  - Probability score per industry per event
  - Threshold: 0.40 (DOMAIN_CONFIDENCE_THRESHOLD)
  - Hard cap: 3 industries per event (MAX_INDUSTRIES_PER_ARTICLE)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Industry keyword vocabulary
# Used to build centroid embeddings for each industry.
# DistilBERT embeds these phrases → centroid = mean embedding.
# New article embedding compared via cosine sim → confidence score.
# ---------------------------------------------------------------------------

INDUSTRY_VOCAB: Dict[str, List[str]] = {
    "semiconductor_electronics": [
        "chip shortage semiconductor wafer fab",
        "electronics component PCB supply disruption",
        "TSMC Intel AMD chip production halt",
        "microchip manufacturing delay foundry",
        "semiconductor supply chain bottleneck",
    ],
    "automotive": [
        "car production halt assembly plant shutdown",
        "automotive parts shortage vehicle delay",
        "EV battery supply disruption electric vehicle",
        "automobile manufacturer supply chain",
        "truck production stop factory closure automotive",
    ],
    "pharmaceuticals_healthcare": [
        "drug shortage pharmaceutical supply disruption",
        "medicine API active pharmaceutical ingredient delay",
        "hospital medical supply chain shortage",
        "vaccine raw material shortage pharma",
        "medical device supply disruption healthcare",
    ],
    "food_agriculture": [
        "food supply disruption crop shortage harvest failure",
        "grain wheat rice supply chain logistics",
        "agriculture fertilizer shortage farming disruption",
        "food import export ban commodity shortage",
        "livestock feed supply chain disruption",
    ],
    "energy_oil_gas": [
        "oil supply disruption crude petroleum shortage",
        "natural gas pipeline disruption energy crisis",
        "LNG LPG fuel supply chain delay",
        "refinery shutdown oil production halt",
        "energy supply chain OPEC price surge",
    ],
    "retail_consumer_goods": [
        "retail supply chain disruption consumer goods shortage",
        "inventory shortage warehouse logistics delay",
        "ecommerce supply chain disruption FMCG",
        "supermarket product shortage retail distribution",
        "consumer product supply delay import",
    ],
    "shipping_logistics": [
        "port congestion shipping delay freight disruption",
        "container shortage vessel delay cargo halt",
        "logistics supply chain disruption shipping route",
        "port strike dock workers freight delay",
        "shipping lane blockade suez canal disruption",
    ],
    "aerospace_defense": [
        "aerospace supply chain disruption aircraft parts",
        "aviation parts shortage airline manufacturing delay",
        "defense contractor supply chain disruption",
        "satellite component shortage aerospace manufacturing",
        "aircraft production halt aerospace supplier",
    ],
    "construction_raw_materials": [
        "steel shortage construction supply chain disruption",
        "lumber cement raw material supply delay",
        "copper mining supply disruption construction",
        "building material shortage infrastructure project delay",
        "construction supply chain bottleneck raw material",
    ],
    "technology_it_hardware": [
        "server shortage data center supply disruption",
        "IT hardware supply chain delay networking equipment",
        "cloud infrastructure hardware shortage technology",
        "storage hardware supply disruption data center",
        "enterprise hardware delay supply chain technology",
    ],
}

# ---------------------------------------------------------------------------
# Disruption type keyword vocabulary
# Scores how much each disruption TYPE is present in the article.
# These are the CAUSES of disruption (orthogonal to industry).
# ---------------------------------------------------------------------------

DISRUPTION_VOCAB: Dict[str, List[str]] = {
    "pandemic_health": [
        "outbreak epidemic pandemic virus disease infection quarantine",
        "lockdown health crisis factory closure workers sick",
        "covid flu virus health emergency plant shutdown",
    ],
    "conflict_geopolitical": [
        "war conflict sanctions geopolitical tension military",
        "trade war tariff embargo blockade dispute",
        "political crisis coup unrest instability sanctions",
    ],
    "weather_climate": [
        "typhoon hurricane flood earthquake drought wildfire",
        "extreme weather climate disaster storm disruption",
        "natural disaster rainfall port closure weather",
    ],
    "labour_strike": [
        "workers strike protest industrial action walkout",
        "dock workers labour dispute union strike shutdown",
        "factory workers protest wage dispute strike",
    ],
    "economic_commodity": [
        "commodity price surge inflation currency crash",
        "raw material price spike economic downturn recession",
        "gold oil price hike supply demand imbalance",
    ],
    "political_policy": [
        "trade policy regulation export ban import restriction",
        "government policy change tariff new law regulation",
        "trade agreement breakdown policy shift restriction",
    ],
    "cyber_attack": [
        "cyber attack ransomware hack breach IT outage",
        "cybersecurity incident system outage digital attack",
        "malware infrastructure attack production system hack",
    ],
    "regulatory_compliance": [
        "regulatory compliance environmental regulation ban",
        "safety recall product ban regulatory shutdown",
        "compliance violation regulatory fine production halt",
    ],
}

# Cache for centroid embeddings — built once per session
_industry_centroids: Optional[Dict[str, np.ndarray]] = None
_disruption_centroids: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Build centroids
# ---------------------------------------------------------------------------

def _build_centroids(vocab: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Embed each phrase in the vocab and take the mean as the centroid.
    Centroid = representative embedding for that industry/disruption type.
    """
    from processing.embedder import embed_text

    centroids = {}
    for label, phrases in vocab.items():
        phrase_embeddings = np.array([embed_text(p) for p in phrases])
        centroid = phrase_embeddings.mean(axis=0)
        # Re-normalise centroid (mean of unit vectors is not unit length)
        norm = np.linalg.norm(centroid)
        centroids[label] = centroid / norm if norm > 0 else centroid
        log.debug("Centroid built for: %s", label)

    return centroids


def _get_industry_centroids() -> Dict[str, np.ndarray]:
    global _industry_centroids
    if _industry_centroids is None:
        log.info("Building industry centroid embeddings ...")
        _industry_centroids = _build_centroids(INDUSTRY_VOCAB)
        log.info("Industry centroids ready for %d industries.", len(_industry_centroids))
    return _industry_centroids


def _get_disruption_centroids() -> Dict[str, np.ndarray]:
    global _disruption_centroids
    if _disruption_centroids is None:
        log.info("Building disruption type centroid embeddings ...")
        _disruption_centroids = _build_centroids(DISRUPTION_VOCAB)
        log.info("Disruption centroids ready for %d types.", len(_disruption_centroids))
    return _disruption_centroids


# ---------------------------------------------------------------------------
# Industry assignment
# ---------------------------------------------------------------------------

def assign_industries_zero_shot(
    events: List[Dict],
    confidence_threshold: float = None,
    max_industries: int = None,
) -> List[Dict]:
    """
    Assign industries to event records using:
      - Primary:   _industry tag from NewsAPI fetch query (high confidence)
      - Secondary: cosine similarity vs industry centroid embeddings

    No external model needed — reuses DistilBERT embeddings from embedder.py.
    Doc spec: probability score per industry, threshold 0.40, max 3 per event.
    """
    from config.settings import (
        DOMAIN_CONFIDENCE_THRESHOLD,
        MAX_INDUSTRIES_PER_ARTICLE,
        INDUSTRY_LABELS,
        INDUSTRY_IDS,
    )

    threshold   = confidence_threshold or DOMAIN_CONFIDENCE_THRESHOLD
    max_ind     = max_industries or MAX_INDUSTRIES_PER_ARTICLE
    centroids   = _get_industry_centroids()

    log.info("Assigning industries to %d events (embedding similarity) ...", len(events))

    for event in events:
        embedding = event.get("embedding")
        if not embedding:
            _apply_hint_fallback(event, INDUSTRY_LABELS)
            continue

        emb = np.array(embedding, dtype=np.float32)

        # Score all industries via cosine similarity (dot product, both normalised)
        scores: Dict[str, float] = {}
        for ind_id, centroid in centroids.items():
            sim = float(np.dot(emb, centroid))
            scores[ind_id] = round(max(0.0, sim), 4)   # cosine can be negative

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # PRIMARY: boost the industry from the fetch query tag (high-confidence signal)
        hints = event.get("industry_hints", [])
        hint_ids = _resolve_hint_ids(hints, INDUSTRY_LABELS)

        # Apply boost: +0.20 to hint industries (they came from targeted queries)
        boosted: Dict[str, float] = {}
        for ind_id, score in scores.items():
            boost = 0.20 if ind_id in hint_ids else 0.0
            boosted[ind_id] = round(min(1.0, score + boost), 4)

        # Re-rank after boost
        ranked_boosted = sorted(boosted.items(), key=lambda x: x[1], reverse=True)

        # Filter by threshold, cap at max_industries
        selected = [
            (ind_id, score)
            for ind_id, score in ranked_boosted
            if score >= threshold
        ][:max_ind]

        if not selected:
            # Fallback: use hint tag with placeholder confidence
            _apply_hint_fallback(event, INDUSTRY_LABELS)
            continue

        industry_scores = {ind_id: score for ind_id, score in selected}

        event["primary_industry"]      = selected[0][0]
        event["secondary_industries"]  = [s[0] for s in selected[1:]]
        event["industry_scores"]       = industry_scores
        event["industry_count"]        = len(selected)

    log.info("Industry assignment complete.")
    return events


# ---------------------------------------------------------------------------
# Disruption type assignment
# ---------------------------------------------------------------------------

def assign_disruption_types(events: List[Dict]) -> List[Dict]:
    """
    Assign disruption types (CAUSE of disruption) via centroid similarity.
    Orthogonal to industry — one event can be:
      industry=Automotive AND disruption_type=labour_strike + weather_climate
    """
    from config.settings import DOMAIN_CONFIDENCE_THRESHOLD

    centroids = _get_disruption_centroids()
    threshold = DOMAIN_CONFIDENCE_THRESHOLD

    log.info("Assigning disruption types to %d events ...", len(events))

    for event in events:
        embedding = event.get("embedding")
        if not embedding:
            event["disruption_types"] = []
            continue

        emb = np.array(embedding, dtype=np.float32)

        scores = {
            dt_id: round(float(max(0.0, np.dot(emb, centroid))), 4)
            for dt_id, centroid in centroids.items()
        }

        event["disruption_types"] = [
            dt_id for dt_id, score in
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score >= threshold
        ]

    log.info("Disruption type assignment complete.")
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_hint_ids(hints: List[str], industry_labels: Dict[str, str]) -> List[str]:
    """Map label strings (e.g. 'Automotive') back to IDs (e.g. 'automotive')."""
    label_to_id = {v: k for k, v in industry_labels.items()}
    resolved = []
    for hint in hints:
        if hint in label_to_id:
            resolved.append(label_to_id[hint])
        elif hint in industry_labels:
            resolved.append(hint)  # already an ID
    return resolved


def _apply_hint_fallback(event: Dict, industry_labels: Dict) -> None:
    """Fallback: use _industry hint tag from fetch query."""
    hints = event.get("industry_hints", [])
    resolved = _resolve_hint_ids(hints, industry_labels)
    if resolved:
        event["primary_industry"]      = resolved[0]
        event["secondary_industries"]  = resolved[1:]
        event["industry_scores"]       = {r: 0.50 for r in resolved}
        event["industry_count"]        = len(resolved)
    else:
        event["primary_industry"]      = "unclassified"
        event["secondary_industries"]  = []
        event["industry_scores"]       = {}
        event["industry_count"]        = 0