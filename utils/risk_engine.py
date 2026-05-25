"""
risk_engine.py  —  ProcureGuard
risk_score = industry_base × sentiment_weight × model_confidence

Industries are grouped into sensitivity tiers that reflect how severely
a supply-chain disruption in that sector cascades across procurement.

Thresholds (calibrated to real-world 0.08–0.30 score range):
  HIGH        : >= 0.22
  MEDIUM-HIGH : >= 0.17
  MEDIUM      : >= 0.12
  LOW-MEDIUM  : >= 0.07
  LOW         : <  0.07
"""

# ── Industry base weights ──────────────────────────────────────────────────
# Higher = more systemic impact when disrupted
INDUSTRY_BASE = {
    # Tier 1 — Critical Infrastructure / Commodities
    'Oil & Gas':            0.92,
    'Semiconductors':       0.90,
    'Pharmaceuticals':      0.88,

    # Tier 2 — Industrial & Strategic
    'Automotive':           0.80,
    'Chemicals':            0.78,
    'Metals & Mining':      0.76,
    'Aerospace & Defence':  0.75,
    'Agriculture & Food':   0.74,

    # Tier 3 — Consumer & Technology
    'Electronics':          0.68,
    'Textiles & Apparel':   0.62,
    'Retail & FMCG':        0.60,
    'Construction':         0.58,
    'Logistics & Shipping': 0.70,   # cross-cutting
    'Energy & Utilities':   0.82,

    # Fallback
    'Other':                0.45,
}

SENTIMENT_WEIGHT = {
    'Negative': 1.00,
    'Neutral':  0.55,
    'Positive': 0.15,
}


def _score_to_level(score: float) -> str:
    if   score >= 0.22: return 'HIGH'
    elif score >= 0.17: return 'MEDIUM-HIGH'
    elif score >= 0.12: return 'MEDIUM'
    elif score >= 0.07: return 'LOW-MEDIUM'
    else:               return 'LOW'


def calculate_risk(industry: str, sentiment: str, model_confidence: float):
    """
    Returns (risk_score: float, risk_level: str)
    """
    base       = INDUSTRY_BASE.get(industry, INDUSTRY_BASE['Other'])
    sent_w     = SENTIMENT_WEIGHT.get(sentiment or 'Neutral', 0.55)
    risk_score = round(base * sent_w * model_confidence, 3)
    risk_level = _score_to_level(risk_score)
    return risk_score, risk_level


def get_all_industries() -> list:
    """Return sorted list of all supported industry names."""
    return sorted(INDUSTRY_BASE.keys())