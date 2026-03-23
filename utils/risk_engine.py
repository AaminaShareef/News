"""
risk_engine.py
risk_score = domain_base x sentiment_weight x dom_confidence

Thresholds recalibrated based on observed real-world scores (0.08 - 0.28 range):
  HIGH        : >= 0.22  (top scores like Japan Hormuz 0.277)
  MEDIUM-HIGH : >= 0.17  (Iran threats 0.212, 0.208)
  MEDIUM      : >= 0.12  (oil release 0.190, missiles 0.193)
  LOW-MEDIUM  : >= 0.07  (labour LPG 0.119, weather 0.128)
  LOW         : <  0.07  (resolved/positive events)
"""

DOMAIN_BASE = {
    'Geopolitical': 0.90,
    'Economic':     0.75,
    'Logistics':    0.60,
    'Weather':      0.55,
    'Labour':       0.50,
}

SENTIMENT_WEIGHT = {
    'Negative': 1.00,
    'Neutral':  0.55,
    'Positive': 0.15,
}


def _score_to_level(score):
    if   score >= 0.22: return 'HIGH'
    elif score >= 0.17: return 'MEDIUM-HIGH'
    elif score >= 0.12: return 'MEDIUM'
    elif score >= 0.07: return 'LOW-MEDIUM'
    else:               return 'LOW'


def calculate_risk(domain, sentiment, dom_confidence):
    base       = DOMAIN_BASE.get(domain, 0.40)
    sent_w     = SENTIMENT_WEIGHT.get(sentiment or 'Neutral', 0.55)
    risk_score = round(base * sent_w * dom_confidence, 3)
    risk_level = _score_to_level(risk_score)
    return risk_score, risk_level