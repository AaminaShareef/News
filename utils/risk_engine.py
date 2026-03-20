def calculate_risk(event_type):
    """
    Assign risk score and level based on event type.
    Returns clean labels with no emojis.
    """
    risk_mapping = {
        "Geopolitical": (0.9,  "HIGH"),
        "Economic":     (0.75, "MEDIUM-HIGH"),
        "Logistics":    (0.6,  "MEDIUM"),
        "Weather":      (0.55, "MEDIUM"),
        "Labour":       (0.5,  "LOW-MEDIUM"),
    }
    return risk_mapping.get(event_type, (0.4, "LOW"))