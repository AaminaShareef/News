def rule_based_filter(text):
    """
    Basic keyword filter (fast filtering)
    """
    if not text:
        return False

    text = text.lower()

    keywords = [
        "war", "strike", "protest", "shutdown",
        "delay", "disruption", "port", "shipping",
        "logistics", "supply", "oil", "gas",
        "trade", "sanction", "conflict"
    ]

    return any(word in text for word in keywords)


def strong_keyword_check(text):
    if not text:
        return False

    text = text.lower()

    strong_keywords = [
        "war", "conflict", "attack", "strike", "shutdown",
        "disruption", "delay", "crisis",
        "port", "shipping", "cargo", "freight",
        "logistics", "transport", "shipment",
        "collapse", "explosion", "fire", "damage",
        "flood", "cyclone", "storm", "earthquake",
        "oil", "gas", "pipeline", "fuel",
        "sanction", "tariff", "export", "import",
        "labor", "union", "walkout"
    ]

    count = sum(1 for word in strong_keywords if word in text)

    return count >= 2  