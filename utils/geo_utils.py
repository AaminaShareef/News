"""
Lightweight country → region mapper.
No external API required — pure dictionary lookup.
"""

# Mapping: lowercase country name → (display name, region)
COUNTRY_REGION_MAP = {
    # Asia
    "china": ("China", "Asia"),
    "india": ("India", "Asia"),
    "japan": ("Japan", "Asia"),
    "south korea": ("South Korea", "Asia"),
    "korea": ("South Korea", "Asia"),
    "taiwan": ("Taiwan", "Asia"),
    "singapore": ("Singapore", "Asia"),
    "indonesia": ("Indonesia", "Asia"),
    "malaysia": ("Malaysia", "Asia"),
    "thailand": ("Thailand", "Asia"),
    "vietnam": ("Vietnam", "Asia"),
    "philippines": ("Philippines", "Asia"),
    "bangladesh": ("Bangladesh", "Asia"),
    "pakistan": ("Pakistan", "Asia"),
    "sri lanka": ("Sri Lanka", "Asia"),
    "myanmar": ("Myanmar", "Asia"),
    "cambodia": ("Cambodia", "Asia"),
    "hong kong": ("Hong Kong", "Asia"),
    "kazakhstan": ("Kazakhstan", "Asia"),
    "uzbekistan": ("Uzbekistan", "Asia"),

    # Middle East
    "iran": ("Iran", "Middle East"),
    "iraq": ("Iraq", "Middle East"),
    "israel": ("Israel", "Middle East"),
    "saudi arabia": ("Saudi Arabia", "Middle East"),
    "uae": ("UAE", "Middle East"),
    "united arab emirates": ("UAE", "Middle East"),
    "qatar": ("Qatar", "Middle East"),
    "kuwait": ("Kuwait", "Middle East"),
    "oman": ("Oman", "Middle East"),
    "bahrain": ("Bahrain", "Middle East"),
    "yemen": ("Yemen", "Middle East"),
    "syria": ("Syria", "Middle East"),
    "lebanon": ("Lebanon", "Middle East"),
    "jordan": ("Jordan", "Middle East"),
    "turkey": ("Turkey", "Middle East"),
    "turkiye": ("Turkey", "Middle East"),
    "egypt": ("Egypt", "Middle East"),

    # Europe
    "russia": ("Russia", "Europe"),
    "ukraine": ("Ukraine", "Europe"),
    "germany": ("Germany", "Europe"),
    "france": ("France", "Europe"),
    "uk": ("UK", "Europe"),
    "united kingdom": ("UK", "Europe"),
    "britain": ("UK", "Europe"),
    "england": ("UK", "Europe"),
    "italy": ("Italy", "Europe"),
    "spain": ("Spain", "Europe"),
    "netherlands": ("Netherlands", "Europe"),
    "poland": ("Poland", "Europe"),
    "sweden": ("Sweden", "Europe"),
    "norway": ("Norway", "Europe"),
    "finland": ("Finland", "Europe"),
    "denmark": ("Denmark", "Europe"),
    "switzerland": ("Switzerland", "Europe"),
    "austria": ("Austria", "Europe"),
    "belgium": ("Belgium", "Europe"),
    "greece": ("Greece", "Europe"),
    "portugal": ("Portugal", "Europe"),
    "hungary": ("Hungary", "Europe"),
    "romania": ("Romania", "Europe"),
    "czech republic": ("Czech Republic", "Europe"),
    "slovakia": ("Slovakia", "Europe"),
    "serbia": ("Serbia", "Europe"),

    # Americas
    "usa": ("USA", "Americas"),
    "united states": ("USA", "Americas"),
    "america": ("USA", "Americas"),
    "us": ("USA", "Americas"),
    "canada": ("Canada", "Americas"),
    "mexico": ("Mexico", "Americas"),
    "brazil": ("Brazil", "Americas"),
    "argentina": ("Argentina", "Americas"),
    "colombia": ("Colombia", "Americas"),
    "chile": ("Chile", "Americas"),
    "peru": ("Peru", "Americas"),
    "venezuela": ("Venezuela", "Americas"),
    "cuba": ("Cuba", "Americas"),

    # Africa
    "nigeria": ("Nigeria", "Africa"),
    "south africa": ("South Africa", "Africa"),
    "kenya": ("Kenya", "Africa"),
    "ethiopia": ("Ethiopia", "Africa"),
    "ghana": ("Ghana", "Africa"),
    "tanzania": ("Tanzania", "Africa"),
    "uganda": ("Uganda", "Africa"),
    "angola": ("Angola", "Africa"),
    "mozambique": ("Mozambique", "Africa"),
    "congo": ("Congo", "Africa"),
    "libya": ("Libya", "Africa"),
    "algeria": ("Algeria", "Africa"),
    "morocco": ("Morocco", "Africa"),
    "tunisia": ("Tunisia", "Africa"),
    "sudan": ("Sudan", "Africa"),
    "somalia": ("Somalia", "Africa"),

    # Oceania
    "australia": ("Australia", "Oceania"),
    "new zealand": ("New Zealand", "Oceania"),
    "papua new guinea": ("Papua New Guinea", "Oceania"),
    "fiji": ("Fiji", "Oceania"),
}

# Ordered by length descending so longer names match before substrings
_SORTED_KEYS = sorted(COUNTRY_REGION_MAP.keys(), key=len, reverse=True)


def detect_country_region(text: str):
    """
    Scan article text for the first recognisable country mention.
    Returns (country_display_name, region) or ("Global", "Global").
    """
    if not text:
        return "Global", "Global"

    text_lower = text.lower()
    for key in _SORTED_KEYS:
        if key in text_lower:
            country, region = COUNTRY_REGION_MAP[key]
            return country, region

    return "Global", "Global"
