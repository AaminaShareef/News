"""
news_fetcher.py  —  ProcureGuard
Fetches supply-chain disruption news from NewsAPI.

When an `industry` is provided the query is automatically enriched with
supply-chain context keywords so results stay relevant even for broad
industry names like "Oil & Gas" or "Semiconductors".
"""

import os
import requests
import pandas as pd

try:
    from config import NEWS_API_KEY
except ImportError:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Suffix appended to every industry-based query to keep results
# focused on procurement / supply-chain disruption signals.
_SC_SUFFIX = "supply chain disruption"

# Per-industry query enrichment terms (added on top of the suffix)
# Keeps searches targeted without over-restricting the API.
_INDUSTRY_BOOST = {
    "Oil & Gas":            "pipeline refinery shortage LNG crude",
    "Semiconductors":       "chip fab wafer shortage TSMC",
    "Pharmaceuticals":      "drug shortage API recall manufacturing",
    "Automotive":           "parts factory halt EV battery",
    "Chemicals":            "plant explosion shortage fertiliser",
    "Metals & Mining":      "mine output shortage tariff rare earth",
    "Aerospace & Defence":  "components MRO titanium shortage",
    "Agriculture & Food":   "harvest crop grain wheat shortage",
    "Electronics":          "components PCB display shortage",
    "Textiles & Apparel":   "cotton yarn garment factory",
    "Retail & FMCG":        "inventory shortage logistics consumer",
    "Construction":         "cement steel lumber material shortage",
    "Logistics & Shipping": "port congestion container freight delay",
    "Energy & Utilities":   "grid outage power plant renewable",
}


def _build_query(raw_query: str, industry: str = None) -> str:
    """
    Build the final NewsAPI query string.

    - If `industry` is supplied AND the raw_query looks like a plain
      industry name (no extra words), inject boost terms + SC suffix.
    - Otherwise append SC suffix to whatever the user typed, unless
      it already contains 'supply chain'.
    """
    q = raw_query.strip()

    if industry and q.lower() in (industry.lower(), ""):
        boost = _INDUSTRY_BOOST.get(industry, "")
        parts = [industry]
        if boost:
            parts.append(boost)
        parts.append(_SC_SUFFIX)
        return " ".join(parts)

    # Free-text query from the search bar
    if _SC_SUFFIX not in q.lower():
        q = f"{q} {_SC_SUFFIX}"
    return q


def fetch_news(
    query:     str = "supply chain disruption",
    page_size: int = 20,
    industry:  str = None,
    from_date: str = None,          # ISO date 'YYYY-MM-DD'
    language:  str = "en",
) -> pd.DataFrame:
    """
    Fetch news articles from NewsAPI /v2/everything.

    Parameters
    ----------
    query      : raw search text (from user input or dropdown)
    page_size  : number of articles to request (max 100)
    industry   : if set, enriches the query with industry-specific terms
    from_date  : only articles published on/after this date
    language   : language filter (default 'en')

    Returns
    -------
    pd.DataFrame with columns:
        title, description, content, source, published_at, url, query_used
    Empty DataFrame on error.
    """
    if not NEWS_API_KEY:
        print("[news_fetcher] ERROR: NEWS_API_KEY is not set.")
        return pd.DataFrame()

    final_query = _build_query(query, industry)

    params = {
        "q":        final_query,
        "language": language,
        "sortBy":   "publishedAt",
        "pageSize": min(int(page_size), 100),
        "apiKey":   NEWS_API_KEY,
    }
    if from_date:
        params["from"] = from_date

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        print(f"[news_fetcher] Request error: {exc}")
        return pd.DataFrame()

    if data.get("status") != "ok":
        print(f"[news_fetcher] NewsAPI error: {data.get('message', 'unknown')}")
        return pd.DataFrame()

    articles = data.get("articles", [])
    if not articles:
        print(f"[news_fetcher] No articles returned for query: '{final_query}'")
        return pd.DataFrame()

    rows = []
    for art in articles:
        rows.append({
            "title":        art.get("title")       or "",
            "description":  art.get("description") or "",
            "content":      art.get("content")     or "",
            "source":       (art.get("source") or {}).get("name", ""),
            "published_at": art.get("publishedAt") or "",
            "url":          art.get("url")         or "",
            "query_used":   final_query,
        })

    df = pd.DataFrame(rows)
    # Drop rows where title is completely empty or '[Removed]'
    df = df[df["title"].str.strip().ne("") & df["title"].ne("[Removed]")]
    df = df.reset_index(drop=True)

    print(f"[news_fetcher] Fetched {len(df)} articles | query: '{final_query}'")
    return df