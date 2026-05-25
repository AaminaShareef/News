"""
fetch_news.py
-------------
Fetches real-time news articles related to supply chain disruptions
from NewsAPI.org and saves them to data/news.json.

Place this file in the project root (same level as config.py).
Run: python fetch_news.py
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta

import requests

# ---------------------------------------------------------------------------
# Import API key from config.py (must be in the same directory)
# ---------------------------------------------------------------------------
try:
    from config import NEWS_API_KEY  # adjust the variable name if different
except ImportError:
    raise ImportError("config.py not found. Make sure this script is in the project root.")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_URL = "https://newsapi.org/v2/everything"
OUTPUT_PATH = os.path.join("data", "news.json")

# How many days back to fetch (free tier supports up to 30 days)
DAYS_BACK = 30

# NewsAPI free tier: max 100 results per request; we paginate to get more
PAGE_SIZE = 100
MAX_PAGES = 5          # 5 pages × 100 = up to 500 articles per query

# Delay between requests to be respectful to the API (seconds)
REQUEST_DELAY = 1.0

# ---------------------------------------------------------------------------
# Industry-wise supply chain disruption queries
# Each entry: (industry_label, search_query)
# ---------------------------------------------------------------------------
INDUSTRY_QUERIES = [
    ("Semiconductor / Electronics",
     "semiconductor supply chain disruption OR chip shortage OR electronics manufacturing delay"),

    ("Automotive",
     "automotive supply chain disruption OR car parts shortage OR vehicle production halt"),

    ("Pharmaceuticals / Healthcare",
     "pharmaceutical supply chain disruption OR drug shortage OR medical supply delay"),

    ("Food & Agriculture",
     "food supply chain disruption OR agricultural supply shortage OR crop shortage logistics"),

    ("Energy / Oil & Gas",
     "energy supply chain disruption OR oil supply shortage OR gas pipeline disruption"),

    ("Retail / Consumer Goods",
     "retail supply chain disruption OR consumer goods shortage OR inventory shortage logistics"),

    ("Shipping / Logistics",
     "shipping disruption OR port congestion OR freight delay OR logistics crisis"),

    ("Aerospace & Defense",
     "aerospace supply chain disruption OR defense manufacturing delay OR aviation parts shortage"),

    ("Construction / Raw Materials",
     "construction supply chain disruption OR steel shortage OR lumber shortage OR raw material delay"),

    ("Technology / IT Hardware",
     "technology hardware supply chain disruption OR server shortage OR data center supply delay"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_articles_for_query(query: str, industry: str) -> list[dict]:
    """Fetch paginated articles for a single query string."""
    from_date = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    all_articles = []

    for page in range(1, MAX_PAGES + 1):
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": PAGE_SIZE,
            "page": page,
            "apiKey": NEWS_API_KEY,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            log.error("Request failed (industry=%s, page=%d): %s", industry, page, exc)
            break

        data = response.json()

        if data.get("status") != "ok":
            log.warning(
                "Non-OK response for industry=%s page=%d: %s",
                industry, page, data.get("message", "unknown error"),
            )
            break

        articles = data.get("articles", [])
        if not articles:
            log.info("  No more articles at page %d.", page)
            break

        # Annotate each article with metadata useful for training
        for article in articles:
            article["_industry"] = industry
            article["_query"] = query
            article["_fetched_at"] = datetime.utcnow().isoformat() + "Z"

        all_articles.extend(articles)
        log.info(
            "  [%s] page %d → %d articles (total so far: %d)",
            industry, page, len(articles), len(all_articles),
        )

        # Stop early if we got fewer results than a full page
        if len(articles) < PAGE_SIZE:
            break

        time.sleep(REQUEST_DELAY)

    return all_articles


def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove duplicate articles by URL."""
    seen = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(a)
    return unique


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    log.info("Saved %d articles → %s", len(data["articles"]), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Starting news fetch  (days_back=%d, max_pages=%d)", DAYS_BACK, MAX_PAGES)

    all_articles: list[dict] = []
    industry_summary: dict[str, int] = {}

    for industry, query in INDUSTRY_QUERIES:
        log.info("Fetching: %s", industry)
        articles = fetch_articles_for_query(query, industry)
        industry_summary[industry] = len(articles)
        all_articles.extend(articles)
        time.sleep(REQUEST_DELAY)

    # Deduplicate across industries
    unique_articles = deduplicate(all_articles)
    log.info(
        "Total fetched: %d  |  After dedup: %d",
        len(all_articles), len(unique_articles),
    )

    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "days_back": DAYS_BACK,
            "total_articles": len(unique_articles),
            "industry_counts": industry_summary,
            "source": "newsapi.org",
        },
        "articles": unique_articles,
    }

    save_json(output, OUTPUT_PATH)
    log.info("Done. ✓")


if __name__ == "__main__":
    main()