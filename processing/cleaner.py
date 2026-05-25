"""
processing/cleaner.py
---------------------
Stage 1 of the ML pipeline.
Cleans raw article text using ftfy + BeautifulSoup + langdetect.
Exactly as specified in doc section 10 (text cleaning libraries).
"""

import re
import logging
import unicodedata
from typing import Optional

log = logging.getLogger(__name__)

# Lazy imports — only load heavy libs when cleaner is first used
_ftfy   = None
_bs4    = None
_detect = None


def _load_libs():
    global _ftfy, _bs4, _detect
    if _ftfy is None:
        try:
            import ftfy
            _ftfy = ftfy
        except ImportError:
            log.warning("ftfy not installed — skipping encoding fix. pip install ftfy")

    if _bs4 is None:
        try:
            from bs4 import BeautifulSoup
            _bs4 = BeautifulSoup
        except ImportError:
            log.warning("beautifulsoup4 not installed. pip install beautifulsoup4")

    if _detect is None:
        try:
            from langdetect import detect, LangDetectException
            _detect = (detect, LangDetectException)
        except ImportError:
            log.warning("langdetect not installed. pip install langdetect")


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> str:
    """Use ftfy to fix mojibake and broken unicode."""
    _load_libs()
    if _ftfy and text:
        return _ftfy.fix_text(text)
    return text


def strip_html(text: str) -> str:
    """Remove HTML tags using BeautifulSoup."""
    _load_libs()
    if not text:
        return ""
    if _bs4:
        return _bs4(text, "html.parser").get_text(separator=" ")
    # Fallback: regex strip
    return re.sub(r"<[^>]+>", " ", text)


def normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def remove_non_printable(text: str) -> str:
    """Remove control characters while keeping unicode letters."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C"
    )


def truncate_body(text: str, max_chars: int = 300) -> str:
    """
    Keep only first max_chars of body text for embedding.
    Doc specifies: title + first 300 characters of body fed to DistilBERT.
    """
    return text[:max_chars]


def is_english(text: str) -> bool:
    """
    Returns True if text is detected as English.
    Doc section processing: keep English only (lang_filter.py).
    """
    _load_libs()
    if not text or len(text) < 20:
        return False
    if _detect:
        detect_fn, LangDetectException = _detect
        try:
            return detect_fn(text) == "en"
        except LangDetectException:
            return False
    return True  # assume English if langdetect unavailable


# ---------------------------------------------------------------------------
# Main cleaner — applied to each raw article dict
# ---------------------------------------------------------------------------

def clean_article(article: dict) -> Optional[dict]:
    """
    Clean a single raw NewsAPI article dict.
    Returns cleaned dict or None if article should be discarded.

    Input fields (from NewsAPI):
        title, description, content, url, publishedAt, source.name,
        _industry, _query, _fetched_at   (added by fetch_news.py)

    Output adds:
        clean_text   — title + cleaned body, used for embedding
        body_snippet — first 300 chars of body (what DistilBERT sees)
    """
    title       = article.get("title") or ""
    description = article.get("description") or ""
    content     = article.get("content") or ""

    # Use description as body if content is truncated (NewsAPI free tier cuts content)
    body = content if len(content) > len(description) else description

    # Step 1: fix encoding
    title = fix_encoding(title)
    body  = fix_encoding(body)

    # Step 2: strip HTML
    title = strip_html(title)
    body  = strip_html(body)

    # Step 3: normalise whitespace
    title = normalise_whitespace(title)
    body  = normalise_whitespace(body)

    # Step 4: remove control chars
    title = remove_non_printable(title)
    body  = remove_non_printable(body)

    # Discard if no meaningful content
    if not title or len(title) < 10:
        return None

    # Step 5: language filter — English only
    combined = f"{title} {body}"
    if not is_english(combined):
        return None

    # Step 6: build clean_text (title + first 300 chars of body)
    body_snippet = truncate_body(body, 300)
    clean_text   = f"{title}. {body_snippet}".strip()

    return {
        **article,
        "title":        title,
        "body":         body,
        "body_snippet": body_snippet,
        "clean_text":   clean_text,
    }


def clean_articles(articles: list[dict]) -> list[dict]:
    """
    Clean a list of raw articles.
    Returns only articles that pass all filters.
    """
    cleaned = []
    discarded = 0
    for art in articles:
        result = clean_article(art)
        if result:
            cleaned.append(result)
        else:
            discarded += 1

    log.info("Cleaner: %d in → %d clean, %d discarded", len(articles), len(cleaned), discarded)
    return cleaned