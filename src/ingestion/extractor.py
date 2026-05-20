"""Full-text article extraction with a fallback chain.

RSS feed entries usually carry only a short summary. This module fetches the
article page once and extracts the full body text, trying three extractors in
descending order of quality:

    trafilatura  ->  newspaper3k  ->  readability-lxml

Used by RSSReader during ingestion (Ship A). World News API articles already
include full text and do not go through this module.
"""

import logging

import requests
import trafilatura
from bs4 import BeautifulSoup
from newspaper import Article
from readability import Document




logger = logging.getLogger(__name__)

# An extraction attempt counts as a success only if its text clears this many
# characters. Shorter results fall through to the next extractor.
MIN_TEXT_LENGTH = 200

# User-Agent sent when fetching article pages. SEC.gov rejects generic
# User-Agents, so this declares the project name and a contact email.
USER_AGENT = "FinNewsRAG finnews-rag@example.com"

# Seconds to wait for an article page to respond before giving up.
FETCH_TIMEOUT = 10


def _fetch_html(url: str) -> str | None:
    """Download the raw HTML for ``url``.

    Use ``requests.get`` with USER_AGENT and FETCH_TIMEOUT. Catch
    ``requests.RequestException`` (timeout, connection error) and treat a
    non-200 status as a failure.

    Returns:
        The page HTML as a string, or None if the request fails.
    """
    try:
        resp = requests.get(
            url, 
            headers={"User-Agent": USER_AGENT}, 
            timeout=FETCH_TIMEOUT,
            )
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch html {url} : {e}")
        return None
    if resp.status_code != 200:
        logger.warning(f"Fetch returned {resp.status_code} for {url}")
        return None

    return resp.text

def _extract_with_trafilatura(html: str) -> str | None:
    """Extract main body text with trafilatura.

    Call ``trafilatura.extract(html, include_comments=False,
    include_tables=False, favor_precision=True)``. trafilatura returns None
    when it cannot locate main content; wrap the call defensively anyway.

    Returns:
        Plain-text body, or None on failure.
    """
    return trafilatura.extract(html, include_comments=False,include_tables=False, favor_precision=True)


def _extract_with_newspaper(url: str, html: str) -> str | None:
    """Extract main body text with newspaper3k, reusing pre-fetched HTML.

    Build ``Article(url)``, then ``article.download(input_html=html)`` (avoids a
    second HTTP request), then ``article.parse()``, then read ``article.text``.
    ``.parse()`` before ``.download()`` raises; catch ``ArticleException``.

    Returns:
        Plain-text body, or None on failure.
    """
    try:
        article = Article(url)
        article.download(input_html=html)
        article.parse()
    except Exception as e:
        logger.warning(f"newspaper3k failed for {url}: {e}")
        return None
    return article.text or None


def _extract_with_readability(html: str) -> str | None:
    """Extract main body text with readability-lxml.

    ``Document(html).summary()`` returns cleaned *HTML*, not text. Strip the
    tags with BeautifulSoup (``get_text(separator="\\n", strip=True)``). Catch
    ``readability.readability.Unparseable``.

    Returns:
        Plain-text body, or None on failure.
    """
    try:
        content_html = Document(html).summary()
    except Exception as e:
        logger.warning(f"readability failed: {e}")
        return None
    text = BeautifulSoup(content_html, "html.parser").get_text(separator="\n", strip=True)
    return text or None


def _is_good(text: str | None) -> bool:
    """Return True if ``text`` is non-empty and clears MIN_TEXT_LENGTH."""
    return text is not None and len(text) >= MIN_TEXT_LENGTH


def extract_full_text(url: str) -> tuple[str | None, str]:
    """Extract the full article body for ``url``.

    Fetch the page once via ``_fetch_html``, then try the extractors in order,
    accepting the first result that passes ``_is_good``. Log which extractor
    succeeded (the caller records it as the article's ``extraction_method``).

    Returns:
        ``(text, method)`` where method is one of
        ``"trafilatura"`` | ``"newspaper3k"`` | ``"readability"`` | ``"rss-only"``.
        On total failure returns ``(None, "rss-only")`` and the caller should
        keep the RSS summary as the article content.
    """
    html = _fetch_html(url)
    if html is None:
        return None, "rss-only"

    text = _extract_with_trafilatura(html)
    if _is_good(text):
        logger.info(f"Extracted {url} via trafilatura")
        return text, "trafilatura"

    text = _extract_with_newspaper(url, html)
    if _is_good(text):
        logger.info(f"Extracted {url} via newspaper3k")
        return text, "newspaper3k"

    text = _extract_with_readability(html)
    if _is_good(text):
        logger.info(f"Extracted {url} via readability")
        return text, "readability"

    logger.warning(f"All extractors failed for {url}; using rss-only fallback")
    return None, "rss-only"
