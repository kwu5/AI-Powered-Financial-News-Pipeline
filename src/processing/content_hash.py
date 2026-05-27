"""Content hashing for Stage 2 dedup.

Catches identical article TEXT published under different URLs — wire syndication
(Reuters → Yahoo → CNBC), re-publishes with new tracking params that survived
Stage 1, etc. Two articles whose normalized title+content match byte-for-byte
must produce the same hash.
"""

import hashlib
import re


def _normalize_text(text: str) -> str:
    """Normalize text so trivial formatting differences don't change the hash.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_content_hash(article: dict) -> str:
    """Return a sha256 hex digest of the article's normalized title + content.

    The hash is computed over `_normalize_text(title) + "\\n" + _normalize_text(content)`.
    Two articles that paraphrase each other will hash differently; two articles
    that differ only in whitespace, casing, or punctuation will hash the same.

    
    """
    title = _normalize_text(article["title"])
    content = _normalize_text(article["content"])
    if not title:
        raise ValueError("article title is empty after normalization")
    if not content:
        raise ValueError("article content is empty after normalization")
    payload = f"{title}\n{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    # Smoke fixtures — fill in once implemented.
    a = {"title": "Fed Raises Rates", "content": "The Federal Reserve raised rates."}
    b = {"title": "  fed raises rates  ", "content": "The Federal Reserve raised rates!!"}
    print(f"a: {compute_content_hash(a)}")
    print(f"b: {compute_content_hash(b)}")
    print(f"equal? {compute_content_hash(a) == compute_content_hash(b)}")
