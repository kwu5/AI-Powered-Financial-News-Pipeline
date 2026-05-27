"""URL canonicalization for Stage 1 dedup.

Two articles with different query-string tracking params or trailing-slash
variations should produce the SAME canonical URL — that is what makes Stage 1
dedup work.
"""

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


# Query params we always strip — analytics / referral / campaign noise.
TRACKING_PARAMS: set[str] = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid",
    "mc_cid", "mc_eid",
    "_hsenc", "_hsmi",
    "ref", "source",
}


def canonicalize_url(url: str) -> str:
    """Return a canonical form of `url` suitable for equality-based dedup.
    """
    if not url:
        raise ValueError("url is empty")

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"cannot parse scheme+host from {url!r}")

    # 1. Lowercase scheme and host.
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # 2. Strip default ports.
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    # 5. Normalize trailing slash on path.
    path = parsed.path
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"

    # 4. Strip tracking params; preserve order of survivors.
    survivors = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(survivors)

    # 3. Drop fragment by passing "" to urlunparse.
    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


if __name__ == "__main__":
    # Smoke fixtures — fill in once implemented.
    samples = [
        "https://www.example.com/article?utm_source=twitter&id=42",
        "HTTPS://WWW.Example.com:443/article/?id=42#section-2",
        "https://www.example.com/article?id=42",
    ]
    for s in samples:
        print(f"{s!r} -> {canonicalize_url(s)!r}")
