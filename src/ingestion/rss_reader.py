from dataclasses import dataclass
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.ingestion import extractor

import feedparser
import yaml
from dateutil import parser as dateparser

from src.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class FeedHealth:
    name: str
    entries: int = 0
    kept: int = 0
    extracted: int = 0          # full-text only, not rss-only
    error: str | None = None

    @property
    def extraction_rate(self) -> float:
        return self.extracted / self.kept if self.kept else 0.0



def _load_feeds(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    feeds = data.get("feeds", [])
    return [
        {"name": e["name"], "url": e["url"], "category": e.get("category", "general")}
        for e in feeds
        if e.get("url")
    ]


class RSSReader:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()  # type: ignore[call-arg]
        self.feeds = _load_feeds(self.settings.FEEDS_CONFIG_PATH)
        self.feed_health: list[FeedHealth] = []
        
    @property
    def feed_urls(self) -> list[str]:
        return [f["url"] for f in self.feeds]

    @feed_urls.setter
    def feed_urls(self, urls: list[str]) -> None:
        self.feeds = [{"name": u, "url": u, "category": "general"} for u in urls]

    def fetch_from_feeds(self) -> List[dict]:
        all_articles: list[dict] = []
        self.feed_health = []
        for feed_cfg in self.feeds:
            url = feed_cfg["url"]
            health = FeedHealth(name=feed_cfg["name"])
            self.feed_health.append(health)
            try:
                feed = feedparser.parse(url)
                health.entries = len(feed.entries)
                source_name = feed_cfg["name"] or feed.get("title", url)

                for entry in feed.entries:
                    article = {
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                        "content": entry.get("summary", ""),
                        "url": entry.get("link", ""),
                        "source": source_name,
                        "category": feed_cfg.get("category", "general"),
                        "published_at": self._parse_date(entry),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if not (article["title"] and article["url"]):
                        continue
                    if "/news/videos/" in article["url"]:
                        continue
                    health.kept += 1
                    text, method = extractor.extract_full_text(article["url"])
                    if text:
                        article["content"] = text
                    article["extraction_method"] = method
                    if method != "rss-only":
                        health.extracted += 1
                    all_articles.append(article)
            except Exception as e:
                health.error = str(e)
                logger.error(f"Failed to fetch feed {url} : {e}")
                continue
        self._log_feed_health()
        return all_articles

    def _log_feed_health(self) -> None:
        """Log one line per feed: entries, kept, full-text rate, or error."""
        logger.info("Per-source feed health:")
        for h in self.feed_health:
            if h.error:
                logger.warning(f"  {h.name}: ERROR - {h.error}")
            else:
                logger.info(
                    f"  {h.name}: {h.entries} entries, {h.kept} kept, "
                    f"{h.extracted} full-text ({h.extraction_rate:.0%})"
                )

    def _parse_date(self, entry) -> str:
        raw = None
        if hasattr(entry, "get"):
            raw = entry.get("published") or entry.get("updated")
        if raw:
            try:
                dt = dateparser.parse(raw)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt:
                    return dt.astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError):
                pass

        for attr in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, attr, None)
            try:
                dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except (TypeError, ValueError):
                continue

        return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reader = RSSReader()
    result = reader.fetch_from_feeds()

    sources = len({a["source"] for a in result})
    methods = Counter(a["extraction_method"] for a in result)
    print(f"\nFetched {len(result)} articles from {sources} sources")
    print(f"Extraction methods: {dict(methods)}")
    for a in result[:3]:
        print(f"{a['source']} | {a['extraction_method']} | {len(a['content'])} chars | {a['title']}")
