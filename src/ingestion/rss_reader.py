import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import feedparser
import yaml
from dateutil import parser as dateparser

from src.config import Settings

logger = logging.getLogger(__name__)


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

    @property
    def feed_urls(self) -> list[str]:
        return [f["url"] for f in self.feeds]

    @feed_urls.setter
    def feed_urls(self, urls: list[str]) -> None:
        self.feeds = [{"name": u, "url": u, "category": "general"} for u in urls]

    def fetch_from_feeds(self) -> List[dict]:
        all_articles: list[dict] = []
        for feed_cfg in self.feeds:
            url = feed_cfg["url"]
            try:
                feed = feedparser.parse(url)
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
                    if article["title"] and article["url"]:
                        all_articles.append(article)
            except Exception as e:
                logger.error(f"Failed to fetch feed {url} : {e}")
                continue
        return all_articles

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
    reader = RSSReader()
    result = reader.fetch_from_feeds()
    print(f"Fetched {len(result)} articles from {len({a['source'] for a in result})} sources")
    for a in result[:3]:
        print(f"{a['source']} | {a['published_at']} | {a['title']}")
