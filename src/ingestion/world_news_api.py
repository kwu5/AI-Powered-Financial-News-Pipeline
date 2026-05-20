import logging
from datetime import date, datetime, timezone
from typing import Optional

import requests
from dateutil import parser as dateparser
from pydantic import BaseModel, Field

from src.config import Settings

logger = logging.getLogger(__name__)


class SearchParams(BaseModel):
    text: str = "stock+market"
    text_match_indexes: Optional[str] = Field(default=None, alias="text-match-indexes")
    source_country: Optional[str] = Field(default="us", alias="source-country")
    language: str = "en"
    min_sentiment: Optional[float] = Field(default=None, alias="min-sentiment")
    max_sentiment: Optional[float] = Field(default=None, alias="max-sentiment")
    earliest_publish_date: str = Field(
        default_factory=lambda: date.today().isoformat(),
        alias="earliest-publish-date",
    )
    latest_publish_date: Optional[str] = Field(default=None, alias="latest-publish-date")
    news_sources: Optional[str] = Field(default=None, alias="news-sources")
    authors: Optional[str] = None
    categories: Optional[str] = None
    entities: Optional[str] = None
    location_filter: Optional[str] = Field(default=None, alias="location-filter")
    sort: Optional[str] = None
    sort_direction: Optional[str] = Field(default=None, alias="sort-direction")
    offset: Optional[int] = None
    number: int = 10


def _to_utc_iso(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        dt = dateparser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def _normalize(raw: dict, fetched_at: str) -> dict:
    return {
        "title": raw.get("title", ""),
        "description": raw.get("summary") or raw.get("text", "")[:300],
        "content": raw.get("text", ""),
        "url": raw.get("url", ""),
        "source": raw.get("source") or raw.get("source_country", "World News API"),
        "category": "general",
        "published_at": _to_utc_iso(raw.get("publish_date")) or fetched_at,
        "fetched_at": fetched_at,
    }


class WorldNewsAPIClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = "https://api.worldnewsapi.com/search-news?"
        self.headers = {"x-api-key": settings.NEWS_API_KEY}

    def fetch_financial_news(self, days_back: int = 1) -> list[dict]:
        params = SearchParams()
        query = params.model_dump(by_alias=True, exclude_none=True)

        response = requests.get(self.base_url, params=query, headers=self.headers)

        if response.status_code != 200:
            logger.error(f"World News API error: {response.status_code} - {response.text[:200]}")
            return []

        fetched_at = datetime.now(timezone.utc).isoformat()
        news = response.json().get("news", [])
        return [_normalize(item, fetched_at) for item in news]


if __name__ == "__main__":
    client = WorldNewsAPIClient(Settings())  # type: ignore
    result = client.fetch_financial_news()
    print(f"Fetched {len(result)} articles")
    for a in result[:3]:
        print(f"{a['source']} | {a['published_at']} | {a['title']}")
