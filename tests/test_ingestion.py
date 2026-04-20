"""Tests for the ingestion layer: WorldNewsAPIClient and RSSReader."""

from unittest.mock import patch, MagicMock
import pytest

from src.ingestion.world_news_api import WorldNewsAPIClient, SearchParams
from src.ingestion.rss_reader import RSSReader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.NEWS_API_KEY = "test-key"
    return s


@pytest.fixture
def sample_api_response():
    return {
        "offset": 0,
        "number": 2,
        "available": 100,
        "news": [
            {
                "title": "Fed Raises Rates",
                "text": "The Federal Reserve raised rates by 25bp.",
                "url": "https://example.com/fed",
                "source": "Reuters",
                "publish_date": "2026-04-20T10:00:00",
            },
            {
                "title": "Apple Earnings Beat",
                "text": "Apple reported record revenue.",
                "url": "https://example.com/apple",
                "source": "CNBC",
                "publish_date": "2026-04-20T11:00:00",
            },
        ],
    }


@pytest.fixture
def sample_rss_feed():
    feed = MagicMock()
    feed.get.return_value = "Test Feed"

    entry1 = MagicMock()
    entry1.get.side_effect = lambda k, d="": {
        "title": "Bitcoin Rally",
        "summary": "BTC hit $100k.",
        "link": "https://example.com/btc",
    }.get(k, d)
    entry1.published_parsed = (2026, 4, 20, 12, 0, 0, 0, 0, 0)

    entry2 = MagicMock()
    entry2.get.side_effect = lambda k, d="": {
        "title": "Oil Prices Surge",
        "summary": "Oil up 5%.",
        "link": "https://example.com/oil",
    }.get(k, d)
    entry2.published_parsed = None

    feed.entries = [entry1, entry2]
    return feed


# ---------------------------------------------------------------------------
# WorldNewsAPIClient
# ---------------------------------------------------------------------------

class TestWorldNewsAPIClient:

    @patch("src.ingestion.world_news_api.requests.get")
    def test_fetch_success_returns_news_list(self, mock_get, mock_settings, sample_api_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_api_response
        mock_get.return_value = mock_resp

        client = WorldNewsAPIClient(mock_settings)
        result = client.fetch_financial_news()

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Fed Raises Rates"

    @patch("src.ingestion.world_news_api.requests.get")
    def test_fetch_error_returns_empty(self, mock_get, mock_settings):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        client = WorldNewsAPIClient(mock_settings)
        result = client.fetch_financial_news()

        assert result == []

    @patch("src.ingestion.world_news_api.requests.get")
    def test_api_key_in_header(self, mock_get, mock_settings):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"news": []}
        mock_get.return_value = mock_resp

        client = WorldNewsAPIClient(mock_settings)
        client.fetch_financial_news()

        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["x-api-key"] == "test-key"


# ---------------------------------------------------------------------------
# SearchParams
# ---------------------------------------------------------------------------

class TestSearchParams:

    def test_defaults(self):
        params = SearchParams()
        assert params.text == "stock+market"
        assert params.language == "en"
        assert params.number == 10

    def test_alias_serialization(self):
        params = SearchParams()
        dumped = params.model_dump(by_alias=True, exclude_none=True)
        assert "earliest-publish-date" in dumped
        assert "source-country" in dumped
        assert "earliest_publish_date" not in dumped


# ---------------------------------------------------------------------------
# RSSReader
# ---------------------------------------------------------------------------

class TestRSSReader:

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_fetch_returns_normalized_articles(self, mock_parse, sample_rss_feed):
        mock_parse.return_value = sample_rss_feed
        reader = RSSReader()
        reader.feed_urls = ["https://fake-feed.com/rss"]

        articles = reader.fetch_from_feeds()

        assert len(articles) == 2
        for a in articles:
            assert "title" in a
            assert "description" in a
            assert "content" in a
            assert "url" in a
            assert "source" in a
            assert "published_at" in a
            assert "fetched_at" in a

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_skips_entries_without_title_or_url(self, mock_parse):
        feed = MagicMock()
        feed.get.return_value = "Test Feed"

        bad_entry = MagicMock()
        bad_entry.get.side_effect = lambda k, d="": {
            "title": "",
            "summary": "No title article.",
            "link": "https://example.com/no-title",
        }.get(k, d)
        bad_entry.published_parsed = None

        feed.entries = [bad_entry]
        mock_parse.return_value = feed

        reader = RSSReader()
        reader.feed_urls = ["https://fake-feed.com/rss"]
        articles = reader.fetch_from_feeds()

        assert len(articles) == 0

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_failing_feed_does_not_crash(self, mock_parse):
        mock_parse.side_effect = Exception("Network error")
        reader = RSSReader()
        reader.feed_urls = ["https://broken-feed.com/rss"]

        articles = reader.fetch_from_feeds()
        assert articles == []

    def test_parse_date_valid(self):
        reader = RSSReader()
        entry = MagicMock()
        entry.published_parsed = (2026, 4, 20, 14, 30, 0, 0, 0, 0)

        result = reader._parse_date(entry)
        assert result == "2026-04-20T14:30:00"

    def test_parse_date_fallback(self):
        reader = RSSReader()
        entry = MagicMock()
        entry.published_parsed = None

        result = reader._parse_date(entry)
        assert "T" in result  # ISO format with T separator
