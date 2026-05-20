"""Tests for the ingestion layer: WorldNewsAPIClient and RSSReader."""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.world_news_api import WorldNewsAPIClient, SearchParams, _to_utc_iso
from src.ingestion.rss_reader import RSSReader, _load_feeds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.NEWS_API_KEY = "test-key"
    return s


@pytest.fixture
def feeds_yaml(tmp_path):
    path = tmp_path / "feeds.yaml"
    path.write_text(
        "feeds:\n"
        "  - name: Fake CNBC\n"
        "    url: https://fake-cnbc.test/rss\n"
        "    category: general\n"
        "  - name: Fake Crypto\n"
        "    url: https://fake-crypto.test/rss\n"
        "    category: crypto\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def reader_with_fake_feeds(feeds_yaml):
    settings = MagicMock()
    settings.FEEDS_CONFIG_PATH = str(feeds_yaml)
    return RSSReader(settings=settings)


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
                "publish_date": "2026-04-20 10:00:00",
            },
            {
                "title": "Apple Earnings Beat",
                "text": "Apple reported record revenue.",
                "url": "https://example.com/apple",
                "source": "CNBC",
                "publish_date": "2026-04-20T11:00:00+05:00",
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
        "published": "Mon, 20 Apr 2026 12:00:00 GMT",
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
    def test_fetch_success_returns_normalized_list(self, mock_get, mock_settings, sample_api_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_api_response
        mock_get.return_value = mock_resp

        client = WorldNewsAPIClient(mock_settings)
        result = client.fetch_financial_news()

        assert len(result) == 2
        assert result[0]["title"] == "Fed Raises Rates"
        # Normalized shape
        for a in result:
            assert {"title", "description", "content", "url", "source",
                    "category", "published_at", "fetched_at"} <= a.keys()

    @patch("src.ingestion.world_news_api.requests.get")
    def test_fetch_error_returns_empty(self, mock_get, mock_settings):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        client = WorldNewsAPIClient(mock_settings)
        assert client.fetch_financial_news() == []

    @patch("src.ingestion.world_news_api.requests.get")
    def test_api_key_in_header(self, mock_get, mock_settings):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"news": []}
        mock_get.return_value = mock_resp

        WorldNewsAPIClient(mock_settings).fetch_financial_news()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["x-api-key"] == "test-key"

    @patch("src.ingestion.world_news_api.requests.get")
    def test_timestamps_are_tz_aware_utc(self, mock_get, mock_settings, sample_api_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_api_response
        mock_get.return_value = mock_resp

        result = WorldNewsAPIClient(mock_settings).fetch_financial_news()
        for a in result:
            pub = datetime.fromisoformat(a["published_at"])
            fetched = datetime.fromisoformat(a["fetched_at"])
            assert pub.tzinfo is not None
            assert pub.utcoffset() == timezone.utc.utcoffset(pub)
            assert fetched.tzinfo is not None

        # +05:00 publish_date should be converted to UTC (06:00)
        apple = next(a for a in result if a["title"] == "Apple Earnings Beat")
        assert datetime.fromisoformat(apple["published_at"]).hour == 6


class TestUtcIsoHelper:

    def test_naive_string_assumed_utc(self):
        assert _to_utc_iso("2026-04-20 10:00:00") == "2026-04-20T10:00:00+00:00"

    def test_tz_aware_string_converted_to_utc(self):
        assert _to_utc_iso("2026-04-20T11:00:00+05:00") == "2026-04-20T06:00:00+00:00"

    def test_empty_returns_none(self):
        assert _to_utc_iso(None) is None
        assert _to_utc_iso("") is None

    def test_garbage_returns_none(self):
        assert _to_utc_iso("not-a-date") is None


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
# Feed config loader
# ---------------------------------------------------------------------------

class TestLoadFeeds:

    def test_loads_feed_entries(self, feeds_yaml):
        feeds = _load_feeds(feeds_yaml)
        assert len(feeds) == 2
        assert feeds[0]["name"] == "Fake CNBC"
        assert feeds[0]["url"] == "https://fake-cnbc.test/rss"
        assert feeds[0]["category"] == "general"
        assert feeds[1]["category"] == "crypto"

    def test_skips_entries_without_url(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text(
            "feeds:\n"
            "  - name: Has URL\n"
            "    url: https://ok.test/rss\n"
            "  - name: Missing URL\n",
            encoding="utf-8",
        )
        feeds = _load_feeds(path)
        assert len(feeds) == 1
        assert feeds[0]["name"] == "Has URL"

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text("", encoding="utf-8")
        assert _load_feeds(path) == []


# ---------------------------------------------------------------------------
# RSSReader
# ---------------------------------------------------------------------------

class TestRSSReader:

    def test_loads_feeds_from_yaml_config(self, reader_with_fake_feeds):
        urls = reader_with_fake_feeds.feed_urls
        assert "https://fake-cnbc.test/rss" in urls
        assert "https://fake-crypto.test/rss" in urls

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_fetch_returns_normalized_articles(self, mock_parse, reader_with_fake_feeds, sample_rss_feed):
        mock_parse.return_value = sample_rss_feed
        articles = reader_with_fake_feeds.fetch_from_feeds()

        # 2 feeds × 2 entries (parser mocked to return same feed for each call)
        assert len(articles) == 4
        for a in articles:
            assert {"title", "description", "content", "url", "source",
                    "category", "published_at", "fetched_at"} <= a.keys()

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_source_name_comes_from_yaml_not_feed_title(self, mock_parse, reader_with_fake_feeds, sample_rss_feed):
        mock_parse.return_value = sample_rss_feed
        articles = reader_with_fake_feeds.fetch_from_feeds()
        sources = {a["source"] for a in articles}
        assert "Fake CNBC" in sources
        assert "Fake Crypto" in sources

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_category_propagates_from_yaml(self, mock_parse, reader_with_fake_feeds, sample_rss_feed):
        mock_parse.return_value = sample_rss_feed
        articles = reader_with_fake_feeds.fetch_from_feeds()
        cats = {a["source"]: a["category"] for a in articles}
        assert cats["Fake CNBC"] == "general"
        assert cats["Fake Crypto"] == "crypto"

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_timestamps_are_tz_aware_utc(self, mock_parse, reader_with_fake_feeds, sample_rss_feed):
        mock_parse.return_value = sample_rss_feed
        articles = reader_with_fake_feeds.fetch_from_feeds()
        for a in articles:
            pub = datetime.fromisoformat(a["published_at"])
            fetched = datetime.fromisoformat(a["fetched_at"])
            assert pub.tzinfo is not None
            assert fetched.tzinfo is not None

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_skips_entries_without_title_or_url(self, mock_parse, reader_with_fake_feeds):
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

        assert reader_with_fake_feeds.fetch_from_feeds() == []

    @patch("src.ingestion.rss_reader.feedparser.parse")
    def test_failing_feed_does_not_crash(self, mock_parse, reader_with_fake_feeds):
        mock_parse.side_effect = Exception("Network error")
        assert reader_with_fake_feeds.fetch_from_feeds() == []

    def test_parse_date_from_published_string(self, reader_with_fake_feeds):
        entry = MagicMock()
        entry.get.side_effect = lambda k, d=None: {
            "published": "Mon, 20 Apr 2026 14:30:00 +0000",
        }.get(k, d)
        result = reader_with_fake_feeds._parse_date(entry)
        assert result == "2026-04-20T14:30:00+00:00"

    def test_parse_date_converts_tz_to_utc(self, reader_with_fake_feeds):
        entry = MagicMock()
        entry.get.side_effect = lambda k, d=None: {
            "published": "2026-04-20T14:30:00+05:00",
        }.get(k, d)
        result = reader_with_fake_feeds._parse_date(entry)
        assert result == "2026-04-20T09:30:00+00:00"

    def test_parse_date_falls_back_to_published_parsed(self, reader_with_fake_feeds):
        entry = MagicMock()
        entry.get.side_effect = lambda k, d=None: d
        entry.published_parsed = (2026, 4, 20, 14, 30, 0, 0, 0, 0)
        result = reader_with_fake_feeds._parse_date(entry)
        assert result == "2026-04-20T14:30:00+00:00"

    def test_parse_date_fallback_to_now_utc(self, reader_with_fake_feeds):
        entry = MagicMock()
        entry.get.side_effect = lambda k, d=None: d
        entry.published_parsed = None

        result = reader_with_fake_feeds._parse_date(entry)
        dt = datetime.fromisoformat(result)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timezone.utc.utcoffset(dt)
