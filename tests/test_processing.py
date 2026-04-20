"""Tests for the processing layer: TextCleaner, Deduplicator."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.processing.cleaner import TextCleaner
from src.processing.deduplicator import Deduplicator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cleaner():
    return TextCleaner()


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.SIMILARITY_THRESHOLD = 0.85
    return s


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TestTextCleaner:

    def test_removes_urls(self, cleaner):
        text = "Check https://example.com and http://test.org/page for details."
        result = cleaner.clean_article(text)
        assert "https://example.com" not in result
        assert "http://test.org/page" not in result
        assert "Check" in result

    def test_removes_emails(self, cleaner):
        text = "Contact us at info@example.com for more info."
        result = cleaner.clean_article(text)
        assert "info@example.com" not in result
        assert "Contact us at" in result

    def test_removes_ad_phrases(self, cleaner):
        text = "Great earnings report. Subscribe Now for more updates. Click here to learn more."
        result = cleaner.clean_article(text)
        assert "subscribe now" not in result.lower()
        assert "click here" not in result.lower()
        assert "Great earnings report" in result

    def test_normalizes_whitespace(self, cleaner):
        text = "Too   many    spaces   and\nnewlines\there."
        result = cleaner.clean_article(text)
        assert "  " not in result
        assert "\n" not in result

    def test_clean_text_unchanged(self, cleaner):
        text = "Apple reported record quarterly revenue of 95 billion dollars."
        result = cleaner.clean_article(text)
        assert result == text

    def test_extract_entities(self, cleaner):
        text = "Apple Inc. CEO Tim Cook reported earnings of $1.2 billion, up 15%."
        entities = cleaner.extract_entities(text)
        assert "ORG" in entities
        assert "PERSON" in entities
        assert "MONEY" in entities
        assert "PERCENT" in entities
        assert any("Apple" in e for e in entities["ORG"])

    def test_extract_entities_empty(self, cleaner):
        text = "Nothing interesting happened today."
        entities = cleaner.extract_entities(text)
        for key in ("ORG", "PERSON", "MONEY", "PERCENT"):
            assert isinstance(entities[key], list)


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class TestDeduplicator:

    def _make_deduplicator(self, embedding_gen, settings):
        with patch("src.processing.deduplicator.Settings", return_value=settings):
            return Deduplicator(embedding_gen)

    def test_cosine_identical(self):
        gen = MagicMock()
        s = MagicMock()
        s.SIMILARITY_THRESHOLD = 0.85
        dedup = self._make_deduplicator(gen, s)

        a = np.array([0.6, 0.8, 0.0])
        assert dedup.cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        gen = MagicMock()
        s = MagicMock()
        s.SIMILARITY_THRESHOLD = 0.85
        dedup = self._make_deduplicator(gen, s)

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert dedup.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_removes_duplicates(self, mock_settings):
        gen = MagicMock()
        call_count = [0]
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.99, 0.1, 0.0]),  # very similar to first
        ]
        def side_effect(text):
            idx = min(call_count[0], len(vectors) - 1)
            call_count[0] += 1
            return vectors[idx]
        gen.generate_embedding.side_effect = side_effect

        dedup = self._make_deduplicator(gen, mock_settings)
        articles = [
            {"title": "Fed Raises Rates", "description": "Rate hike announced."},
            {"title": "Fed Hikes Rates", "description": "Rate increase confirmed."},
        ]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 1

    def test_keeps_distinct(self, mock_settings):
        gen = MagicMock()
        call_count = [0]
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),  # orthogonal — completely different
        ]
        def side_effect(text):
            idx = min(call_count[0], len(vectors) - 1)
            call_count[0] += 1
            return vectors[idx]
        gen.generate_embedding.side_effect = side_effect

        dedup = self._make_deduplicator(gen, mock_settings)
        articles = [
            {"title": "Fed Raises Rates", "description": "Rate hike."},
            {"title": "Apple Earnings", "description": "Record revenue."},
        ]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 2

    def test_single_article(self, mock_settings):
        gen = MagicMock()
        gen.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])

        dedup = self._make_deduplicator(gen, mock_settings)
        articles = [{"title": "Solo Article", "description": "Only one."}]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 1

    def test_empty_list(self, mock_settings):
        gen = MagicMock()
        dedup = self._make_deduplicator(gen, mock_settings)
        result = dedup.deduplicate_articles([])
        assert result == []
