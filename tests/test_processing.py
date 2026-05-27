"""Tests for the processing layer: TextCleaner, Deduplicator, url_canon, content_hash."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.processing.cleaner import TextCleaner
from src.processing.content_hash import _normalize_text, compute_content_hash
from src.processing.deduplicator import Deduplicator
from src.processing.url_canon import TRACKING_PARAMS, canonicalize_url


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

    def test_stage_3_removes_similar(self, mock_settings):
        """Stage 3 (embedding similarity) catches near-duplicates that pass Stages 1+2."""
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
            {"title": "Fed Raises Rates", "description": "Rate hike announced.",
             "canonical_url": "https://a.com/1", "content_hash": "h1"},
            {"title": "Fed Hikes Rates", "description": "Rate increase confirmed.",
             "canonical_url": "https://b.com/2", "content_hash": "h2"},
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
            {"title": "Fed Raises Rates", "description": "Rate hike.",
             "canonical_url": "https://a.com/1", "content_hash": "h1"},
            {"title": "Apple Earnings", "description": "Record revenue.",
             "canonical_url": "https://b.com/2", "content_hash": "h2"},
        ]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 2

    def test_single_article(self, mock_settings):
        gen = MagicMock()
        gen.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])

        dedup = self._make_deduplicator(gen, mock_settings)
        articles = [{"title": "Solo Article", "description": "Only one.",
                     "canonical_url": "https://a.com/1", "content_hash": "h1"}]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 1

    def test_empty_list(self, mock_settings):
        gen = MagicMock()
        dedup = self._make_deduplicator(gen, mock_settings)
        result = dedup.deduplicate_articles([])
        assert result == []

    def test_stage_1_drops_same_canonical_url(self, mock_settings):
        """Stage 1: same canonical_url -> drop, keep the first occurrence."""
        gen = MagicMock()
        gen.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])
        dedup = self._make_deduplicator(gen, mock_settings)
        articles = [
            {"title": "First", "description": "x",
             "canonical_url": "https://a.com/x", "content_hash": "h1"},
            {"title": "Second", "description": "y",
             "canonical_url": "https://a.com/x", "content_hash": "h2"},
        ]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 1
        assert result[0]["title"] == "First"

    def test_stage_2_drops_same_content_hash(self, mock_settings):
        """Stage 2: same content_hash but different canonical_url -> drop."""
        gen = MagicMock()
        gen.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])
        dedup = self._make_deduplicator(gen, mock_settings)
        articles = [
            {"title": "First", "description": "x",
             "canonical_url": "https://a.com/1", "content_hash": "same"},
            {"title": "Second", "description": "y",
             "canonical_url": "https://b.com/2", "content_hash": "same"},
        ]
        result = dedup.deduplicate_articles(articles)
        assert len(result) == 1
        assert result[0]["title"] == "First"


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalizeUrl:

    def test_strips_utm_params(self):
        a = canonicalize_url("https://example.com/article?utm_source=twitter&id=42")
        b = canonicalize_url("https://example.com/article?id=42")
        assert a == b

    def test_lowercases_scheme_and_host_but_not_path(self):
        result = canonicalize_url("HTTPS://EXAMPLE.COM/Article/Foo")
        assert result.startswith("https://example.com")
        assert "/Article/Foo" in result  # path case preserved

    def test_strips_default_https_port(self):
        assert canonicalize_url("https://example.com:443/foo") == "https://example.com/foo"

    def test_strips_default_http_port(self):
        assert canonicalize_url("http://example.com:80/foo") == "http://example.com/foo"

    def test_keeps_non_default_port(self):
        assert ":8080" in canonicalize_url("https://example.com:8080/foo")

    def test_drops_fragment(self):
        assert "#" not in canonicalize_url("https://example.com/foo#section-2")

    def test_strips_every_known_tracking_param(self):
        query = "&".join(f"{p}=x" for p in TRACKING_PARAMS) + "&id=42"
        result = canonicalize_url(f"https://example.com/foo?{query}")
        assert result == "https://example.com/foo?id=42"

    def test_tracking_param_match_is_case_insensitive(self):
        result = canonicalize_url("https://example.com/foo?UTM_Source=x&id=42")
        assert "utm_source" not in result.lower()
        assert "UTM_Source" not in result
        assert "id=42" in result

    def test_strips_trailing_slash_from_non_root(self):
        assert canonicalize_url("https://example.com/foo/") == "https://example.com/foo"

    def test_preserves_root_slash(self):
        assert canonicalize_url("https://example.com/") == "https://example.com/"

    def test_root_when_path_missing(self):
        # urlparse gives "" for the path here; canonical form should normalize to "/".
        assert canonicalize_url("https://example.com") == "https://example.com/"

    def test_collapses_double_slashes(self):
        assert canonicalize_url("https://example.com//foo//bar") == "https://example.com/foo/bar"

    def test_empty_url_raises(self):
        with pytest.raises(ValueError):
            canonicalize_url("")

    def test_unparseable_url_raises(self):
        with pytest.raises(ValueError):
            canonicalize_url("not a url")

    def test_three_real_world_variants_collapse_to_one(self):
        """The smoke fixtures in __main__ — all three should canonicalize identically."""
        a = canonicalize_url("https://www.example.com/article?utm_source=twitter&id=42")
        b = canonicalize_url("HTTPS://WWW.Example.com:443/article/?id=42#section-2")
        c = canonicalize_url("https://www.example.com/article?id=42")
        assert a == b == c


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

class TestNormalizeText:

    def test_lowercases(self):
        assert _normalize_text("HELLO World") == "hello world"

    def test_collapses_whitespace_runs(self):
        assert _normalize_text("hello   world\t\nfoo") == "hello world foo"

    def test_strips_leading_and_trailing(self):
        assert _normalize_text("  hello  ") == "hello"

    def test_punctuation_becomes_space_preserving_word_boundary(self):
        # Hyphen splits the word — must NOT concatenate to "fedraises".
        assert _normalize_text("fed-raises-rates") == "fed raises rates"

    def test_drops_trailing_punctuation(self):
        assert _normalize_text("Fed raises rates!!") == "fed raises rates"

    def test_preserves_digits(self):
        assert _normalize_text("Bitcoin hits $100k") == "bitcoin hits 100k"

    def test_only_punctuation_normalizes_to_empty(self):
        assert _normalize_text("!!! ??? ...") == ""


class TestComputeContentHash:

    def test_returns_64_char_lowercase_hex(self):
        h = compute_content_hash({"title": "Hello", "content": "World"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        a = {"title": "Fed Raises Rates", "content": "The Federal Reserve raised rates."}
        assert compute_content_hash(a) == compute_content_hash(a)

    def test_case_punctuation_whitespace_equivalence(self):
        """The smoke fixtures in __main__ — both must hash identically."""
        a = {"title": "Fed Raises Rates", "content": "The Federal Reserve raised rates."}
        b = {"title": "  fed raises rates  ", "content": "The Federal Reserve raised rates!!"}
        assert compute_content_hash(a) == compute_content_hash(b)

    def test_different_content_differs(self):
        a = {"title": "Fed Raises Rates", "content": "Foo"}
        b = {"title": "Fed Raises Rates", "content": "Bar"}
        assert compute_content_hash(a) != compute_content_hash(b)

    def test_different_title_differs(self):
        a = {"title": "Fed Raises", "content": "Same content here."}
        b = {"title": "Fed Hikes", "content": "Same content here."}
        assert compute_content_hash(a) != compute_content_hash(b)

    def test_title_content_swap_differs(self):
        """Hash must depend on FIELD identity, not just the multiset of strings."""
        a = {"title": "alpha", "content": "beta"}
        b = {"title": "beta", "content": "alpha"}
        assert compute_content_hash(a) != compute_content_hash(b)

    def test_missing_title_raises_keyerror(self):
        with pytest.raises(KeyError):
            compute_content_hash({"content": "x"})

    def test_missing_content_raises_keyerror(self):
        with pytest.raises(KeyError):
            compute_content_hash({"title": "x"})

    def test_empty_title_after_normalization_raises(self):
        with pytest.raises(ValueError):
            compute_content_hash({"title": "!!!", "content": "valid content"})

    def test_empty_content_after_normalization_raises(self):
        with pytest.raises(ValueError):
            compute_content_hash({"title": "valid title", "content": "..."})
