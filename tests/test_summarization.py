"""Tests for the summarization layer: LLMClient and ReportGenerator."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.summarization.llm_client import LLMClient
from src.summarization.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path):
    s = MagicMock()
    s.OPENAI_API_KEY = "test-key"
    s.LLM_MODEL = "gpt-4o-mini"
    s.OUTPUT_DIR = str(tmp_path / "output")
    return s


@pytest.fixture
def sample_articles():
    return [
        {
            "title": "Fed Raises Rates",
            "source": "Reuters",
            "content": "The Federal Reserve raised interest rates by 25 basis points.",
        },
        {
            "title": "Apple Earnings",
            "source": "CNBC",
            "content": "Apple reported record quarterly revenue of $95 billion.",
        },
    ]


@pytest.fixture
def sample_markdown():
    return """## Major Market Movements
- S&P 500 closed up 1.2%.

## Corporate Earnings & News
- Apple reported record revenue.
"""


@pytest.fixture
def mock_openai_response():
    choice = MagicMock()
    choice.message.content = "## Major Market Movements\n- Fed raised rates.\n\n## Market Sentiment\nBullish."
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class TestLLMClient:

    @patch("src.summarization.llm_client.OpenAI")
    def test_generate_summary_returns_content(self, mock_openai_cls, mock_settings, sample_articles, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        llm = LLMClient(mock_settings)
        result = llm.generate_summary(sample_articles)

        assert result == "## Major Market Movements\n- Fed raised rates.\n\n## Market Sentiment\nBullish."

    @patch("src.summarization.llm_client.OpenAI")
    def test_generate_summary_prompt_format(self, mock_openai_cls, mock_settings, sample_articles, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        llm = LLMClient(mock_settings)
        llm.generate_summary(sample_articles)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_msg = messages[1]["content"]
        assert "Article 1" in user_msg
        assert "Article 2" in user_msg
        assert "Fed Raises Rates" in user_msg

    @patch("src.summarization.llm_client.OpenAI")
    def test_classify_sentiment(self, mock_openai_cls, mock_settings):
        choice = MagicMock()
        choice.message.content = "POSITIVE"
        resp = MagicMock()
        resp.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_openai_cls.return_value = mock_client

        llm = LLMClient(mock_settings)
        result = llm.classify_sentiment("Apple stock surged 10%.")

        assert result == "POSITIVE"

    @patch("src.summarization.llm_client.OpenAI")
    def test_classify_sentiment_strips_whitespace(self, mock_openai_cls, mock_settings):
        choice = MagicMock()
        choice.message.content = "  negative \n"
        resp = MagicMock()
        resp.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_openai_cls.return_value = mock_client

        llm = LLMClient(mock_settings)
        result = llm.classify_sentiment("Stock prices fell sharply.")

        assert result == "NEGATIVE"


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:

    def test_save_markdown_creates_file(self, mock_settings, sample_markdown):
        rg = ReportGenerator(mock_settings)
        date = datetime(2026, 4, 20)
        filepath = rg.save_markdown(sample_markdown, date)

        assert Path(filepath).exists()
        assert "financial_briefing_2026-04-20.md" in filepath

    def test_save_markdown_content(self, mock_settings, sample_markdown):
        rg = ReportGenerator(mock_settings)
        date = datetime(2026, 4, 20)
        filepath = rg.save_markdown(sample_markdown, date)

        content = Path(filepath).read_text(encoding="utf-8")
        assert "S&P 500 closed up 1.2%" in content

    @patch("src.summarization.report_generator.Environment")
    def test_generate_html_creates_file(self, mock_jinja_env_cls, mock_settings, sample_markdown):
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "<html><body>Report</body></html>"
        mock_env.get_template.return_value = mock_template
        mock_jinja_env_cls.return_value = mock_env

        rg = ReportGenerator(mock_settings)
        rg.jinja_env = mock_env
        date = datetime(2026, 4, 20)
        filepath = rg.generate_html(sample_markdown, date)

        assert Path(filepath).exists()
        assert "financial_briefing_2026-04-20.html" in filepath

    def test_markdown_to_html_headers(self, mock_settings):
        rg = ReportGenerator(mock_settings)
        result = rg._markdown_to_html("## Market Movements")
        assert "<h2>Market Movements</h2>" in result

    def test_markdown_to_html_bullets(self, mock_settings):
        rg = ReportGenerator(mock_settings)
        result = rg._markdown_to_html("- Fed raised rates.\n- Apple beat earnings.")
        assert "<li>Fed raised rates.</li>" in result
        assert "<li>Apple beat earnings.</li>" in result
        assert "<ul>" in result
        assert "</ul>" in result

    def test_markdown_to_html_paragraphs(self, mock_settings):
        rg = ReportGenerator(mock_settings)
        result = rg._markdown_to_html("Overall bullish sentiment.")
        assert "<p>Overall bullish sentiment.</p>" in result
