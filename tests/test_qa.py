"""Tests for the Ship D Q&A engine — grounding + citation resolution.

The QAEngine takes a Retriever and an LLMClient as constructor args, so both are
injected as mocks here; no OpenAI patching needed. The contract under test:
  - empty retrieval short-circuits with NO LLM call;
  - the numbered context block is 1-based and text-only;
  - citations are built from the RETRIEVED hits (not the model), resolved by
    marker, with out-of-range markers dropped.
"""

from unittest.mock import MagicMock

from src.rag.qa import QAEngine, Citation, GroundedAnswer
from src.summarization.llm_client import GroundedLLMResponse


def hit(i):
    """A retriever hit dict (same shape Retriever.retrieve emits)."""
    return {
        "chunk_id": f"{i}:0",
        "article_id": i,
        "text": f"text {i}",
        "title": f"Title {i}",
        "source": f"Source {i}",
        "url": f"https://example.com/{i}",
        "published_at": "2026-06-01T00:00:00Z",
        "distance": 0.1,
    }


def make_engine(hits, llm_response=None):
    retriever = MagicMock()
    retriever.retrieve.return_value = hits
    llm = MagicMock()
    if llm_response is not None:
        llm.generate_grounded_answer.return_value = llm_response
    return QAEngine(retriever, llm), retriever, llm


class TestAnswerQuery:

    def test_empty_hits_short_circuit_no_llm_call(self):
        engine, _, llm = make_engine([])
        result = engine.answer_query("anything", top_k=5)

        assert isinstance(result, GroundedAnswer)
        assert result.answered_from_context is False
        assert result.citations == []
        llm.generate_grounded_answer.assert_not_called()

    def test_numbered_context_is_one_based_and_text_only(self):
        resp = GroundedLLMResponse(answer="A [1].", used_markers=[1], answered_from_context=True)
        engine, _, llm = make_engine([hit(1), hit(2)], resp)
        engine.answer_query("q", top_k=2)

        _query, numbered_context = llm.generate_grounded_answer.call_args[0]
        assert numbered_context == "[1] text 1\n\n[2] text 2\n\n"

    def test_citations_resolve_by_marker(self):
        resp = GroundedLLMResponse(answer="X [1] Y [2].", used_markers=[1, 2], answered_from_context=True)
        engine, _, _ = make_engine([hit(1), hit(2)], resp)
        result = engine.answer_query("q", top_k=2)

        assert result.answer == "X [1] Y [2]."
        assert result.answered_from_context is True
        assert [c.marker for c in result.citations] == [1, 2]
        # marker n maps to hits[n-1] — authoritative fields come from the hit
        c1 = result.citations[0]
        assert isinstance(c1, Citation)
        assert c1.chunk_id == "1:0"
        assert c1.article_id == 1
        assert c1.title == "Title 1"
        assert c1.source == "Source 1"
        assert c1.url == "https://example.com/1"
        assert result.citations[1].url == "https://example.com/2"

    def test_out_of_range_marker_is_dropped(self):
        # only 2 hits, but the model cited [3] — drop it, keep the valid one
        resp = GroundedLLMResponse(answer="Z [1] [3].", used_markers=[1, 3], answered_from_context=True)
        engine, _, _ = make_engine([hit(1), hit(2)], resp)
        result = engine.answer_query("q", top_k=2)

        assert [c.marker for c in result.citations] == [1]

    def test_answered_from_context_false_passthrough(self):
        # hits exist, but the model judged the context insufficient
        resp = GroundedLLMResponse(answer="Not enough info.", used_markers=[], answered_from_context=False)
        engine, _, _ = make_engine([hit(1)], resp)
        result = engine.answer_query("q", top_k=1)

        assert result.answered_from_context is False
        assert result.citations == []
