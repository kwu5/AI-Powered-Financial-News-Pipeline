"""Grounded, source-cited Q&A over the chunk index (Ship D).

answer_query() is the seam Ship F will evaluate: retrieve top-k chunks, generate
an answer constrained to that context, and resolve inline [n] markers back to the
real source chunks. Citations are built from the RETRIEVED hits, never from the
LLM's output — the model picks which source numbers to cite, we map those numbers
to the authoritative chunk_id / article_id / url.
"""

import logging
from typing import List

from pydantic import BaseModel

from src.rag.retriever import Retriever
from src.summarization.llm_client import LLMClient

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    marker: int          # the [1], [2]... number used inline in `answer`
    chunk_id: str        # "{article_id}:{chunk_index}"
    article_id: int
    title: str           # article headline — display metadata from the hit
    source: str          # feed/source name — display metadata from the hit
    url: str


class GroundedAnswer(BaseModel):
    answer: str                   # inline [n] markers reference `citations`
    citations: List[Citation]
    answered_from_context: bool   # False -> retrieved context was insufficient


class QAEngine:
    def __init__(self, retriever: Retriever, llm: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer_query(self, query: str, top_k: int) -> GroundedAnswer:
        """Retrieve -> ground -> cite. Returns a GroundedAnswer.
        """
        
        hits = self.retriever.retrieve(query, top_k)
        if not hits:
            return GroundedAnswer(
                answer = "I don't have enough indexed context to answer that.",
                citations = [],
                answered_from_context = False,
            )

        # Numbered context block — 1-based, just the chunk text. The numbers here
        # are what the model cites with [n] and reports back in resp.used_markers.
        numbered_context = ""
        for i, hit in enumerate(hits, start=1):
            numbered_context += f"[{i}] {hit['text']}\n\n"

        resp = self.llm.generate_grounded_answer(query, numbered_context)

        # Build citations from `hits`, NOT from the model. The model only tells us
        # WHICH source numbers it used; we map each back to the authoritative hit.
        citations = []
        for n in resp.used_markers:
            if n < 1 or n > len(hits):
                logger.warning("Model cited out-of-range source [%d]; dropping", n)
                continue
            hit = hits[n - 1]
            citations.append(
                Citation(
                    marker=n,
                    chunk_id=hit["chunk_id"],
                    article_id=hit["article_id"],
                    title=hit["title"],
                    source=hit["source"],
                    url=hit["url"],
                )
            )

        return GroundedAnswer(
            answer=resp.answer,
            citations=citations,
            answered_from_context=resp.answered_from_context,
        )
