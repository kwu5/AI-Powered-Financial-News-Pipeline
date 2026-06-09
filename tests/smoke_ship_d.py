"""Ship D smoke test — grounded cited Q&A against the REAL chunk index.

Prerequisites:
    The ChromaDB collection must already be populated (run the pipeline / Ship C
    smoke at least once). This test embeds queries and calls OpenAI — it costs a
    couple of real model calls.

Run:
    python -m tests.smoke_ship_d

What it checks:
  1. IN-DOMAIN — a finance query returns a non-empty answer, answered_from_context
     is True, has >=1 citation, and every citation url matches a url that was
     actually retrieved (proves citations come from real hits, not the model).
  2. OUT-OF-DOMAIN — an obviously off-topic query reports answered_from_context
     False instead of fabricating an answer.

Reuses the pipeline's shared embedder / vstore / llm singletons so the embedding
model loads once.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.pipeline import embedder, vstore, llm, settings
from src.rag.retriever import Retriever
from src.rag.qa import QAEngine

IN_DOMAIN = "What has the Federal Reserve signaled about interest rates?"
OUT_OF_DOMAIN = "What is the best pizza in Naples?"


def main() -> None:
    top_k = settings.RETRIEVAL_TOP_K
    retriever = Retriever(embedder, vstore)
    engine = QAEngine(retriever, llm)
    failures = []

    if vstore.financial_news_collection.count() == 0:
        print("Chroma is empty — populate the index first (run the pipeline). Aborting.")
        return

    # --- Check 1: in-domain query is grounded and citations trace to real hits ---
    print(f"\n--- IN-DOMAIN: {IN_DOMAIN!r} ---")
    hit_urls = {h["url"] for h in retriever.retrieve(IN_DOMAIN, top_k)}
    result = engine.answer_query(IN_DOMAIN, top_k)

    print(f"answered_from_context: {result.answered_from_context}")
    print(f"answer: {result.answer[:300]}")
    for c in result.citations:
        print(f"  [{c.marker}] {c.title} — {c.source} — {c.url}")

    if not result.answer.strip():
        failures.append("in-domain: empty answer")
    if not result.answered_from_context:
        failures.append("in-domain: answered_from_context is False (expected True)")
    if not result.citations:
        failures.append("in-domain: no citations (expected >=1)")
    stray = [c.url for c in result.citations if c.url not in hit_urls]
    if stray:
        failures.append(f"in-domain: citation url(s) not among retrieved hits: {stray}")

    # --- Check 2: out-of-domain query refuses to fabricate ---
    print(f"\n--- OUT-OF-DOMAIN: {OUT_OF_DOMAIN!r} ---")
    od = engine.answer_query(OUT_OF_DOMAIN, top_k)
    print(f"answered_from_context: {od.answered_from_context}")
    print(f"answer: {od.answer[:300]}")
    if od.answered_from_context:
        failures.append("out-of-domain: answered_from_context is True (expected False)")

    # --- Result ---
    print("\n=== RESULT ===")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASS — grounded answers cite real hits; out-of-domain refuses to fabricate.")


if __name__ == "__main__":
    main()
