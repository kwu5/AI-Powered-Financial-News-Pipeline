"""THROWAWAY helper (Ship E accelerated-review assist) — NOT part of the pipeline.

Non-interactive sibling of label_testset.py: pools candidates for every
still-unlabeled in-domain curated query and dumps them to JSON so the labeling
review sheet can be built without driving the interactive y/n loop. Relevance is
still decided by the human — this only gathers the pool. Delete after use.

Run:  python -m eval._pool_dump
"""

from __future__ import annotations

import json
import os

from src.config import Settings
from src.processing.embeddings import EmbeddingGenerator
from src.storage.database import Database
from src.storage.vector_store import VectorStore
from src.rag.retriever import Retriever

from eval.label_testset import load_queries, load_progress, pool_candidates, IN_PATH, OUT_PATH

POOL_DEPTH = 25
SNIPPET_CHARS = 500
DUMP_PATH = "eval/_pool_dump.json"


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    retriever = Retriever(EmbeddingGenerator(settings), VectorStore(settings))

    queries = load_queries(IN_PATH)
    seen, next_id = load_progress(OUT_PATH)
    todo = [q for q in queries if q["query"].lower() not in seen]

    out = {"next_id": next_id, "pool_depth": POOL_DEPTH, "queries": []}
    for q in todo:
        entry = {
            "query": q["query"],
            "source": q["source"],
            "type": q["type"],
            "notes": q["notes"],
            "candidates": [],
        }
        if q["type"] != "out_of_domain":
            for c in pool_candidates(retriever, q["query"], POOL_DEPTH):
                snippet = (c["text"] or "").strip().replace("\n", " ")[:SNIPPET_CHARS]
                entry["candidates"].append({
                    "article_id": c["article_id"],
                    "title": c["title"],
                    "source": c["source"],
                    "snippet": snippet,
                })
        out["queries"].append(entry)

    with open(DUMP_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    n_in = sum(1 for q in todo if q["type"] != "out_of_domain")
    n_out = len(todo) - n_in
    print(f"dumped {len(todo)} queries ({n_in} in-domain pooled, {n_out} out-of-domain) "
          f"-> {DUMP_PATH}; next_id={next_id}")


if __name__ == "__main__":
    main()
