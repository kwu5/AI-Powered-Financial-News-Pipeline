"""eval/label_testset.py — Ship E, step 2: interactive relevance labeling.

Turns curated queries into the hand-verified ground truth `eval/testset.jsonl`.
For each in-domain query it pools candidate articles via the real retriever (at a
deliberately deep pool, larger than RETRIEVAL_TOP_K, so labels aren't biased
toward the top-k the system actually serves), dedups chunks down to articles, and
asks YOU y/n on each. The model never decides relevance — that's the whole point.

Workflow position:
    eval/gen_queries.py -> eval/queries_candidates.jsonl
    YOU curate + hand-add hard / out-of-domain queries -> eval/queries_curated.jsonl
    THIS script: pool + label                          -> eval/testset.jsonl   <-- ground truth
    eval/validate_testset.py: check it before declaring Ship E done

Input file (`--in`, default eval/queries_curated.jsonl), one JSON object per line:
    {"query": "...", "source": "llm"|"hand", "type": "in_domain"|"out_of_domain",
     "notes": ""}
`source` defaults to "llm", `type` to "in_domain", `notes` to "" when absent, so a
lightly-edited candidates file works as-is. Out-of-domain rows are written with an
empty relevant set and never pooled (by definition nothing is relevant).

Resumable: queries whose text already appears in eval/testset.jsonl are skipped, and
each labeled query is appended immediately — kill it any time and rerun to continue.

Run:  python -m eval.label_testset [--in PATH] [--out PATH] [--pool-depth N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Tuple

from src.config import Settings
from src.processing.embeddings import EmbeddingGenerator
from src.storage.database import Database
from src.storage.vector_store import VectorStore
from src.rag.retriever import Retriever


# Pool depth is EVAL TOOLING, not runtime config — it stays a local constant here.
# Only RETRIEVAL_TOP_K (the depth the system actually serves) lives in Settings.
# Deep pool (>> top_k) so the ground truth isn't biased toward the served top-k.
POOL_DEPTH = 25

IN_PATH = "eval/queries_curated.jsonl"
OUT_PATH = "eval/testset.jsonl"
SNIPPET_CHARS = 280


# --- I/O helpers ----------------------------------------------------------------

def load_queries(path: str) -> List[dict]:
    """Read curated queries, filling defaults for source/type/notes."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No curated query file at {path}. Create it by curating "
            "eval/queries_candidates.jsonl and hand-adding your out-of-domain queries."
        )
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}") from e
            queries.append({
                "query": obj["query"].strip(),
                "source": obj.get("source", "llm"),
                "type": obj.get("type", "in_domain"),
                "notes": obj.get("notes", ""),
            })
    return queries


def load_progress(path: str) -> Tuple[set, int]:
    """Return (set of already-labeled query texts, next q-number) from the output.

    Resumability keys on the query TEXT, so reordering/curating the input file
    doesn't re-ask labeled queries. The next id continues past the highest q### seen.
    """
    seen: set = set()
    max_n = 0
    if not os.path.exists(path):
        return seen, 1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seen.add(obj["query"].strip().lower())
            m = re.match(r"q(\d+)$", obj.get("query_id", ""))
            if m:
                max_n = max(max_n, int(m.group(1)))
    return seen, max_n + 1


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Candidate building ---------------------------------------------------------

def pool_candidates(retriever: Retriever, query: str, pool_depth: int) -> List[dict]:
    """Retrieve a deep pool and dedup chunks -> unique articles (best hit kept).

    Multiple chunks from one article must not prompt the same y/n twice, so we
    keep the first (lowest-distance) hit per article_id and preserve that order.
    """
    hits = retriever.retrieve(query, pool_depth)
    by_article: dict = {}
    for h in hits:
        aid = h["article_id"]
        if aid not in by_article:
            by_article[aid] = {
                "article_id": aid,
                "title": h["title"],
                "source": h["source"],
                "text": h["text"],
            }
    return list(by_article.values())


# --- Interactive labeling -------------------------------------------------------

def _prompt(query: str, cand: dict, idx: int, total: int) -> str:
    """Show one candidate; return 'y' / 'n' / 's' / 'q'. Re-asks on bad input."""
    snippet = (cand["text"] or "").strip().replace("\n", " ")[:SNIPPET_CHARS]
    print("\n" + "-" * 72)
    print(f"Q: {query}")
    print(f"  candidate {idx}/{total}  [article {cand['article_id']}]")
    print(f"  {cand['title']}  ({cand['source']})")
    print(f"  {snippet}...")
    while True:
        ans = input("  relevant? [y]es / [n]o / [s]kip / [q]uit+save: ").strip().lower()
        if ans in ("y", "n", "s", "q"):
            return ans
        print("  please enter y, n, s, or q")


def label_candidates(query: str, candidates: List[dict]) -> Tuple[List[int], bool]:
    """Walk candidates; return (relevant_article_ids, quit_flag).

    quit_flag True means the user hit 'q' — caller should stop the whole session.
    """
    relevant: List[int] = []
    for i, cand in enumerate(candidates, 1):
        ans = _prompt(query, cand, i, len(candidates))
        if ans == "q":
            return relevant, True
        if ans == "y":
            relevant.append(cand["article_id"])
    return relevant, False


def spot_check(db: Database, query: str, exclude_ids: set) -> Tuple[List[int], bool]:
    """Optional pass: label a few random non-pooled articles to catch pool misses.

    Returns (extra_relevant_ids, quit_flag). Enter to skip keeps the flow fast.
    """
    raw = input("\n  spot-check N random non-pooled articles? [number / Enter to skip]: ").strip()
    if not raw:
        return [], False
    try:
        n = int(raw)
    except ValueError:
        print("  not a number — skipping spot-check")
        return [], False
    if n <= 0:
        return [], False

    # Over-sample then filter out anything already in the pool, so we still get ~n fresh ones.
    sample = db.get_articles_sample(n + len(exclude_ids))
    fresh = [
        {"article_id": a.id, "title": a.title, "source": a.source, "text": a.content}
        for a in sample if a.id not in exclude_ids
    ][:n]
    if not fresh:
        print("  no non-pooled articles available to spot-check")
        return [], False
    return label_candidates(query, fresh)


# --- Main -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively label the Ship E test set.")
    parser.add_argument("--in", dest="in_path", type=str, default=IN_PATH)
    parser.add_argument("--out", dest="out_path", type=str, default=OUT_PATH)
    parser.add_argument("--pool-depth", type=int, default=POOL_DEPTH)
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    db = Database()
    retriever = Retriever(EmbeddingGenerator(settings), VectorStore(settings))

    queries = load_queries(args.in_path)
    seen, next_id = load_progress(args.out_path)
    todo = [q for q in queries if q["query"].lower() not in seen]

    print(f"{len(queries)} curated quer(ies); {len(seen)} already labeled; "
          f"{len(todo)} to go. Pool depth = {args.pool_depth}.")
    if not todo:
        print("Nothing left to label. Run eval/validate_testset.py to check the set.")
        return

    for q in todo:
        query, qtype = q["query"], q["type"]
        query_id = f"q{next_id:03d}"
        print("\n" + "=" * 72)
        print(f"{query_id}  ({qtype}, source={q['source']})")

        if qtype == "out_of_domain":
            # By definition nothing in the corpus answers it — don't pool, just record [].
            relevant: List[int] = []
            print("  out-of-domain -> relevant_article_ids = []  (tests the abstention path)")
        else:
            candidates = pool_candidates(retriever, query, args.pool_depth)
            if not candidates:
                print("  pool is empty (retriever returned nothing). Skipping for now — "
                      "is the corpus indexed?")
                continue
            relevant, quit_flag = label_candidates(query, candidates)
            if quit_flag:
                print("\nQuit — progress saved. Rerun to continue where you left off.")
                return
            extra, quit_flag = spot_check(db, query, exclude_ids={c["article_id"] for c in candidates})
            relevant.extend(extra)
            if quit_flag:
                # Still record what we labeled for THIS query before exiting.
                _write(args, query_id, query, sorted(set(relevant)), q)
                print("\nQuit — progress saved. Rerun to continue where you left off.")
                return

        _write(args, query_id, query, sorted(set(relevant)), q)
        next_id += 1

    print("\nAll curated queries labeled. Run eval/validate_testset.py before declaring Ship E done.")


def _write(args, query_id: str, query: str, relevant_ids: List[int], q: dict) -> None:
    row = {
        "query_id": query_id,
        "query": query,
        "relevant_article_ids": relevant_ids,
        "source": q["source"],
        "type": q["type"],
        "notes": q["notes"],
    }
    append_row(args.out_path, row)
    print(f"  saved {query_id}: {len(relevant_ids)} relevant -> {args.out_path}")


if __name__ == "__main__":
    main()
