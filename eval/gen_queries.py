"""eval/gen_queries.py — Ship E, step 1: LLM-assisted query candidate generation.

Samples articles already indexed in the DB and asks the LLM to propose finance
questions that are answerable FROM each article's own text, writing them to
`eval/queries_candidates.jsonl` for human curation.

This produces a *candidate* file, NOT the final test set. The workflow is:
    1. Run this script  -> eval/queries_candidates.jsonl   (llm-sourced)
    2. YOU curate it    -> drop dupes/bad questions, keep the good ones
    3. YOU hand-add ~10 hard / out-of-domain queries        (source="hand")
    4. eval/label_testset.py pools + labels relevance       -> eval/testset.jsonl

Design rule (from doc/ship-e-testset.md): the model proposes *queries* only.
Relevance is decided by you in the labeler, never by the model here. Keep the
out-of-domain set hand-written — don't ask the model to invent unanswerable
questions.

Run:  python -m eval.gen_queries [--num-articles N] [--per-article K] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from pydantic import BaseModel

from src.config import Settings
from src.storage.database import Database, Article
from src.summarization.llm_client import LLMClient


# --- Output location & defaults -------------------------------------------------

OUTPUT_PATH = "eval/queries_candidates.jsonl"
DEFAULT_NUM_ARTICLES = 60        # seed articles to sample; ~1 good query each after curation
DEFAULT_QUESTIONS_PER_ARTICLE = 1  # ask for a few, keep the best; 1-2 is plenty

# Cap how much article text we send per generation call — keeps token cost bounded
# and the lead of a financial article carries the askable facts anyway.
_CONTENT_CHARS = 2000


# --- Structured-output schema for the generation call ---------------------------

class GeneratedQueries(BaseModel):
    """Flat structured-output schema for one article's generated questions.

    Kept flat (a single list of strings) for reliable gpt-4o-mini structured
    output, mirroring GroundedLLMResponse in llm_client.py. Each string is one
    self-contained finance question answerable from the seed article's text.
    """
    questions: List[str]


# --- Steps ----------------------------------------------------------------------

def sample_articles(db: Database, n: int) -> List[Article]:
    """Return up to `n` seed articles to generate questions from.

    Pull a DIVERSE sample, not the first n rows — vary across `source` and
    `published_at` so the candidate queries aren't all about one feed/day.
    Skip rows with empty or very short `content` (nothing to ask about).

    NOTE: `Database` has no random-sample accessor yet. Either add a small one
    (e.g. `get_articles_sample(n)`) to database.py, or open a session here and
    query directly. `expire_on_commit=False` means the returned Article objects
    stay usable after the session closes (see get_unindexed_articles).
    """
    # get_articles_sample() randomizes order (func.random()) and filters out
    # stub-length content, which gives the source/day diversity we want for free.
    return db.get_articles_sample(n)


def generate_questions_for_article(llm: LLMClient, article: Article, k: int) -> List[str]:
    """Ask the LLM for `k` questions answerable strictly from `article.content`.

    Prompt guidance to bake in:
      - Questions must be answerable from THIS article's text alone.
      - Make them self-contained — do NOT reference "the article"/"this story"
        (a test-set query has to stand on its own).
      - Prefer specific, factual questions (figures, who/what/when) over yes/no.
      - Return exactly `k` (or fewer) distinct questions, no numbering/prose.

    Use OpenAI structured output with `GeneratedQueries` as the response_format
    (the same `llm.client.beta.chat.completions.parse(...)` pattern used by
    LLMClient.generate_grounded_answer), temperature low for consistency. Return
    the cleaned `.questions` list; on a refusal / empty parse, return [].
    """
    system_prompt = (
        "You generate evaluation questions for a financial-news retrieval system. "
        "Given one news article, write specific, factual questions that can be "
        "answered using ONLY that article's text.\n"
        "Rules:\n"
        f"- Return at most {k} distinct question(s).\n"
        "- Each question must be self-contained: do NOT say 'the article', 'this "
        "story', 'the report', or 'according to the text'. A reader who never saw "
        "the article must understand the question.\n"
        "- Name the concrete subject (company, person, index, asset, agency) so the "
        "question stands alone, e.g. 'How much did Apple's Q2 revenue grow?'.\n"
        "- Prefer specific facts (figures, names, dates, who/what/when) over yes/no "
        "or vague questions.\n"
        "- Return only the questions, no numbering or commentary."
        "If the article is not about finance, markets, the economy, business, or financial regulation, return an empty list"
    )
    content = (article.content or "")[:_CONTENT_CHARS]
    user_prompt = (
        f"Title: {article.title}\n"
        f"Source: {article.source}\n\n"
        f"Article text:\n{content}"
    )

    try:
        completion = llm.client.beta.chat.completions.parse(
            model=llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format=GeneratedQueries,
        )
    except Exception as e:  # network / API error — skip this article, keep going
        print(f"  ! generation failed for article {article.id}: {e}")
        return []

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []

    # Clean: strip, drop empties, dedup within this article (case-insensitive), cap at k.
    cleaned: List[str] = []
    seen = set()
    for q in parsed.questions:
        q = (q or "").strip()
        key = q.lower()
        if not q or key in seen:
            continue
        seen.add(key)
        cleaned.append(q)
    return cleaned[:k]


def write_candidates(rows: List[dict], path: str = OUTPUT_PATH) -> None:
    """Append candidate rows to `path` as JSONL, one object per line.

    Each row: {"query": str, "seed_article_id": int, "source": "llm"}.
    Append (don't truncate) so the file can be built over several runs. Optionally
    skip near-duplicate `query` strings already written — exact-duplicate dropping
    here is cheap; fuzzy dedup can wait for the manual curation pass.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Load already-written queries so re-runs don't pile up exact duplicates.
    existing = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line)["query"].strip().lower())
                except (json.JSONDecodeError, KeyError):
                    continue

    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            key = row["query"].strip().lower()
            if key in existing:
                continue
            existing.add(key)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} new candidate(s) ({len(rows) - written} dupe(s) skipped) -> {path}")


def main() -> None:
    """Wire the steps: parse args -> build Settings/Database/LLMClient -> sample
    articles -> generate questions per article -> collect rows -> write_candidates.

    Log progress per article (id + how many questions kept) — this is a slow,
    API-cost-incurring loop and you'll want to watch it. Print the final candidate
    count and remind the user this file still needs hand-curation + the ~10
    hand-written hard/out-of-domain queries before labeling.
    """
    parser = argparse.ArgumentParser(description="Generate query candidates for the Ship E test set.")
    parser.add_argument("--num-articles", type=int, default=DEFAULT_NUM_ARTICLES)
    parser.add_argument("--per-article", type=int, default=DEFAULT_QUESTIONS_PER_ARTICLE)
    parser.add_argument("--out", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    db = Database()
    llm = LLMClient(settings)

    articles = sample_articles(db, args.num_articles)
    if not articles:
        print("No articles with usable content in the DB — run the ingestion pipeline first.")
        return
    print(f"Sampled {len(articles)} seed article(s); generating up to "
          f"{args.per_article} question(s) each...\n")

    rows: List[dict] = []
    for i, article in enumerate(articles, 1):
        questions = generate_questions_for_article(llm, article, args.per_article)
        print(f"[{i}/{len(articles)}] article {article.id}: kept {len(questions)} question(s)")
        for q in questions:
            rows.append({"query": q, "seed_article_id": article.id, "source": "llm"})

    write_candidates(rows, args.out)

    print(
        "\nDone. Next steps (manual):\n"
        f"  1. Curate {args.out}: delete weak/duplicate questions.\n"
        "  2. Hand-add ~10 queries: a few genuinely hard in-domain ones plus "
        "~5-10 out-of-domain (source=\"hand\").\n"
        "  3. Run eval/label_testset.py to pool + label relevance into eval/testset.jsonl."
    )


if __name__ == "__main__":
    main()
