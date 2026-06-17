# Ship E — Labeled test set (50–100 query / relevant-article pairs)

**Status:** Done (2026-06-17) — `eval/testset.jsonl` labeled (q001–q093) and passes
`validate_testset.py`. Labeling was split under time pressure: assistant labeled
q005–q049 from drafted suggestions, user hand-verified q050–q093. The
assistant-labeled half + ~7 pooling misses (seed not surfaced by `retrieve()`) are
flagged for audit in Ship I.
**Parent plan:** `IMPLEMENTATION_PLAN.md` (roadmap ship E)
**Predecessors:** Ship C (chunk-level ChromaDB) ✅, Ship D (`answer_query()` / retriever seam) ✅
**Schedule:** Week 2 · Jun 8–14 (`doc/june-weekly-schedule.md`). Flagged as the
**schedule wildcard** — this is manual labeling, not coding, so it gets a full
week to itself and the weekend as buffer.

## Goal

Produce `eval/testset.jsonl`: 50–100 finance questions, each labeled with the set
of **article ids** in the corpus that are relevant to it. This is the ground truth
Ships F (retrieval precision/recall) and G (faithfulness / answer-relevance) score
against. Relevance is labeled at the **article** level — a retrieved chunk counts
as relevant if its `article_id` is in the query's `relevant_article_ids`.

## Why now

Ship D made `answer_query()` real but unmeasured. Every metric in F–H scores that
seam against a fixed set of known-correct answers; without a hand-verified test
set there is nothing to score. The test set is the foundation of the whole eval
half — and the headline differentiator is the eval, so the labels have to be
trustworthy. Hand-verification is the point, not a nicety.

## Decisions (locked 2026-06-09)

| Decision | Choice | Rationale |
|---|---|---|
| Query sourcing | **Hybrid** — mostly LLM-generated from indexed articles, plus a handful of hand-written hard / out-of-domain queries | DB-sourced queries are fast and guaranteed answerable; hand-written ones stress retrieval and add an honest out-of-domain set |
| Relevance gathering | **Pooling + spot-check** — pool candidates via `retrieve()` at a high pool-depth, label them, then manually scan a few extra articles per query | Cuts manual searching to a fraction while a light spot-check catches obvious pool misses |
| Labeling tooling | **Small CLI helper** — shows query + candidate (title/source/snippet), `y`/`n`/`s`kip, writes JSONL | Removes JSON hand-editing friction and malformed-line risk over 50–100 rows |
| Label granularity | **Article level** (per master plan) | A chunk is relevant iff its `article_id` is labeled relevant; keeps labels stable across chunk-size changes in Ship H |
| Pool depth | High (≈20–30), **larger than `RETRIEVAL_TOP_K`** | Labeling only the top-5 the system uses would bias the ground truth toward the system being evaluated |

## Test-set schema (`eval/testset.jsonl`, one JSON object per line)

```jsonc
{
  "query_id": "q001",
  "query": "What did the Fed signal about rate cuts?",
  "relevant_article_ids": [12, 45, 88],   // [] for out-of-domain queries
  "source": "llm" | "hand",               // provenance, for later analysis
  "type": "in_domain" | "out_of_domain",  // out_of_domain => relevant set is []
  "notes": ""                              // optional labeler note
}
```

- **Out-of-domain queries** carry `relevant_article_ids: []` and `type:
  "out_of_domain"`. They test the `answered_from_context=False` abstention path
  (Ship D's short-circuit) and let Ship F report "correctly retrieved nothing
  relevant" separately from in-domain precision/recall — keep them a clearly
  flagged minority (≈5–10 of the set), not mixed silently into the P/R averages.
- `relevant_article_ids` are SQLite `articles.id` integers — the same id that
  appears in ChromaDB chunk metadata as `article_id`, so Ship F can map a
  retrieved chunk straight to a label with no extra join.

## Tasks

> Code is built by the user (build-to-learn). These are the build targets; this
> doc is the spec, not an instruction to generate the code.

- [ ] **Query candidate generator** (`eval/gen_queries.py`) — sample N articles
      from the DB, ask the LLM (reuse `LLMClient`) to produce 1–2 questions each
      that are answerable *from that article's text*, dump to
      `eval/queries_candidates.jsonl` (`{query, seed_article_id, source:"llm"}`).
      This is a *candidate* file for human curation — not the final set. Dedup /
      drop near-duplicate questions. Then hand-add ~10 queries by hand: a few
      genuinely hard in-domain ones and ~5–10 out-of-domain (`source:"hand"`).
- [ ] **Labeling CLI helper** (`eval/label_testset.py`) — for each curated query:
      1. `hits = retrieve(query, top_k=POOL_DEPTH)` (POOL_DEPTH ≈ 20–30, a local
         constant in this script — **not** a `Settings` field; it's eval tooling).
      2. **Dedup hits to unique `article_id`s** before prompting — multiple chunks
         from one article must not ask the same y/n twice.
      3. For each candidate article, print query + title + source + a short text
         snippet; read `y` (relevant) / `n` (not) / `s` (skip) / `q` (save+quit).
      4. **Spot-check pass:** after the pool, optionally surface a few extra
         articles (random sample, or same-source/same-day as a labeled-relevant
         one) to catch obvious pool misses; label those too.
      5. Append a well-formed line to `eval/testset.jsonl` with the accumulated
         `relevant_article_ids`. Make it **resumable** — skip `query_id`s already
         present in the output so a session can stop and continue (this is a
         multi-day manual job).
- [ ] **Article snippet accessor** — labeling needs `title / source / short text`
      per `article_id`. Pooled candidates already carry these in the retriever hit
      dict (no DB read needed). Only the spot-check's "other articles" need a DB
      fetch — add a tiny `get_articles_by_ids(ids)` / random-sample accessor to
      `database.py` only if the spot-check path actually needs it.
- [ ] **Validation script** (`eval/validate_testset.py` or a pytest) — assert:
      every line parses; `query_id`s unique; `relevant_article_ids` all exist in
      `articles`; `out_of_domain` ⇒ empty relevant set and vice-versa; count is
      50–100; in-domain queries have ≥1 relevant id. Run it before declaring done.
- [ ] **Commit the JSONL.** `eval/testset.jsonl` is hand-verified ground truth and
      **belongs in git** (unlike `data/`). Confirm `.gitignore` doesn't sweep
      `eval/` — add a negative pattern if needed.

## Done when

`eval/testset.jsonl` holds 50–100 hand-verified rows that pass the validation
script, every `relevant_article_id` resolves to a real article, the out-of-domain
subset is present and flagged, and the file is committed. Ship F can load it and
score `retrieve()` with no further labeling.

## Watch-outs

- **Pooling bias.** The pool is built from `retrieve()` under one config — labels
  can miss relevant articles that *this* retriever never surfaces, and (worse for
  Ship H) skew the eval toward the config used to pool. Mitigate by pooling
  **deep** (k ≈ 20–30, not 5) and by the spot-check pass. If Ship H compares
  embedding models, note that an ideal pool unions retrievals across the compared
  configs — that deeper re-pool can be deferred but must be flagged in Ship H, not
  silently ignored.
- **Article-level vs chunk-level.** Labels are article ids; retrieval returns
  chunks. Dedup chunks→articles when pooling (don't double-prompt) and remember
  Ship F maps a chunk hit to a label via its `article_id`.
- **Resumability.** 50–100 manual labels won't happen in one sitting. The helper
  must skip already-labeled `query_id`s and append, never truncate.
- **Don't let the LLM both write the query and pick the answer.** The generator
  proposes *queries*; relevance labels are decided by **you** in the CLI, not by
  the model. Hand-verification is the deliverable's whole value.
- **Keep out-of-domain a flagged minority.** Useful for the abstention metric, but
  if they leak unflagged into in-domain P/R they distort it. Type field gates this.
- **`POOL_DEPTH` is eval tooling, not runtime config** — keep it local to the
  labeling script; only `RETRIEVAL_TOP_K` (Ship D) lives in `Settings`.

## Deferred to later ships

- Loading the test set + computing retrieval precision/recall + latency/cost →
  **Ship F**.
- RAGAS faithfulness / answer-relevance over the same queries → **Ship G**.
- Multi-config sweep and (if pursued) union-pooling across configs → **Ship H**.
