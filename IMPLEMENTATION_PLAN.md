# Implementation Plan — FinNews-RAG (Project B)

**Plan revised:** 2026-05-20
**Direction:** Evolve the working MVP into **FinNews-RAG** — a retrieval-augmented
financial-news Q&A system whose headline differentiator is a **self-evaluation
harness** (retrieval precision/recall, faithfulness, latency/cost across configs).
**References:** `PROJECT_B_README.md` (vision), `financial_news_system_design_notes.md` (design notes)

## Goal

Turn the daily-summary MVP into a system that:
1. Ingests financial news from 8–10 free RSS feeds (World News API secondary).
2. **Chunks** articles and indexes chunk-level embeddings in ChromaDB.
3. **Retrieves** top-k chunks for a user query and generates a **grounded,
   source-cited answer** constrained to the retrieved context.
4. **Evaluates** itself — retrieval P/R on a hand-labeled set, RAGAS faithfulness
   and answer-relevance, latency/cost — and compares configurations.
5. Retains the daily briefing report as a secondary feature.

## Decisions (locked in)

| Decision | Choice | Rationale |
|---|---|---|
| Product direction | RAG Q&A + eval harness = headline; daily briefing = secondary | Defensible portfolio angle; pull model over push |
| Scope | Evolve in place (same repo) | Pipeline already works — reuse, don't rebuild |
| Orchestration framework | None — raw OpenAI / ChromaDB / sentence-transformers | Shows internals; avoids rewrite churn |
| Evaluation | Hybrid: RAGAS (faithfulness, answer-relevance) + custom (retrieval P/R, latency/cost) | Recognized metrics fast + ground-truth retrieval metrics |
| Demo UI | Streamlit | Fast to build, screenshots well |
| RSS strategy | RSS-first, 8–10 free/open feeds | Source diversity drives retrieval quality |
| World News API | Secondary fallback (backfill / low RSS yield) | Not primary |
| Paywalled feeds (FT, WSJ, Bloomberg) | Excluded | Extraction failure rate too high |
| Vector store | ChromaDB (persistent) | Already built |
| Embeddings | `all-MiniLM-L6-v2` | Local, free — also an eval comparison axis |
| Generation LLM | OpenAI `gpt-4o-mini` | Cost-effective |
| Chunking | ~256 tokens, ~15% overlap | Chunk size is an eval comparison axis |
| Test set | 50–100 query/relevant-doc pairs, **labels hand-verified** | Ground truth for retrieval metrics |
| Test-set granularity | Relevance labeled at **article** level | A retrieved chunk counts as relevant if its `article_id` is labeled relevant |
| DB migration | Wipe + recreate `data/news.db`; `output/*.md` preserved | Cleanest |
| Clustering (old v2 stage) | **Deferred / optional** | Not needed for query-time RAG |
| Hallucination check (old v2 Week 4) | **Folded into eval harness faithfulness metric** | Don't build it twice |
| Alembic | Not used | One-time wipe |
| Tiered hot/warm/cold retention | Deferred | Storage small (<200 MB / year) |

## What carries over vs. what's new

| Layer | Status |
|---|---|
| Config, RSS + World News ingestion, cleaner/NER, embeddings, SQLite + ChromaDB | Reuse |
| Dedup (Stages 1–2: URL canon + content hash) | Keep — prevents duplicate chunks polluting retrieval/eval |
| Daily briefing summarizer + report generator | Keep as secondary feature, unchanged |
| Chunking layer | New |
| Query-time retriever (top-k) | New |
| Grounded, source-cited Q&A | New |
| Evaluation harness + labeled test set + multi-config runner | New |
| Streamlit demo (`app.py`) | New |

New code areas: `src/rag/` (chunker, retriever, qa), `src/evaluation/` (harness,
metrics, testset), `app.py` (Streamlit), `evaluate.py` (eval entry point),
`eval/testset.jsonl` (labeled set).

## Schema (v2 + RAG)

```
articles  (SQLite)
  id, url, canonical_url (UNIQUE), content_hash (UNIQUE),
  title, description, content, source,
  published_at (UTC ISO 8601), fetched_at,
  extraction_method ('trafilatura' | 'newspaper3k' | 'readability' | 'rss-only'),
  indexed   (bool)   -- chunked + embedded into ChromaDB
  processed (bool)   -- included in a daily briefing

daily_reports  (SQLite)
  id, report_date (UNIQUE), content, article_count, created_at
  -- retained forever

ChromaDB collection "financial_news"
  one entry per CHUNK; id = "{article_id}:{chunk_index}"
  metadata: article_id, chunk_index, title, source, url, published_at
  document: chunk text
```

Dropped from the old v2 schema: `cluster_id` column and the `clusters` table
(clustering deferred). No SQLite `chunks` table — chunk→article mapping lives in
ChromaDB metadata; eval maps retrieved chunks to `article_id`.

## Structured output schemas

```python
# Grounded Q&A — lock before Ship D
class Citation(BaseModel):
    marker: int          # the [1], [2]... number used inline in `answer`
    chunk_id: str        # "{article_id}:{chunk_index}"
    article_id: int
    url: str

class GroundedAnswer(BaseModel):
    answer: str                   # inline [n] markers reference `citations`
    citations: list[Citation]
    answered_from_context: bool   # False -> retrieved context was insufficient
```

The daily-briefing structured model is unchanged from the MVP.

## Roadmap — 3-day ships

| Ship | Dates | Focus | Status |
|---|---|---|---|
| A | 2026-05-20 → 05-22 | Ingestion overhaul — RSS-first, full-text extraction, source health | **Current** |
| B | 2026-05-23 → 05-25 | Dedup Stages 1–2 (URL canon + content hash); wipe + recreate DB | Planned |
| C | 2026-05-26 → 05-28 | Chunking layer + chunk-level embeddings indexed in ChromaDB | Planned |
| D | 2026-05-29 → 05-31 | Retriever + grounded cited Q&A; Streamlit skeleton | Planned |
| E | 2026-06-01 → 06-03 | Labeled test set (50–100 query/relevant-doc pairs) | Planned |
| F | 2026-06-04 → 06-06 | Eval harness pt.1 — retrieval precision/recall + latency/cost | Planned |
| G | 2026-06-07 → 06-09 | Eval harness pt.2 — RAGAS faithfulness + answer-relevance | Planned |
| H | 2026-06-10 → 06-12 | Multi-config comparison runner + written findings | Planned |
| I | 2026-06-13 → 06-15 | Streamlit polish + README finalize; stretch: signal extraction | Planned |

**Cadence rule:** at each 3-day boundary, re-read this file, adjust the table if
the previous ship slipped, and rewrite only the next ship's detail section to full
resolution. Keep one ship detailed at a time.

### Ship A — Ingestion overhaul (CURRENT, 2026-05-20 → 2026-05-22)

**Goal:** RSS-first ingestion with full-text extraction and per-source health
monitoring. Commit the in-progress work first.

- [ ] Commit the uncommitted Week 1 changes already in the working tree
      (`rss_reader.py`, `world_news_api.py`, `config.py`, `test_ingestion.py`,
      `requirements.txt`).
- [ ] Finalize the 8–10 free feed list. Candidates: Reuters Business, CNBC,
      MarketWatch, Yahoo Finance, CoinDesk (crypto), Federal Reserve press
      releases, SEC press releases, Investing.com markets.
- [ ] Move the feed list to config (env var or YAML) — not hardcoded.
- [ ] Add a full-text extraction fallback chain: `trafilatura` → `newspaper3k`
      → `readability-lxml`. Record which path succeeded as `extraction_method`.
- [ ] Per-source health tracking: success count, last-success timestamp,
      extraction-success rate; log on every run.
- [ ] Demote `WorldNewsAPIClient` to secondary — call only if RSS yield is below
      threshold (~15 articles) or in backfill mode.
- [ ] Normalize all timestamps to UTC ISO 8601 at the ingestion boundary.
- [ ] Update `tests/test_ingestion.py`.

**Done when:** a single `main.py --mode ingest` pulls full text from 8+ feeds,
logs per-source health, and falls back to World News API only when RSS yield is
short.

## Risk register

| Risk | Mitigation |
|---|---|
| Trafilatura/newspaper3k flaky on some sites | Three-tier fallback; tolerate ~10% extraction failures, exclude those articles |
| Chunk size too small loses context / too large dilutes retrieval | Chunk size is an eval comparison axis; start at 256 tokens |
| Test set too small to be meaningful | 50–100 pairs minimum; hand-verify every relevance label |
| RAGAS LLM-as-judge calls add cost | Run on a subset; cache judge outputs; budget per eval run |
| OpenAI structured-output rejects schema | Keep `GroundedAnswer` flat; test with small fixtures first |
| Config tuning eats more than allotted time | Hard-cap Ship H; ship with defaults (chunk 256, top-k 5) if inconclusive |
| Scope creep into clustering / tiered retention | Explicitly deferred — re-read this doc if tempted |

## Out of scope for FinNews-RAG v1

- Cross-day topic clustering and weekly briefings.
- Tiered hot/warm/cold retention (defer until DB > 1 GB).
- Paywalled feeds (FT, WSJ, Bloomberg).
- Finnhub/Marketaux replacement for World News API.
- Real-time market data integration.
- Email/Slack delivery, multi-language.
- Backtest of sentiment vs. price movement (README stretch goal only).
