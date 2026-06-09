# Ship D — Retriever + grounded cited Q&A + Streamlit skeleton

**Status:** Done (completed 2026-06-08)
**Parent plan:** `IMPLEMENTATION_PLAN.md` (roadmap ship D)
**Predecessors:** Ship A (ingestion) ✅, Ship B (dedup) ✅, Ship C (chunk-level ChromaDB) ✅
**Schedule:** Week 1 · Jun 1–7 (`doc/june-weekly-schedule.md`). This is **the demo** —
the moment it lands, a Streamlit screenshot goes in the README and the posted link
becomes demo-backed.

## Goal

Turn the chunk-level index from Ship C into a **query-time RAG path**: take a user
question, retrieve the top-k most relevant chunks, and generate a **grounded answer
constrained to that retrieved context**, with inline `[n]` citation markers that
resolve to real source articles. Surface it through a minimal Streamlit app.

This is the headline differentiator's first visible payoff — everything in Ships
E–H exists to *measure* this path, so it has to exist first.

## Why now

Ship C made retrieval *possible* (passage-sized vectors with `article_id` in
metadata). Ship D makes it *usable*: a retriever wrapper, a cited-answer
generator, and a UI. Ship E (labeled test set) and Ships F–G (eval harness) all
need a working `answer_query()` to evaluate, so D is the prerequisite for the
entire eval half.

## Resolved up front

- **Citations are built from the retrieved chunks, not from the LLM's output.**
  The LLM is told to cite by source *number* (`[1]`, `[2]`…). We map those numbers
  back to the authoritative `chunk_id` / `article_id` / `url` we already hold from
  retrieval. The model never hand-copies a URL — it can't hallucinate a citation
  target, only mis-number one. This is the single most important design decision
  in the ship.
- **No new DB accessors.** ChromaDB chunk metadata already carries
  `article_id / chunk_index / title / source / url / published_at` (added in
  Ship C). The retriever returns those directly, so QA + citations need zero
  SQLite reads.
- **The daily briefing path is untouched.** Ship D adds a *parallel* query-time
  path (`src/rag/`). `run_pipeline()` and `LLMClient.generate_summary()` don't
  change.
- **ChromaDB result shape is nested.** `collection.query(...)` returns each field
  wrapped one level deep — `results["ids"][0]`, `results["metadatas"][0]`, etc.
  (one inner list per query; we send a single query). The retriever flattens this
  so nothing downstream touches `[0]`.

## Structured output schemas (lock these before writing qa.py)

Already drafted in `IMPLEMENTATION_PLAN.md`; restated here as the build target.
Live in `src/rag/qa.py`.

```python
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

Keep flat — gpt-4o-mini structured output is reliable on flat schemas. `marker`
is an int that indexes the numbered source block we build at prompt time.

## Tasks

- [x] **Config** (`src/config.py`) — add `RETRIEVAL_TOP_K: int = 5`. Becomes an
      eval comparison axis in Ship H, so make it a setting, not a constant.
- [x] **New module `src/rag/retriever.py`** — a thin wrapper over the existing
      `EmbeddingGenerator` + `VectorStore.search_similar`. Signature roughly
      `retrieve(query: str, top_k: int) -> list[dict]`. Steps: embed the query
      (`generate_embedding(query).tolist()`), call
      `search_similar(query_embedding, n_results=top_k)`, then **flatten** the
      nested ChromaDB result into a clean list of hit dicts:
      `{chunk_id, article_id, text, title, source, url, published_at, distance}`
      (`chunk_id` = `results["ids"][0][i]`, `text` = `documents[0][i]`, the rest
      from `metadatas[0][i]`, `distance` = `distances[0][i]`). Empty collection or
      no hits → return `[]`. Construct with the same shared `EmbeddingGenerator` /
      `VectorStore` instances the pipeline uses (don't reload the model).
- [x] **New module `src/rag/qa.py`** — holds the `Citation` / `GroundedAnswer`
      schemas and `answer_query(query: str, top_k: int) -> GroundedAnswer`:
      1. `hits = retrieve(query, top_k)`.
      2. **Empty-retrieval short-circuit:** if `not hits`, return
         `GroundedAnswer(answer="I don't have enough indexed context to answer
         that.", citations=[], answered_from_context=False)` — **no LLM call**.
      3. Build a numbered context block: `[1] <chunk text>\n[2] <chunk text>…`,
         numbering = position in `hits` (1-based).
      4. Call the grounded generator (see next task) → model returns `answer`
         (with inline `[n]`), the set of source numbers it actually used, and
         `answered_from_context`.
      5. **Build citations from `hits`, not from the model**: for each source
         number `n` the model cited, look up `hits[n-1]` and emit
         `Citation(marker=n, chunk_id=…, article_id=…, url=…)`. Drop any number
         out of range (model mis-cited) and log it.
- [x] **Grounded generation** — add `generate_grounded_answer(query: str,
      numbered_context: str) -> <parsed>` to `LLMClient` (small mod to an existing
      file → drafted directly per the collaboration split). Use OpenAI structured
      output (`client.beta.chat.completions.parse`, `response_format=` a small
      flat pydantic model carrying `answer`, `used_markers: list[int]`,
      `answered_from_context: bool`). System prompt: *answer ONLY from the
      numbered context; cite every claim with the bracketed source number; if the
      context doesn't contain the answer, set `answered_from_context=false` and
      say so — do not use outside knowledge.* `temperature=0` for determinism
      (matters for Ship F/G eval reproducibility).
- [x] **New `app.py`** (Streamlit) — query text box + "Ask" button →
      `answer_query(query, settings.RETRIEVAL_TOP_K)` → render `answer` markdown,
      then a **Sources** list (one line per citation: `[n] title — source — url`).
      If `answered_from_context` is False, show a muted "answered from context: no"
      note. Minimal styling — polish is Ship I. Reuse the module-level singletons
      so the embedder loads once.
- [x] **Tests** (`tests/test_retriever.py` / `tests/test_qa.py`) — retriever:
      flattening produces k well-formed hit dicts from a mocked `search_similar`
      result; empty result → `[]`. qa: empty hits → `answered_from_context=False`
      with no LLM call (mock the LLM, assert not called); a normal mocked LLM
      response → citations resolve to the right `chunk_id`/`url` by marker; an
      out-of-range marker is dropped.
- [x] **Smoke test** (`tests/smoke_ship_d.py`) — against the real index: an
      in-domain finance query returns a non-empty answer with ≥1 citation whose
      `url` matches a retrieved hit; an obviously out-of-domain query (e.g.
      "best pizza in Naples") returns `answered_from_context=False`.

## Done when

Typing a finance question into the Streamlit app returns a grounded answer with
inline `[n]` markers and a resolved source list; out-of-domain or
nothing-retrieved queries report insufficient context instead of fabricating;
`answer_query()` is importable and returns a `GroundedAnswer` (the seam Ship F
will evaluate).

## Watch-outs

- **Flatten the `[0]`.** Every ChromaDB query field is nested one level. Do it
  once in the retriever; never let `[0]` leak into qa.py or app.py.
- **Citations from retrieval, never from the LLM.** Trust the model for the
  *answer text and which numbers it used* — nothing else. `chunk_id`/`url` come
  from `hits`.
- **Empty / thin index.** Early on the collection may hold few chunks;
  `n_results` larger than the collection is fine (ChromaDB caps it), but handle
  the zero-hit case explicitly before any LLM call.
- **`temperature=0`** on the grounded call — Ships F/G replay the same queries and
  need stable outputs.
- **Distances are cosine distance (1 − similarity), lower = closer.** If a
  relevance floor is ever wanted, threshold on distance — but defer; for Ship D,
  top-k with no floor is fine.
- **Streamlit cold start** imports the embedder (~loads the model). Acceptable;
  note it so it isn't mistaken for a hang on first query.

## Deferred to later ships

- Labeled query/relevant-doc test set → **Ship E**.
- Retrieval precision/recall + latency/cost harness → **Ship F**.
- RAGAS faithfulness + answer-relevance → **Ship G**.
- Multi-config sweep over `RETRIEVAL_TOP_K` / chunk size / embedding model →
  **Ship H**.
- Relevance-floor on distance, conversational history, Streamlit polish →
  **Ship I** / stretch.
