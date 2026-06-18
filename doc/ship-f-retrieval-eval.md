# Ship F — Eval harness pt.1: retrieval P/R + latency/cost

**Status:** Done (built + first run 2026-06-17; committed 2026-06-18)
**Parent plan:** `IMPLEMENTATION_PLAN.md` (roadmap ship F)
**Predecessors:** Ship D (`retrieve()` / `answer_query()` seam) ✅, Ship E
(`eval/testset.jsonl` hand-verified ground truth, q001–q093) ✅
**Schedule:** see `doc/june-weekly-schedule.md` (scheduling is tracked there, not here).

## Goal

Score `retrieve()` against `eval/testset.jsonl` and put numbers on retrieval
quality:

- **Article-level precision / recall / MRR** over the in-domain queries.
- **Abstention accuracy** over the out-of-domain queries, reported separately.
- **Latency** per query (embed + ChromaDB query), p50/p95.

Output a small report (Markdown + JSON) under `output/eval/`. This is the first
half of the headline differentiator — the test set built in Ship E finally gets
used.

## Why now

Ship E produced trustworthy ground truth but nothing scores against it. Ship F
turns the `retrieve()` / `answer_query()` seam from "works in the demo" into
"measurably this good" and lays the metric plumbing that Ships G (faithfulness /
answer-relevance) and H (multi-config sweep) extend.

## Decisions (locked 2026-06-17)

| Decision | Choice | Rationale |
|---|---|---|
| Relevance granularity | **Article level** | A chunk hits iff its `article_id` ∈ `relevant_article_ids`; matches Ship E labels and survives chunk-size changes |
| Metric set | `precision@k`, `recall@k`, `MRR` | Standard, cheap, interpretable; nDCG deferred (labels are binary, not graded) |
| k reporting | Headline at `RETRIEVAL_TOP_K` (5); sweep k ∈ {1,3,5,10} | Headline = what the system serves; sweep gives Ship H a baseline curve |
| In vs out of domain | P/R/MRR over **in-domain only**; OOD scored as **abstention** | Mixing OOD `[]` into P/R distorts it (Ship E watch-out) |
| Abstention signal | `answer_query().answered_from_context` | Ship D already short-circuits to `False` on empty/insufficient retrieval |
| Cost | **Latency only** (retrieval path is local MiniLM + local ChromaDB ≈ free) | No API spend until the LLM-judge in Ship G; report latency instead |
| Determinism | Reuse existing `temperature=0` paths; embeddings are deterministic | Reproducible numbers across runs |
| Mutation | Harness is **read-only** over DB + ChromaDB | Eval must not alter the corpus it measures |

## Seams it consumes (do not modify these)

- **`eval/testset.jsonl`** — rows: `query_id, query, relevant_article_ids: list[int],
  source, type ∈ {in_domain, out_of_domain}, notes`. OOD ⇒ `relevant_article_ids == []`.
- **`Retriever.retrieve(query, top_k) -> list[dict]`** — each hit:
  `{chunk_id, article_id, text, title, source, url, published_at, distance}`,
  ordered by ascending distance (best first). Empty query/no hits ⇒ `[]`.
- **`qa.answer_query(query, top_k) -> GroundedAnswer`** — `answered_from_context: bool`.
- **`Settings.RETRIEVAL_TOP_K`** (= 5) — the served depth.

## Metric definitions (as implemented)

`R` = the query's relevant article-id set. Retrieve `max(k_sweep ∪ {top_k})`
chunks once. For each depth `k`, take the first `k` **chunks**, then **dedup to
unique `article_id`s preserving rank** → `A_k` (the articles the system actually
surfaces at depth k; top chunks often share an article). All metrics are functions
of `(A_k, R)` only — pure, in `metrics.py`:

- **precision@k** = `|A_k ∩ R| / |A_k|` — denominator is the *distinct articles
  surfaced* at depth k, **not k**. Dividing by k would conflate "few chunks" with
  "wrong articles". `A_k` empty → 0. (Refined from the earlier draft, which divided
  by k.)
- **recall@k** = `|A_k ∩ R| / |R|`. Empty `R` → 0 (never happens for in-domain).
- **hit@k** = 1 if `A_k ∩ R` else 0 (success@k).
- **MRR** = `1 / rank` of the first relevant article in `A` at the deepest depth;
  0 if none surfaced.
- **abstention_correct** (OOD) = `answered_from_context is False`. (The helper also
  scores the in-domain direction — answered ⇒ correct — as an available diagnostic.)
- **Aggregate** = mean of each metric over in-domain queries per `k`; abstention
  accuracy = mean over OOD.

Note: most in-domain queries have **a single** relevant article, so `recall@k` /
`hit@k` collapse to "was the one right article in the top k?" — a meaningful, blunt
signal — while **precision runs low by construction**. Multi-article rows (the
clusters labeled in Ship E) are where recall<1 is informative.

## Tasks (assistant-implemented 2026-06-17; user to review)

- [x] **`src/evaluation/testset.py`** — `load_testset()` → frozen `TestQuery`
      dataclass rows; trusts a file already checked by `eval/validate_testset.py`.
- [x] **`src/evaluation/metrics.py`** — *pure* functions over
      `(ranked_article_ids, relevant_set)`: `precision`, `recall`, `hit_rate`,
      `reciprocal_rank`, plus `abstention_correct(answered_from_context, is_out_of_domain)`.
      No I/O, no retriever import. (k-slicing happens in the harness, not here.)
- [x] **`src/evaluation/harness.py`** — `evaluate_retrieval()`: warm-up; per row,
      in-domain calls `retrieve()` (timed) → slice chunks[:k] → `dedup_to_articles`
      → per-k metrics; OOD calls `answer_query()` → abstention. Aggregates split by
      `type`; returns an `EvalReport`. No printing.
- [x] **`evaluate.py`** — CLI: `--top-k` (default `RETRIEVAL_TOP_K`),
      `--k-sweep 1,3,5,10`, `--testset`, `--out` (default `<OUTPUT_DIR>/eval`).
      Builds embedder/vstore/llm/retriever/qa directly (not via `pipeline`, to avoid
      its ingestion side-effects), aborts on empty index, runs harness, writes report.
- [x] **Report** (`output/eval/retrieval_eval_<date>.md` + `.json`) — per-k
      precision/recall/hit table, MRR, abstention accuracy (`x/n`), latency p50/p95,
      run metadata, and the explicit **pooling-bias caveat** (recall capped by Ship E
      pooling; ~7 known misses + assistant-labeled half flagged for the Ship I audit).
- [x] **Tests** (`tests/test_evaluation.py`, 11 passing) — `metrics.py` on
      hand-built fixtures (perfect / miss / partial / empty / empty-relevant +
      abstention truth table); loader round-trip; harness with mocked retriever/qa
      covering dedup, in/out-of-domain split, and abstention scoring.

## Done when

`python evaluate.py` loads the committed test set and prints article-level
P/R/MRR at the served top-k plus the k-sweep, reports OOD abstention accuracy as a
separate number, reports per-query latency p50/p95, and states the pooling-bias
caveat in the output. `metrics.py` is covered by unit tests.

## Findings (first run, 2026-06-17)

Full report: `output/eval/retrieval_eval_2026-06-17.md` (+ `.json`). 88 in-domain
+ 5 out-of-domain; defaults (top_k=5, MiniLM, chunk 256/38).

| metric | k=1 | k=3 | k=5 (served) | k=10 |
|---|---|---|---|---|
| precision | 0.670 | 0.462 | 0.326 | 0.161 |
| recall | 0.640 | 0.743 | **0.784** | 0.835 |
| hit-rate | 0.670 | 0.761 | **0.795** | 0.852 |

MRR 0.729 · OOD abstention 4/5 (0.80) · latency p50 19.8 ms / p95 24.2 ms.

**Read: the retriever is strong; the headline is dragged down by Ship E label
debt, not by retrieval.** When the relevant article is findable it lands at rank
1–2 (hit@1 = 0.67, MRR 0.73), and the misses are not random — they concentrate on
queries already flagged during labeling.

The 13 in-domain misses@10 decompose as:
- **9 documented pooling misses** (label points to an article `retrieve()` never
  surfaced → a recall *ceiling*, not a retriever failure): q023, q024, q046, q048,
  q059, q064, q076, q088, q093.
- **4 vague / self-referential queries** (flagged at labeling): q007, q032, q082, q084.

**Excluding only the 9 pooling misses → hit@5 ≈ 0.89, hit@10 ≈ 0.95.** That gap
between raw (0.795) and label-adjusted (~0.89) *is* the Ship E label debt the
harness was built to surface — good validation of the eval approach.

Actionable:
- **top_k=5 is slightly tight.** 5 queries had the relevant article at rank 6–10
  (q027, q049, q050, q077, q089); recall 0.784 → 0.835 going k=5→10. Test a served
  top_k ≈ 8–10 in **Ship H** (top_k is already an eval axis).
- **Ship I audit worklist is now precise:** fix/re-pool the 9 pooling-miss labels;
  reword or drop q007/q032/q082/q084; q077 is mislabeled (article 214 *is*
  retrievable, but the real APY answer is the rate-tracker cluster, not 214).
- **OOD leak is q081 "How to become rich?"** — it legitimately overlaps the
  corpus's personal-finance articles, so it's a fuzzy OOD rather than a true
  failure; the distance relevance-floor (**Ship I**) is the lever if we want to
  tighten it.

## Watch-outs

- **Dedup chunks→articles before scoring** — multiple chunks of one article must
  not inflate precision; keep the best-ranked hit per `article_id`.
- **Keep OOD out of the P/R averages** — gate on `type`; abstention is its own line.
- **Recall is bounded by Ship E pooling.** Some relevant articles were never
  pooled (≈7 known misses where `retrieve()` didn't surface the seed), so low
  recall is partly a labeling ceiling, not purely a retriever failure. State it;
  don't silently treat the number as ground-truth-complete. (Audit → Ship I.)
- **Define and document the MRR cutoff and precision@k denominator** — small
  conventions that change the numbers; write them in the report so they're
  comparable in Ship H.
- **Empty-retrieval edge case** — `retrieve()` can return `[]`; metrics must yield
  `0`, not divide-by-zero.
- **Hold `top_k` / chunk params at defaults** — sweeping *configurations* (chunk
  size, embedding model, top_k as a tuning axis) is **Ship H**, not F. F sweeps
  only the *reporting* k over a fixed retrieval.
- **Latency hygiene** — warm the embedding model once before timing (exclude
  import/cold-start); measure embed + query only; report percentiles, not just mean.
- **Read-only** — never let the harness write to `data/news.db` or `data/chroma/`.

## Deferred to later ships

- RAGAS faithfulness / answer-relevance over the same queries → **Ship G**.
- Multi-config sweep (chunk size, embedding model, top_k) + union-pooling across
  configs → **Ship H**.
- Distance relevance-floor tuning (abstain when best distance is too large) →
  **Ship I**.
