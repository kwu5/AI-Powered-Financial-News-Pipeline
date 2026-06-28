"""src/evaluation/sweep.py — Ship H: OFAT multi-config comparison runner.

Sweep a small one-factor-at-a-time grid of retrieval/index configs around the
default baseline (chunk 256, top_k 5, MiniLM), run BOTH eval halves per config
(Ship F retrieval P/R/MRR + latency, Ship G faithfulness + answer-relevance +
cost), and collect one comparison row per config so evaluate.py --sweep can rank
them and doc/ship-h-findings.md can name a winner.

Forks resolved 2026-06-26:
  - FORK A: embedding model held FIXED at all-MiniLM-L6-v2 (mpnet cut). So the only
    expensive axis is chunk_size → 3 index builds, not 6.
  - FORK B: union-pooling DEFERRED to Ship I. Retrieval P/R is reported with the
    standing pooling-bias caveat; the headline config call leans on the bias-free
    generation metrics (faithfulness/relevance).

Cost structure that drives the design:
  - chunk_size change  → EXPENSIVE: re-chunk + re-embed the whole corpus into a
    fresh ChromaDB persist dir.
  - top_k change       → FREE: query-time only; the stored index is unchanged.
  So: build one index per UNIQUE (chunk_size, embedding_model); loop top_k innermost
  over that already-built index. evaluate_retrieval() already k-sweeps in a single
  call, so it runs ONCE per index; evaluate_generation() takes a single top_k, so it
  runs ONCE per top_k.

Read-only over the canonical corpus: reads articles from SQLite read-only, writes
ONLY to data/chroma_sweep/<index-slug>/ (gitignored under data/). NEVER touches
data/chroma/ or data/news.db. The judge cache + reports are the only other writes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import Settings
from src.evaluation.harness import EvalReport, GenerationReport
from src.evaluation.testset import TestQuery

# Baseline = the served default config (bold values in the plan's OFAT grid).
BASELINE_CHUNK_SIZE = 256
BASELINE_OVERLAP = 38
BASELINE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASELINE_TOP_K = 5

# OFAT axis values. Baseline value appears in each so build_grid can dedup it.
CHUNK_SIZES = (128, 256, 512)
TOP_KS = (3, 5, 10)

# Where throwaway per-config indexes live (under the already-gitignored data/).
SWEEP_ROOT = os.path.join("data", "chroma_sweep")


@dataclass(frozen=True)
class SweepConfig:
    """One point in the OFAT grid. Two configs that differ ONLY in top_k share the
    same physical index (see index_slug)."""
    chunk_size: int
    overlap: int
    embedding_model: str
    top_k: int

    @property
    def index_slug(self) -> str:
        """Identity of the *stored index* — (chunk_size, embedding_model) only.
        top_k is deliberately excluded: it's a free query-time knob, so configs that
        differ only in top_k must map to the SAME persist dir (build once, reuse).

        TODO: return a filesystem-safe slug, e.g. f"chunk{chunk_size}_{model_slug}".
        """
        raise NotImplementedError

    @property
    def label(self) -> str:
        """Human-readable row label for the comparison table, e.g.
        "chunk256 / top_k5 / MiniLM". Includes top_k (unlike index_slug)."""
        raise NotImplementedError


def build_grid(
    chunk_sizes=CHUNK_SIZES,
    top_ks=TOP_KS,
    embedding_model: str = BASELINE_EMBEDDING_MODEL,
    overlap: int = BASELINE_OVERLAP,
) -> List[SweepConfig]:
    """Enumerate the OFAT grid around the baseline. Vary ONE axis at a time:

      - chunk_size ∈ chunk_sizes, top_k = BASELINE_TOP_K   (3 configs; 3 index builds)
      - top_k ∈ top_ks, chunk_size = BASELINE_CHUNK_SIZE   (3 configs; 0 new builds)

    The baseline (chunk 256, top_k 5) is shared by both arms — emit it exactly ONCE.
    embedding_model is fixed (Fork A). Net: 5 distinct configs over 3 index builds.

    NOTE: overlap should scale with chunk_size to keep ~15% (e.g. round(0.15*size)),
    rather than pinning the baseline's 38 onto the 128/512 indexes — decide and
    document which, since it affects chunk boundaries.

    TODO: build the deduped list of SweepConfig. Order it so build_index runs once
    per index_slug (group by chunk_size).
    """
    raise NotImplementedError


def build_index(
    config: SweepConfig,
    settings: Settings,
    db,
    embedder,
    sweep_root: str = SWEEP_ROOT,
):
    """Build (or reuse) the throwaway chunk-level index for `config` and return a
    Retriever bound to it.

    Mirrors the pipeline.py indexing block (lines ~101–122) but: reads the WHOLE
    corpus (db.get_all_articles(), read-only), chunks at config.chunk_size /
    config.overlap, embeds with the shared MiniLM `embedder`, and writes to an
    ISOLATED persist dir data/chroma_sweep/<config.index_slug>/.

    Index isolation without forking VectorStore: clone settings with the per-config
    dir, then point a VectorStore at it —
        cfg_settings = settings.model_copy(update={"CHROMA_PERSIST_DIR": dir})
        vstore = VectorStore(cfg_settings)
    Different path ⇒ different ChromaDB ⇒ canonical data/chroma/ is untouched.

    Resumable: if the persist dir already exists AND its collection count > 0, SKIP
    the rebuild and just bind a Retriever to it.

    Steps when building:
      1. dir = os.path.join(sweep_root, config.index_slug)
      2. rows = db.get_all_articles()  # ORM rows, read-only
      3. for each row: convert to the chunk_article dict (id/content/title/source/
         url/published_at — see pipeline.py:104), chunk with embedder.model.tokenizer
         at config.chunk_size / config.overlap, collect chunks.
      4. embeddings = embedder.generate_embeddings([c["text"] for c in chunks]).tolist()
      5. vstore.add_chunks(chunks, embeddings)
      6. return Retriever(embedder, vstore)

    Embedder reuse: MiniLM is fixed (Fork A), so the SAME embedder serves every
    config. If the embedding-model axis is ever reopened, construct an
    EmbeddingGenerator per config.embedding_model here instead.

    TODO: implement; never write outside sweep_root.
    """
    raise NotImplementedError


@dataclass
class SweepRow:
    """One config's full result line for the comparison table."""
    config: SweepConfig
    retrieval: EvalReport          # k-swept; read metrics at config.top_k
    generation: GenerationReport   # judged at config.top_k
    # Convenience scalars pulled out for ranking/printing (fill from the reports):
    precision: Optional[float] = None
    recall: Optional[float] = None
    hit_rate: Optional[float] = None
    mrr: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    est_cost_usd: float = 0.0


def run_sweep(
    testset: List[TestQuery],
    settings: Settings,
    db,
    embedder,
    llm,
    cache,
    configs: Optional[List[SweepConfig]] = None,
) -> List[SweepRow]:
    """Run the whole sweep and return one SweepRow per config.

    Order to minimize re-indexing (the whole point):
      group configs by index_slug → for each unique index:
          retriever = build_index(any-config-with-this-slug, ...)   # ONCE
          qa_engine = QAEngine(retriever, llm)
          retr_report = evaluate_retrieval(testset, retriever, qa_engine,
                                           top_k=BASELINE_TOP_K, k_values=sorted top_ks)
              # k-swept in a single call → covers every top_k for this index
          for top_k in this index's configs (innermost, FREE):
              gen_report = evaluate_generation(testset, retriever, qa_engine, llm,
                                               embedder, top_k, cache)
              row = SweepRow(config, retr_report, gen_report, ...scalars at top_k...)

    Notes:
      - One shared `cache` (JudgeCache) across all configs is CORRECT: its key hashes
        (answer+context), which differ per chunk_size and per top_k, so each config
        re-judges — i.e. each pays a full generation-eval. Budget ~$0.05 × #configs.
      - Pull retrieval scalars from retr_report.per_k_means[config.top_k]; faithfulness
        / relevance from gen_report; est_cost via evaluate.py's _est_cost on the
        gen_report token totals (or recompute here).
      - Read-only over the corpus; only build_index writes (to sweep dirs).

    TODO: implement the grouped loop; return the rows.
    """
    raise NotImplementedError


def rank_rows(rows: List[SweepRow]) -> List[SweepRow]:
    """Return rows ordered best→worst for the findings table. Pure function over the
    scalar fields — no I/O, no index — so it unit-tests against hand-built SweepRows.

    Ranking key (Fork B): lead with the bias-free generation metrics, since retrieval
    recall carries pooling bias. Suggested primary sort = faithfulness, then
    answer-relevance, then hit_rate/MRR as tie-breakers; treat tiny gaps as ties
    (judge variance). Document the exact key in the findings doc.

    TODO: implement the sort; decide + document the tie threshold.
    """
    raise NotImplementedError
