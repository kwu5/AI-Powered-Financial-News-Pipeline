"""evaluate.py — Ship F entry point: retrieval precision/recall + latency.

Loads the committed test set (eval/testset.jsonl), scores `Retriever.retrieve()`
against it at the served top-k plus a k-sweep, reports out-of-domain abstention
accuracy separately, and writes a Markdown + JSON report under output/eval/.

Retrieval is local (MiniLM + ChromaDB) so it costs nothing; the only LLM calls are
the out-of-domain abstention checks (a handful). Read-only over the corpus.

Run:  python evaluate.py [--top-k N] [--k-sweep 1,3,5,10] [--testset PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from datetime import datetime

from src.config import Settings
from src.evaluation.harness import EvalReport, evaluate_retrieval
from src.evaluation.testset import TESTSET_PATH, load_testset
from src.processing.embeddings import EmbeddingGenerator
from src.rag.qa import QAEngine
from src.rag.retriever import Retriever
from src.storage.vector_store import VectorStore
from src.summarization.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def build_markdown(report: EvalReport) -> str:
    m = report.meta
    lines = [
        f"# Retrieval eval — {m.get('timestamp', '')}",
        "",
        f"**Test set:** `{m.get('testset_path')}` "
        f"({report.n_in_domain} in-domain, {report.n_out_of_domain} out-of-domain)  ",
        f"**Served top_k:** {report.top_k}  ·  **k-sweep:** {report.k_values}  ",
        f"**Embedding:** `{m.get('embedding_model')}`  ·  "
        f"**Chunk:** {m.get('chunk_size')}/{m.get('chunk_overlap')} tok  ·  "
        f"retrieval cost ≈ $0 (local); LLM calls: {report.n_out_of_domain} (OOD abstention only)",
        "",
        "## Retrieval quality (in-domain, article-level)",
        "",
        "| k | precision | recall | hit-rate |",
        "|---|-----------|--------|----------|",
    ]
    for k in report.k_values:
        row = report.per_k_means[k]
        marker = "  ← served" if k == report.top_k else ""
        lines.append(
            f"| {k} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['hit'])} |{marker}"
        )
    lines += [
        "",
        f"**MRR** (first relevant article, depth {max([*report.k_values, report.top_k])}): "
        f"{_fmt(report.mean_mrr)}",
        "",
        "## Out-of-domain abstention",
        "",
    ]
    if report.abstention_accuracy is None:
        lines.append("_No out-of-domain queries in the set._")
    else:
        n_correct = sum(1 for r in report.results if r.abstention_correct)
        lines.append(
            f"**{_fmt(report.abstention_accuracy)}** "
            f"({n_correct}/{report.n_out_of_domain} abstained correctly — "
            f"`answered_from_context=False` on off-topic queries)"
        )
    lines += [
        "",
        "## Latency (retrieval path, per query)",
        "",
        f"p50 {_fmt(report.latency_p50_ms, 1)} ms · p95 {_fmt(report.latency_p95_ms, 1)} ms  ",
        "_Embed query + ChromaDB search; ~independent of k. Excludes model cold-start._",
        "",
        "## Caveats",
        "",
        "- **Recall is capped by Ship E pooling.** Labels were pooled from this same "
        "retriever, and ~7 in-domain seeds were never surfaced (`retrieve()` missed them), "
        "so low recall is partly a labeling ceiling, not purely a retriever failure. "
        "Flagged for the Ship I audit.",
        "- **precision@k denominator = distinct articles surfaced by the top-k chunks** "
        "(not k). Most in-domain queries have a single relevant article, so **hit-rate / "
        "recall / MRR** are the meaningful headline; precision runs low by construction.",
        "- Single config (top_k, chunk size, embedding model held at defaults). "
        "Sweeping those is Ship H.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ship F — score retrieve() against the test set.")
    parser.add_argument("--top-k", type=int, default=None, help="served depth (default RETRIEVAL_TOP_K)")
    parser.add_argument("--k-sweep", type=str, default="1,3,5,10", help="comma-separated k values")
    parser.add_argument("--testset", type=str, default=TESTSET_PATH)
    parser.add_argument("--out", type=str, default=None, help="output dir (default <OUTPUT_DIR>/eval)")
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    top_k = args.top_k if args.top_k is not None else settings.RETRIEVAL_TOP_K
    k_values = [int(x) for x in args.k_sweep.split(",") if x.strip()]
    out_dir = args.out or os.path.join(settings.OUTPUT_DIR, "eval")

    testset = load_testset(args.testset)
    if not testset:
        logger.error("Test set %s is empty — nothing to evaluate.", args.testset)
        return

    embedder = EmbeddingGenerator(settings)
    vstore = VectorStore(settings)
    if vstore.financial_news_collection.count() == 0:
        logger.error("ChromaDB is empty — populate the index first (run the pipeline). Aborting.")
        return
    retriever = Retriever(embedder, vstore)
    qa_engine = QAEngine(retriever, LLMClient(settings))

    logger.info(
        "Evaluating %d queries (top_k=%d, k-sweep=%s)…", len(testset), top_k, k_values
    )
    report = evaluate_retrieval(testset, retriever, qa_engine, top_k, k_values)
    report.meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "testset_path": args.testset,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE_TOKENS,
        "chunk_overlap": settings.CHUNK_OVERLAP_TOKENS,
    }

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(out_dir, f"retrieval_eval_{stamp}.md")
    json_path = os.path.join(out_dir, f"retrieval_eval_{stamp}.json")

    markdown = build_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, ensure_ascii=False, indent=2)

    print("\n" + markdown)
    print(f"\nWrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
