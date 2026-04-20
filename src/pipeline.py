from src.config import Settings
from src.ingestion.world_news_api import WorldNewsAPIClient
from src.ingestion.rss_reader import RSSReader
from src.processing.cleaner import TextCleaner
from src.processing.embeddings import EmbeddingGenerator
from src.processing.deduplicator import Deduplicator
from src.storage.database import Database
from src.storage.vector_store import VectorStore
from src.summarization.llm_client import LLMClient
from src.summarization.report_generator import ReportGenerator


# ---------- Module-level component init (heavy models load once on import) ----------
settings = Settings()  # type: ignore

news_api = WorldNewsAPIClient(settings)
rss      = RSSReader()
cleaner  = TextCleaner()
embedder = EmbeddingGenerator(settings)
dedup    = Deduplicator(embedder)
db       = Database()
vstore   = VectorStore(settings)
llm      = LLMClient(settings)
reporter = ReportGenerator(settings)


def run_pipeline() -> tuple[str, int]:
    """Run the full ingest → summarize → export pipeline.

    Returns:
        (summary_markdown, article_count)

    Raises:
        RuntimeError on pipeline failure. Callers translate to their own
        error type (HTTPException for the API, log-and-swallow for the scheduler).
    """
    from datetime import datetime

    # 1. Fetch from both sources
    articles = news_api.fetch_financial_news() + rss.fetch_from_feeds()
    if not articles:
        raise RuntimeError("No articles fetched")

    # 1b. Drop articles missing required fields
    articles = [a for a in articles if a.get("title") and a.get("content")]
    if not articles:
        raise RuntimeError("No valid articles after filtering")

    # 2. Clean each article's content in place
    for a in articles:
        a["content"] = cleaner.clean_article(a["content"])

    # 3. Deduplicate
    articles = dedup.deduplicate_articles(articles)

    # 4. Persist to SQL (skips existing URLs)
    db.save_articles(articles)

    # 5. Generate embeddings and push to ChromaDB
    texts = [f"{a['title']} {a.get('description', '')}" for a in articles]
    embeddings = embedder.generate_embeddings(texts).tolist()
    vstore.add_articles(articles, embeddings)

    # 6. LLM summary
    summary = llm.generate_summary(articles)
    if not summary:
        raise RuntimeError("LLM returned empty summary")

    # 7. Export Markdown + HTML
    now = datetime.now()
    reporter.save_markdown(summary, now)
    reporter.generate_html(summary, now)

    # 8. Save report row in DB
    db.save_report(summary, len(articles))

    return summary, len(articles)
