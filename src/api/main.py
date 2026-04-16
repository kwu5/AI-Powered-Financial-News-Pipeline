from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel


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


# ---------- Pydantic request/response models ----------
class GenerateRequest(BaseModel):
    force_refresh: bool = False

class ReportResponse(BaseModel):
    date: str
    content: str
    article_count: int
    generated_at: datetime


# ---------- App + component initialization ----------
settings = Settings()   # type:ignore
app = FastAPI(title="Daily Financial News Summarizer", version="0.1.0")


news_api = WorldNewsAPIClient(settings)
rss      = RSSReader()
cleaner  = TextCleaner()
embedder = EmbeddingGenerator(settings)
dedup    = Deduplicator(embedder)
db       = Database()
vstore   = VectorStore(settings)
llm      = LLMClient(settings)
reporter = ReportGenerator(settings)


# ---------- Endpoints ----------
@app.get("/")
def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "endpoints": [r.path for r in app.routes if isinstance(r,APIRoute)]
    }


@app.get("/health")
def health() -> dict:
    return {"status": "OK"}

@app.post("/generate", response_model=ReportResponse)
def generate(req: GenerateRequest) -> ReportResponse:
    today = datetime.now().strftime("%Y-%m-%d")
    existing_md = Path(settings.OUTPUT_DIR) / f"financial_briefing_{today}.md"
    if existing_md.exists() and not req.force_refresh:
        return ReportResponse(
            date=today,
            content=existing_md.read_text(encoding="utf-8"),
            article_count=0,
            generated_at=datetime.fromtimestamp(existing_md.stat().st_mtime),
        )

    try:
        # 1. Fetch from both sources
        articles = news_api.fetch_financial_news() + rss.fetch_from_feeds()
        if not articles:
            raise HTTPException(status_code=503, detail="No articles fetched")

        # 1b. Drop articles missing required fields
        articles = [a for a in articles if a.get("title") and a.get("content")]
        if not articles:
            raise HTTPException(status_code=503, detail="No valid articles after filtering")

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
            raise HTTPException(status_code=502, detail="LLM returned empty summary")

        # 7. Export Markdown + HTML
        now = datetime.now()
        reporter.save_markdown(summary, now)
        reporter.generate_html(summary, now)

        # 8. Save report row in DB
        db.save_report(summary, len(articles))

        # 9. Return response
        return ReportResponse(
            date=now.strftime("%Y-%m-%d"),
            content=summary,
            article_count=len(articles),
            generated_at=now,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")




@app.get("/report/{report_date}", response_class=HTMLResponse)
def get_report(report_date: str) -> str:
    path = Path(settings.OUTPUT_DIR)/f"financial_briefing_{report_date}.html"
    if not path.exists():
        raise HTTPException(404, detail=f"report on {report_date} not found")
    return path.read_text(encoding="utf-8")


@app.get("/reports")
def list_reports() -> list[dict]:
    output_dir = Path(settings.OUTPUT_DIR)
    if not output_dir.exists():
        raise HTTPException(500, detail="Output directory does not exist")
    reports = []
    for html in sorted(output_dir.glob("financial_briefing_*.html")):
        date = html.stem.split("_")[-1]
        md = html.with_suffix(".md")
        reports.append({
            "date": date,
            "html": html.name,
            "markdown": md.name if md.exists() else None,
        })
    return reports
