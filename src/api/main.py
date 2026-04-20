from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel


from src.pipeline import run_pipeline, settings


# ---------- Pydantic request/response models ----------
class GenerateRequest(BaseModel):
    force_refresh: bool = False

class ReportResponse(BaseModel):
    date: str
    content: str
    article_count: int
    generated_at: datetime


# ---------- App ----------
app = FastAPI(title="Daily Financial News Summarizer", version="0.1.0")


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
        summary, count = run_pipeline()
        now = datetime.now()
        return ReportResponse(
            date=now.strftime("%Y-%m-%d"),
            content=summary,
            article_count=count,
            generated_at=now,
        )
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
