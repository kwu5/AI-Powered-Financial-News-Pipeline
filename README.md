# AI-Powered Daily Financial News Summarizer

Automatically collects daily financial news from multiple sources and generates structured summaries of key market events using GPT-4o-mini.

## Architecture

```
News Sources (World News API + RSS)
        |
        v
  Text Cleaning (spaCy NER, regex)
        |
        v
  Deduplication (sentence-transformers cosine similarity)
        |
        v
  Storage (SQLite + ChromaDB)
        |
        v
  Summarization (OpenAI GPT-4o-mini)
        |
        v
  Reports (Markdown + HTML)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data Ingestion | World News API + RSS (CNBC, Yahoo Finance, Reuters) |
| Database | SQLite via SQLAlchemy |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | OpenAI GPT-4o-mini |
| NLP | spaCy (en_core_web_sm) |
| Web Framework | FastAPI |
| Scheduler | APScheduler |
| Output | Markdown + HTML (Jinja2) |

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd Daily-Financial-News-Summarization
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
NEWS_API_KEY=your_world_news_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run

```bash
# Run the full pipeline once (good for first-time test)
python main.py --mode run-once

# Start API server with daily scheduler
python main.py --mode api

# Start scheduler only (no HTTP server)
python main.py --mode scheduler
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/generate` | Generate today's report |
| GET | `/report/{date}` | Get report by date (HTML) |
| GET | `/reports` | List all available reports |

Interactive docs available at `http://localhost:8000/docs` when the API is running.

### Examples

```bash
# Generate report
curl -X POST http://localhost:8000/generate

# Get today's report
curl http://localhost:8000/report/2026-04-20

# List all reports
curl http://localhost:8000/reports

# Health check
curl http://localhost:8000/health
```

## Docker

```bash
docker-compose up --build
```

This starts the API server with the daily scheduler on port 8000. SQLite database and generated reports are persisted via volume mounts to `data/` and `output/`.

## Running Tests

```bash
pytest
pytest -v              # verbose output
pytest tests/test_ingestion.py   # single file
```

## Project Structure

```
├── src/
│   ├── config.py                 # Pydantic settings (.env loading)
│   ├── pipeline.py               # Central 9-step pipeline orchestration
│   ├── ingestion/
│   │   ├── world_news_api.py     # World News API client
│   │   └── rss_reader.py         # RSS feed parser
│   ├── processing/
│   │   ├── cleaner.py            # Text cleaning + spaCy NER
│   │   ├── embeddings.py         # Sentence-transformer embeddings
│   │   └── deduplicator.py       # Cosine similarity deduplication
│   ├── storage/
│   │   ├── database.py           # SQLAlchemy models (Article, DailyReport)
│   │   └── vector_store.py       # ChromaDB integration
│   ├── summarization/
│   │   ├── llm_client.py         # OpenAI API wrapper
│   │   └── report_generator.py   # Markdown/HTML report export
│   ├── api/
│   │   └── main.py               # FastAPI application
│   └── scheduler/
│       └── jobs.py               # APScheduler daily job
├── templates/
│   └── report.html               # Jinja2 HTML template
├── tests/
│   ├── test_ingestion.py
│   ├── test_processing.py
│   └── test_summarization.py
├── data/                         # SQLite DB + ChromaDB (gitignored)
├── output/                       # Generated reports (gitignored)
├── main.py                       # CLI entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Configuration

All settings are managed via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWS_API_KEY` | *required* | World News API key |
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `DATABASE_URL` | `sqlite:///./data/news.db` | SQLAlchemy database URL |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for summarization |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `MAX_ARTICLES_PER_DAY` | `30` | Max articles to process |
| `SIMILARITY_THRESHOLD` | `0.85` | Deduplication cosine threshold |
| `DAILY_RUN_HOUR` | `18` | Hour (UTC) for scheduled daily run |
| `OUTPUT_DIR` | `./output` | Report output directory |
