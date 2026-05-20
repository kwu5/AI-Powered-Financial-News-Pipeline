# FinNews-RAG: Retrieval-Augmented Financial News Analysis with Evaluation

> A retrieval-augmented generation (RAG) system that ingests financial news, answers grounded questions with source citations, and—critically—**measures its own quality** through a custom evaluation harness covering retrieval precision, answer faithfulness, and latency/cost across configurations.

<!-- PLACEHOLDER: Add a one-line demo link or screenshot here once available -->
<!-- e.g. **Live demo:** https://... | **Demo video:** https://... -->

---

## Why this project exists

Most RAG demos stop at "it returns an answer." Production LLM systems live or die on whether that answer is *correct*, *grounded*, and *affordable*. This project treats evaluation as a first-class component: every change to the pipeline (embedding model, chunk size, retrieval depth, generation model) is measured against a labeled test set so improvements are evidence-based rather than vibes-based.

The domain is financial news because I want to learn investing as a beginner.

---

## What it does

1. **Ingests** financial news from 8–10 free RSS feeds (Reuters, CNBC, MarketWatch, Yahoo Finance, CoinDesk, Federal Reserve, SEC, and others), with the World News API free tier as a secondary fallback.
2. **Chunks and embeds** articles (~256-token chunks), storing chunk-level vectors in ChromaDB.
3. **Retrieves** the most relevant chunks for a user query and generates a **grounded, source-cited answer** using OpenAI `gpt-4o-mini`, with prompts that constrain the model to answer only from retrieved context.
4. **Evaluates** pipeline quality across multiple dimensions and configurations (see Evaluation below).
5. **Extracts structured signals**: entities/tickers, event types (earnings, M&A, leadership change, regulatory), and per-entity sentiment.

*Secondary feature (inherited from the original MVP): a scheduled **daily briefing report** that summarizes the day's financial news into a structured Markdown/HTML digest.*

---

## Architecture

```
[News source] --> [Ingestion + chunking] --> [Embedding model] --> [Vector store]
                                                                        |
User query --------------------------------> [Retriever (top-k)] <------+
                                                     |
                                                     v
                                          [Grounded prompt + LLM] --> [Cited answer]
                                                     |
                                                     v
                                          [Evaluation harness] --> [Metrics + config comparison]
```



---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Orchestration | None — direct OpenAI / ChromaDB / sentence-transformers calls (no LangChain/LlamaIndex) |
| Vector store | ChromaDB (persistent) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Generation LLM | OpenAI `gpt-4o-mini` |
| Evaluation | Hybrid — RAGAS (faithfulness, answer relevance) + a custom harness (retrieval precision/recall, latency/cost) |
| Interface | Streamlit |

---

## Evaluation (the core differentiator)

The evaluation harness scores the pipeline on a hand-labeled test set of 50–100 query/relevant-document pairs. Relevance is labeled at the article level; a retrieved chunk counts as relevant if its source article is labeled relevant.

**Metrics:**
- **Retrieval precision & recall** — are the retrieved chunks actually relevant to the query? Computed by the custom harness against the labeled set.
- **Faithfulness** — does the generated answer stay grounded in retrieved sources (vs. hallucinating)? Measured via **RAGAS faithfulness**.
- **Answer relevance** — does the answer address the question asked? Measured via **RAGAS answer relevance**.
- **Latency & cost** — per-query response time and token cost, tracked by the custom harness.

**Configuration comparison:** the harness runs across combinations of embedding model, chunk size, and retrieval depth (top-k), and reports the tradeoffs.

**Key findings:** <!-- PLACEHOLDER: fill in after running the multi-config comparison — e.g. "Reducing chunk size from 512 to 256 tokens improved retrieval precision from X to Y but increased index size by Z%." This section is what interviewers will read most closely. -->

<!-- PLACEHOLDER: insert a results table or chart image here -->

---

## Setup

```bash
git clone https://github.com/[YOUR-USERNAME]/[REPO-NAME].git
cd [REPO-NAME]
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env              # add your API keys
```

**Environment variables** (`.env`):

| Variable | Purpose |
|---|---|
| `WORLD_NEWS_API_KEY` | World News API key (secondary ingestion source) |
| `OPENAI_API_KEY` | OpenAI key for generation and RAGAS judging |
| `DATABASE_URL` | SQLite article store (default `sqlite:///./data/news.db`) |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory (default `./data/chroma`) |

**Never commit real API keys.** Use `.env` (gitignored) and provide a `.env.example` with placeholder values.

---

## Usage

```bash
python main.py --mode ingest    # fetch RSS + World News, chunk, embed, index into ChromaDB
streamlit run app.py            # launch the Q&A demo — ask grounded, source-cited questions
python evaluate.py              # run the evaluation harness; write metrics + config comparison
python main.py --mode run-once  # secondary: generate today's daily briefing report
```

---

## Roadmap

The detailed 3-day-ship plan lives in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). High level:

- [x] Ingestion + storage + embedding pipeline (inherited MVP)
- [ ] Chunking + chunk-level retrieval
- [ ] Grounded, source-cited Q&A
- [ ] Evaluation harness with retrieval + faithfulness metrics on a labeled set
- [ ] Multi-configuration comparison with written findings
- [ ] Streamlit demo interface
- [ ] (Stretch) Structured signal extraction (entities, events, sentiment)
- [ ] (Stretch) Backtest: does extracted sentiment correlate with short-term price movement?

---

## Notes & limitations

- RSS and World News API free tiers impose rate limits; ingestion is capped at roughly 30 articles per run.
- The evaluation test set is small (50–100 hand-labeled pairs) — metrics indicate direction, not statistical certainty.
- Full-text extraction (`trafilatura` / `newspaper3k`) fails on ~10% of articles; those are dropped rather than indexed partially.
- This is an open-source-standard learning project, not enterprise-grade financial infrastructure (Bloomberg / Refinitiv / Dow Jones operate at a different tier).

## License

MIT
