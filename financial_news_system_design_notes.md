# Financial News Summarization System — Design Notes & Recommendations

> **Purpose of this document:** Reference notes for improving an existing AI-powered financial news summarization system. Use these recommendations to compare against the current implementation and identify upgrade priorities.
>
> **Project stack (existing):** Python 3.11+, SQLite, ChromaDB, sentence-transformers, OpenAI GPT-4o-mini, FastAPI, APScheduler, Docker.
>
> **Status:** MVP built. Focus is now on architectural improvements, not new features.

---

## Table of Contents

1. [Data Source Strategy](#1-data-source-strategy)
2. [Rolling-Window Storage Architecture](#2-rolling-window-storage-architecture)
3. [Deduplication Pipeline](#3-deduplication-pipeline)
4. [Storage Requirements](#4-storage-requirements)
5. [Hardware Requirements](#5-hardware-requirements)
6. [Known Challenges](#6-known-challenges)
7. [Reference Projects](#7-reference-projects)
8. [Comparison Checklist](#8-comparison-checklist)

---

## 1. Data Source Strategy

### Decision: RSS-first, News API secondary

**Priority order:**
1. **Primary — RSS feeds** (10-15 sources)
2. **Secondary — Financial news API** (Finnhub or Marketaux) for ticker tagging and enrichment
3. **Tertiary (future) — Targeted scraping** for high-value sources without RSS (e.g., Fed FOMC statements)

### Why RSS is primary

- **Freshness:** Publishers push to RSS as soon as articles are published. Better for daily briefings.
- **No rate limits, no keys, no quotas.** NewsAPI free tier has a 24-hour delay that breaks "daily briefing" use cases.
- **Source diversity is the quality lever for dedup.** Adding a new RSS source = appending a URL to a config list.
- **Stability:** RSS is a 25-year-old format that rarely breaks. API providers change pricing or deprecate endpoints.
- **Stronger technical narrative:** "I ingest 15 RSS feeds and dedupe with vector embeddings" is more interesting than "I call one API."

### Why APIs still earn a place (secondary)

- RSS gives summaries, not full article bodies — you need `trafilatura` / `newspaper3k` / `readability-lxml` to extract full text.
- RSS has no structured metadata (no ticker tagging, no sentiment scores).
- RSS has no backfill — if your scheduler misses a day, you can't reconstruct it.

### Recommended RSS source list

- **General financial:** Reuters Business, CNBC, MarketWatch, Yahoo Finance, FT, WSJ
- **Specialized:** CoinDesk (crypto), Federal Reserve press releases
- **Aggregators:** Seeking Alpha, Bloomberg (limited feeds available)

### Recommended news APIs (pick one as secondary)

| API | Free tier | Best for |
|---|---|---|
| **Finnhub** | 60 calls/min | News + market data combined |
| **Marketaux** | 100 req/day | Built-in entity tagging + sentiment |
| **Alpha Vantage** | 25 req/day | Pre-computed sentiment scores |
| **NewsAPI** | 100/day, 24hr delay | ❌ Skip — delay breaks daily briefings |

---

## 2. Rolling-Window Storage Architecture

### Core principle

RSS feeds are ephemeral (publishers expose only the latest 10-50 items). Persist what you fetch, or lose it. **Do NOT use strict "delete oldest when newest arrives"** — use tiered retention instead.

### Three-tier retention model

| Tier | Age | Contents | Per-article size |
|---|---|---|---|
| **Hot** | 0-90 days | Full text + embeddings + metadata | ~10 KB |
| **Warm** | 90 days - 1 year | Summary + embedding + metadata (no full text) | ~2 KB |
| **Cold** | 1+ years | Metadata + daily summary reference only | ~0.5 KB |

### Schema sketch

```sql
articles
  id, url, canonical_url, content_hash,
  title, source, published_at, fetched_at,
  raw_text (nullable after 90 days),
  cleaned_text (nullable after 90 days),
  status: 'hot' | 'warm' | 'cold',
  cluster_id  -- for event clustering

daily_summaries
  date, summary_markdown, source_article_ids, generated_at
  -- retained FOREVER; this is the output product

embeddings (in ChromaDB)
  -- aligned with articles.id
  -- pruned for cold-tier articles
```

### Cleanup job design

- **Decouple ingestion from cleanup.** Run as separate APScheduler jobs.
- **Cleanup cadence:** weekly is fine, not daily.
- **Cleanup logic:**
  - Articles > 90 days → null `raw_text` and `cleaned_text`, set `status='warm'`
  - Articles > 1 year → drop embeddings from ChromaDB, set `status='cold'`
  - Articles > 2 years → delete entirely (configurable)
  - **Never delete `daily_summaries`** — these are the project's output product
- Log every cleanup operation for auditing

### What this unlocks

- **Smarter dedup** — compare against last 7-30 days, not just today's batch
- **Real RAG** — vector store becomes a meaningful retrieval corpus
- **Theme tracking over time** — "tech sector mentions up 40% this week"
- **Weekly/monthly briefings** — aggregate stored dailies into longer reports
- **Backfill recovery** — if scheduler misses a day, data is still there

---

## 3. Deduplication Pipeline

### The four types of "duplicate"

Most dedup implementations fail because they treat all duplicates the same way. There are actually four distinct categories:

| Type | Description | Example | Action |
|---|---|---|---|
| **Type 1** | Exact duplicates with different URLs | Tracking params, AMP versions | Drop |
| **Type 2** | Near-exact (syndicated wire stories) | AP story republished by 20 outlets | Flag + keep one |
| **Type 3** | Same event, different reporting | CNBC + Bloomberg + WSJ on Fed decision | **Cluster** (don't drop) |
| **Type 4** | Story evolution over time | "Fed signals cut" Mon → "Fed cuts" Fri | Keep both, link |

### Cascading filter architecture

Cheap exact checks first, expensive semantic checks last.

```
RSS fetch
  ↓
[Stage 1] URL canonicalization → drop if duplicate canonical URL
  ↓
Article extraction (trafilatura → newspaper3k → readability fallback)
  ↓
[Stage 2] Content hash → drop if duplicate hash
  ↓
[Stage 3] MinHash check vs last 48 hours
  ├─ Jaccard > 0.8 → mark as duplicate of X, store but flag
  └─ Otherwise → continue
  ↓
Embedding generation
  ↓
[Stage 4] Embedding cluster vs last 48 hours
  ├─ Cosine > 0.75 → assign to existing cluster
  └─ Otherwise → start new cluster
  ↓
Store article + cluster_id
  ↓
Daily summarization: ONE SUMMARY PER CLUSTER, not per article
```

### Stage 1: URL canonicalization

```python
def canonicalize_url(url):
    TRACKING_PARAMS = {'utm_source', 'utm_medium', 'utm_campaign',
                       'utm_term', 'utm_content', 'fbclid', 'gclid',
                       'ref', 'ref_src', 'mc_cid', 'mc_eid'}
    parsed = urlparse(url)
    query = {k: v for k, v in parse_qs(parsed.query).items()
             if k not in TRACKING_PARAMS}
    path = parsed.path.replace('/amp/', '/').rstrip('/amp')
    host = parsed.netloc.lower().replace('www.', '')
    return urlunparse((parsed.scheme, host, path, '', urlencode(query), ''))
```

Store as `canonical_url` with a UNIQUE index. Eliminates 10-20% of incoming articles.

### Stage 2: Content hashing

```python
def content_hash(text):
    normalized = re.sub(r'\s+', ' ', text.lower())
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()
```

UNIQUE index on this too. Catches different URLs with identical content.

### Stage 3: MinHash for near-duplicates

**This is the stage most projects skip — adding it gives the biggest quality jump.**

Use the `datasketch` library:

```python
from datasketch import MinHash, MinHashLSH

def make_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - 4):
        shingle = ' '.join(words[i:i+5])
        m.update(shingle.encode('utf8'))
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)
# Insert all hot-tier articles at startup
# Query: lsh.query(new_minhash) returns near-duplicate IDs
```

**Why MinHash beats embeddings for Type 2:**
- Designed for lexical overlap (exactly what wire stories have)
- Orders of magnitude faster than pairwise embedding comparison
- LSH gives sub-linear lookup
- Jaccard threshold has clear semantic meaning

### Stage 4: Embedding-based clustering

**Key paradigm shift: this is NOT for dedup-and-discard. It's for clustering same-event articles for richer summarization.**

```python
def find_event_cluster(article, recent_articles, threshold=0.75):
    new_emb = embed(article.text)
    cluster = [article]
    for past in recent_articles:
        sim = cosine_similarity(new_emb, past.embedding)
        if sim > threshold:
            cluster.append(past)
    return cluster
```

Then send the **cluster** to the LLM with a prompt like:

> "These N articles cover the same event from different sources. Synthesize a single summary capturing key facts. Note any disagreements between sources."

### Threshold tuning methodology

Don't guess — measure on your own data:

1. Run pipeline for 1-2 weeks in **logging-only mode** (no dedup decisions enforced)
2. Sample 100 pairs across similarity ranges (0.5-0.7, 0.7-0.85, 0.85-1.0)
3. Manually label: "true duplicate" / "same event" / "different event"
4. Plot precision/recall curves per stage

**Typical post-tuning thresholds for financial news:**
- MinHash Jaccard: **0.75-0.85** for near-duplicates
- Embedding cosine: **0.78-0.85** for same event
- Embedding cosine: **0.60-0.75** for same broad topic

### Time-windowing (non-negotiable)

- Type 2/3 dedup: **48-hour window**
- Slow-developing stories (earnings season): **7-day window**
- Never compare against the entire corpus — "Fed cuts rates" in March vs October are different events with high similarity

---

## 4. Storage Requirements

### Per-article breakdown

| Component | Size |
|---|---|
| Cleaned article text | 3-8 KB |
| Title + metadata | 0.5-1 KB |
| Embedding (MiniLM-L6-v2, 384-dim) | 1.5 KB |
| MinHash signature (128 perm) | 1 KB |
| SQLite index overhead | ~30% of data |
| **Total per article (hot tier)** | **~8-12 KB** |

### Annual projections

| Volume | Daily | 90-day hot | 1 year (no tiering) |
|---|---|---|---|
| Conservative (30 articles/day) | 300 KB | 27 MB | 110 MB |
| Realistic (100 articles/day) | 1 MB | 90 MB | 365 MB |
| Aggressive (300 articles/day) | 3 MB | 270 MB | 1.1 GB |

### With tiered retention (2 years operation, 100 articles/day)

- Hot tier: ~90 MB
- Warm tier: ~55 MB
- Cold tier: ~18 MB
- Daily summaries (kept forever): ~50 KB
- **Total: ~165 MB**

### ChromaDB sizing

- ~2-3x raw vector size due to HNSW index overhead
- 9,000 vectors (100 articles/day × 90 days) = **~35-45 MB on disk**
- ChromaDB only slows past 100K+ vectors (not a concern at this scale)

### Recommended disk allocation

- **Minimum to start:** 1 GB
- **Comfortable production:** 10 GB
- **VPS spec:** 2 vCPU, 4 GB RAM, 20-40 GB SSD (~$5-12/month)

### Storage pitfalls to avoid

1. ❌ Storing raw HTML in the database (10-20x larger than cleaned text)
2. ❌ Logging full article bodies in error cases (truncate strings to ~200 chars)
3. ❌ Keeping failed-extraction garbage in main table
4. ❌ Running multiple embedding model variants in parallel
5. ❌ Caching image/media URLs from RSS enclosures unless explicitly needed

---

## 5. Hardware Requirements

### Bottom line: no special hardware needed

A modern laptop or $5-10/month VPS handles the entire workload.

### Embedding model footprint

- `all-MiniLM-L6-v2`: ~90 MB RAM
- `all-mpnet-base-v2`: ~420 MB RAM
- `paraphrase-multilingual-MiniLM-L12-v2`: ~438 MB RAM

CPU inference is fine at this scale (embedding 30-300 articles/day in batches takes seconds).

### Container sizing warning

Avoid undersized containers. A 1 vCPU / 1 GB container can be **15-20x slower** than a normal laptop for the same workload. Minimum recommended:

- **2 vCPU**
- **4 GB RAM**
- **20 GB SSD**

### Optional speedups (if ever needed)

- **ONNX backend** for sentence-transformers: 2-3x faster on CPU
- **LightEmbed** library: minimal-dependency ONNX runtime alternative
- **Model quantization** (int8): smaller memory footprint, minimal accuracy loss

### No GPU needed

Don't pay for GPU instances. Embedding workload is too small to benefit, GPU would sit idle 99% of the time.

### LLM cost estimate

- ~30 articles/day × 2K tokens avg × GPT-4o-mini pricing ≈ **a few cents per day**
- Even with generous summarization budgets: **under $5/month**

---

## 6. Known Challenges

Real issues that bite every project in this category:

### 6.1 RSS feed brittleness
- Feeds break, get rate-limited, change URLs, or silently drop articles
- **Fix:** per-source health monitoring, exponential backoff, alerts on feeds stale > 24 hours

### 6.2 Article extraction failures
- ~5-15% of pages fail extraction (paywalls, JS rendering, anti-bot, weird layouts)
- **Fix:** fallback chain (trafilatura → newspaper3k → readability-lxml), log success rate per source

### 6.3 Near-duplicate detection complexity
- Wire stories republished with minor edits
- Cosine similarity alone produces false positives AND false negatives
- **Fix:** hybrid pipeline (URL canon + content hash + MinHash + embeddings) — see Section 3

### 6.4 Source bias and coverage gaps
- 10 US-centric feeds give a US-centric view of "global markets"
- **Fix:** intentional source diversity in feed list

### 6.5 LLM output inconsistency
- LLMs produce different outputs across runs
- **Fix:** strict structured output (JSON schema, pydantic validation), regenerate-on-validation-failure loop

### 6.6 Hallucinations in summaries (CRITICAL for finance)
- LLM may invent tickers, swap numbers, misattribute quotes
- **Fix:** post-generation validation — verify entities mentioned in summary actually appear in source articles
- Secondary news API with ticker tagging earns its keep here

### 6.7 Temporal drift in dedup
- "Fed signals rate cut" today vs same phrase last month are different events
- **Fix:** time-window all dedup checks (48 hours typical)

### 6.8 Vector DB scaling
- ChromaDB sluggish past 100K-500K vectors on commodity hardware
- **Fix:** at your scale (2,700 vectors for 90-day window), non-issue. Plan ahead if extending window.

### 6.9 Operational nuisances
- Time zones (mixed UTC/local in publication times)
- HTML entity decoding
- Character encoding issues
- Feeds that lie about their `pubDate`
- **Fix:** normalize everything to UTC ISO 8601, use `dateutil` for parsing

---

## 7. Reference Projects

Study these before/during improvement work:

### Closest to your project (study deeply)

- **finaldie/auto-news** — Personal news aggregator with multi-source ingestion + LLM via LangChain. Best reference for LLM orchestration patterns.
- **dhivyeshrk/Retrieval-Augmented-Generation-for-news** — Same ChromaDB + RSS + sentence-transformers stack as yours. Direct architectural reference.

### Production-grade RSS infrastructure

- **FreshRSS** — Battle-tested self-hosted RSS aggregator. Study its schema and feed-polling logic.
- **Tiny Tiny RSS** — Another mature aggregator. Good reference for feed health monitoring.

### Academic / large-scale reference

- **GDELT Project** — Open global news database since 1979. Updates every 15 minutes. Direct academic validation of "news → financial signal" pattern (Italian sovereign bond market forecasting research).

### Recent innovation worth noting

- **LLM-based RSS filtering** (Hacker News, March 2025) — Open-source reader using LLMs to tag and score articles, filtering out 80% noise from 1000+ daily articles. Pattern applicable to your project.

---

## 8. Comparison Checklist

Use this checklist to compare against the current project implementation.

### Data sources
- [ ] RSS feeds configured as primary source (10+ feeds)
- [ ] Per-source health monitoring in place
- [ ] Secondary news API integrated for ticker tagging / enrichment
- [ ] Source diversity audit (not 100% US-centric)

### Storage architecture
- [ ] Articles stored with both `url` and `canonical_url`
- [ ] `content_hash` column with UNIQUE index
- [ ] `cluster_id` column on articles table
- [ ] `daily_summaries` table separate from `articles` (retained forever)
- [ ] Tiered retention implemented (hot/warm/cold)
- [ ] Cleanup job runs separately from ingestion (weekly cadence)
- [ ] `raw_text` nulled out after 90 days
- [ ] ChromaDB pruned in same cleanup job

### Deduplication pipeline
- [ ] **Stage 1:** URL canonicalization with tracking-param stripping
- [ ] **Stage 2:** Content hash check before embedding
- [ ] **Stage 3:** MinHash + LSH for near-duplicates (datasketch library)
- [ ] **Stage 4:** Embedding clustering for same-event grouping (NOT dedup)
- [ ] Time-windowed comparisons (48 hours default)
- [ ] Thresholds tuned on real data, not guessed
- [ ] Syndicated articles flagged but kept (audit trail)

### LLM integration
- [ ] Prompts send clusters of articles, not individual articles
- [ ] Structured output enforced (JSON schema or pydantic)
- [ ] Validation loop on malformed responses
- [ ] Hallucination check: entities in summary verified against source articles
- [ ] Source diversity surfaced in final report

### Extraction robustness
- [ ] Fallback chain (trafilatura → newspaper3k → readability)
- [ ] Per-source extraction success rate logged
- [ ] Failed extractions don't pollute main article table

### Operational
- [ ] All timestamps stored as UTC ISO 8601
- [ ] Feeds with stale data > 24 hours surfaced in logs/report
- [ ] Logging truncates large strings (no full article bodies in logs)
- [ ] SQLite in WAL mode for concurrent reads during ingestion
- [ ] Weekly VACUUM on SQLite

### Hardware / deployment
- [ ] Deployed on minimum 2 vCPU / 4 GB RAM environment
- [ ] No GPU dependency
- [ ] Disk allocation ≥ 10 GB
- [ ] Considered ONNX backend for embedding speedup (if relevant)

---

## Quick prioritization for improvement work

If you only have time to do three things, do these in order:

1. **Add the MinHash dedup stage (Section 3, Stage 3).** Biggest quality jump per hour of work. Most projects skip this.
2. **Shift from "embedding dedup" to "embedding clustering" for LLM prompts (Section 3, Stage 4).** Architectural change that doubles summary quality.
3. **Add hallucination post-validation (Section 6.6).** Critical for finance — prevents fabricated tickers/numbers from reaching the output.

Everything else is incremental polish on top of these three.

---

*End of document.*
