# Ship 1 — RSS feed expansion, config externalization, UTC normalization

**Date:** 2026-05-18
**Window:** 2026-05-18 → 05-20 (3-day ship cadence)
**Plan reference:** `IMPLEMENTATION_PLAN.md` Week 1 (partial) · `~/.claude/plans/refresh-your-memory-read-piped-penguin.md` Ship 1

## Goal

Replace the 3 hardcoded RSS feeds in `rss_reader.py` with 8–10 feeds loaded from a YAML config, and normalize every article timestamp (RSS + World News API) to TZ-aware UTC ISO 8601 at the ingestion boundary.

## Why first

- Lowest risk Week 1 task — no new libraries beyond `pyyaml`/`python-dateutil`, no schema changes.
- Source diversity is the lever that everything downstream (dedup quality, clustering, hallucination guard) depends on. Doubling sources before touching the dedup pipeline maximizes the value of Ship 4–7.
- Externalized feed list gives Ship 3 (per-source health) and Ship 11 (stale-feed alerts) a place to read from.

## Files changed

| File | Change |
|---|---|
| `config/feeds.yaml` | **new** — 10 feeds with `{name, url, category}` |
| `src/config.py` | added `FEEDS_CONFIG_PATH: str = "./config/feeds.yaml"` |
| `src/ingestion/rss_reader.py` | rewritten — YAML loader, dateutil-based `_parse_date` returning TZ-aware UTC, `fetched_at` uses `datetime.now(timezone.utc)`, `source` from YAML name, `category` propagated |
| `src/ingestion/world_news_api.py` | added `_to_utc_iso` helper and `_normalize` to match RSS dict shape; `fetch_financial_news` returns normalized dicts with UTC timestamps |
| `tests/test_ingestion.py` | expanded from 10 → 24 tests; covers YAML loader, UTC tz-conversion for both ingest paths, MagicMock-defensive `_parse_date` |
| `requirements.txt` | added `pyyaml >= 6.0, <7` and `python-dateutil >= 2.8, <3` |

## Feed list

Validated each candidate URL with `feedparser.parse(url)` before adding. Dropped Reuters Business (`feeds.reuters.com/reuters/businessNews` returns 0 entries — discontinued) and the duplicate CNBC Markets feed to avoid source-dominance.

| Source | Category | Entries observed |
|---|---|---|
| CNBC Top News | general | 30 |
| Yahoo Finance | general | 48 |
| MarketWatch Top Stories | general | 10 |
| NYT Business | general | 50 |
| BBC Business | general | 52 |
| Investing.com News | general | 10 |
| Seeking Alpha Market Currents | general | 7 |
| CoinDesk | crypto | 25 |
| Federal Reserve Press Releases | fed | 20 |
| SEC Press Releases | regulatory | 25 |

Federal Reserve feed reports `bozo=True` (minor XML compliance warning) but parses 20 entries cleanly — kept.

## Key implementation notes

### `_parse_date` (RSS)

Order of attempts:
1. `entry.get("published")` or `entry.get("updated")` — parse with `dateutil.parser`. If parsed naive, assume UTC. Convert to UTC.
2. Fallback to `entry.published_parsed` / `entry.updated_parsed` struct_time. Treat as UTC.
3. Fallback to `datetime.now(timezone.utc)`.

Each step is wrapped in `try/except (TypeError, ValueError)` so a malformed field never raises out of `_parse_date` (which would abort the whole feed under the existing outer `try`).

### `_to_utc_iso` (WNA)

Same logic, isolated for unit testing. Returns `None` on empty/garbage input so the caller can fall through to `fetched_at`.

### WNA shape alignment

Previously `WorldNewsAPIClient.fetch_financial_news` returned the raw API dicts (`title`, `text`, `publish_date`, …). The pipeline treated these interchangeably with RSS dicts, which only worked because downstream code used `.get` with defaults. WNA articles now produce the same keys as RSS articles: `title`, `description`, `content`, `url`, `source`, `category`, `published_at`, `fetched_at`. `category` is hardcoded to `"general"` for WNA — Ship 3 can revisit if needed.

### `feed_urls` setter on `RSSReader`

Kept as a property setter so existing test code (`reader.feed_urls = [url]`) keeps working. Removable in Ship 3 once health tracking forces a `Feed` dataclass.

## Verification

- `pytest`: **47/47 passing** (24 ingestion, 13 processing, 10 summarization).
- Live smoke test (`python -c "from src.ingestion.rss_reader import RSSReader; ..."`): **277 articles from all 10 sources**, all `published_at` and `fetched_at` are TZ-aware UTC.
- Spot-checked timezone conversion in tests:
  - `"2026-04-20T11:00:00+05:00"` → `"2026-04-20T06:00:00+00:00"` (WNA path).
  - `"2026-04-20T14:30:00+05:00"` → `"2026-04-20T09:30:00+00:00"` (RSS path).

## Open questions for review

1. Final feed list — any swaps? Reuters substitute, FT/WSJ excluded as planned.
2. `category` field — keep, or drop until Ship 3 actually uses it?
3. `feed_urls` property setter — keep (compat) or remove (cleaner) ahead of Ship 3?
4. WNA `description` derived from first 300 chars of `text` when API summary missing — acceptable, or change?

## Out of scope (deferred to later ships)

- Full-text extraction via `trafilatura` / `newspaper3k` / `readability` (Ship 2).
- Per-source health metrics + WNA demotion to fallback (Ship 3).
- `canonical_url` / `content_hash` / `cluster_id` columns (Ship 4).
- Any dedup pipeline changes (Ships 5–7).
