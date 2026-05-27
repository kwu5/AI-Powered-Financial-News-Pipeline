"""Ship B smoke test — confirm cross-run dedup blocks re-saves of Run-1 articles.

This script ONLY runs the verification half (Run 2). It assumes Run 1 has
already populated the DB. If the DB is empty it bails out with a warning.

Prerequisites — do these once before running this script:
    Remove-Item .\\data\\news.db                                  # optional: start clean
    python -c "from src.storage.database import Database; Database()"   # rebuild schema
    python -m src.pipeline                                       # Run 1: populate DB

Then run the smoke (Run 2 + verify):
    python -m tests.smoke_ship_b

What this script does:
  1. Snapshots the current set of article ids (the "Run 1 baseline").
  2. Runs the pipeline a second time on the same day.
  3. For every row added during Run 2, looks for an OLDER row from the same
     source whose title starts with the same first 6 words. Any such pair is a
     "twin" and means the cross-run dedup failed.
  4. PASS if no twins are flagged. New rows without twins are fine — RSS feeds
     publish continuously, so a few genuinely fresh articles between Run 1 and
     Run 2 is expected and healthy.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.storage.database import Database, Article
from src.pipeline import run_pipeline


def _title_prefix(title: str, n_words: int = 6) -> str:
    return " ".join(title.split()[:n_words]).lower()


def main() -> None:
    db = Database()

    with db.SessionLocal() as s:
        baseline_ids = {row.id for row in s.query(Article.id).all()}
    before = len(baseline_ids)
    print(f"\nDB row count BEFORE run 2: {before}")
    if before == 0:
        print("WARNING: DB is empty — run the pipeline once first, then re-run this script.")
        return

    print("\n--- Running pipeline (run 2) ---\n")
    _, pipeline_count = run_pipeline()

    with db.SessionLocal() as s:
        new_rows = (
            s.query(Article)
            .filter(~Article.id.in_(baseline_ids))
            .all()
        )
        after = s.query(Article).count()

        twins = []
        for r in new_rows:
            prefix = _title_prefix(r.title)
            if len(prefix) < 12:
                continue
            older_twin = (
                s.query(Article)
                .filter(Article.id.in_(baseline_ids))
                .filter(Article.source == r.source)
                .filter(Article.title.ilike(f"%{prefix}%"))
                .first()
            )
            if older_twin:
                twins.append((r, older_twin))

    delta = after - before
    print(f"\n=== Run 2 result ===")
    print(f"Pipeline reported in-batch survivors: {pipeline_count}")
    print(f"DB row count BEFORE: {before}")
    print(f"DB row count AFTER:  {after}")
    print(f"New rows persisted:  {delta}")
    print(f"Twins flagged:       {len(twins)}")

    if twins:
        print("\nFAIL — these new rows look like duplicates of older rows:\n")
        for new_row, older in twins:
            print(f"  source={new_row.source!r}")
            print(f"    NEW id={new_row.id}: {new_row.title[:90]!r}")
            print(f"        url={new_row.url}")
            print(f"        canonical_url={new_row.canonical_url}")
            print(f"        content_hash={new_row.content_hash[:16]}...")
            print(f"    OLD id={older.id}: {older.title[:90]!r}")
            print(f"        url={older.url}")
            print(f"        canonical_url={older.canonical_url}")
            print(f"        content_hash={older.content_hash[:16]}...")
            print()
    else:
        print("\nPASS — every new row appears to be a genuinely fresh article.")


if __name__ == "__main__":
    main()
