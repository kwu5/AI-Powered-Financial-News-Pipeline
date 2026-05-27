"""Inspect the rows added during the last pipeline run.

Pairs with smoke_ship_b.py — after that reports `N new rows persisted`, run
this to see whether those N rows are genuinely new articles or look like
duplicates of older rows that slipped through with different canonical_url /
content_hash values.

Run:
    .venv/Scripts/python.exe -m tests.smoke_ship_b_inspect
"""

from sqlalchemy import desc

from src.storage.database import Database, Article


def main(n: int = 20) -> None:
    db = Database()
    with db.SessionLocal() as s:
        total = s.query(Article).count()
        rows = (
            s.query(Article)
            .order_by(desc(Article.id))
            .limit(n)
            .all()
        )
        print(f"\nTotal rows in DB: {total}")
        print(f"Showing last {len(rows)} by id (most recent first):\n")
        for r in rows:
            print(f"  id={r.id}  source={r.source!r}")
            print(f"    title:         {r.title[:90]!r}")
            print(f"    url:           {r.url}")
            print(f"    canonical_url: {r.canonical_url}")
            print(f"    content_hash:  {r.content_hash[:16]}...")
            print(f"    fetched_at:    {r.fetched_at}")

            # Look for older rows from the same source with a similar title.
            # "Similar" = first 6 words of the title appear in another title.
            title_prefix = " ".join(r.title.split()[:6]).lower()
            if len(title_prefix) >= 12:
                similar = (
                    s.query(Article)
                    .filter(Article.id != r.id)
                    .filter(Article.source == r.source)
                    .filter(Article.title.ilike(f"%{title_prefix}%"))
                    .first()
                )
                if similar:
                    print(f"    ! possible older twin: id={similar.id}")
                    print(f"      twin url:           {similar.url}")
                    print(f"      twin canonical_url: {similar.canonical_url}")
                    print(f"      twin content_hash:  {similar.content_hash[:16]}...")
            print()


if __name__ == "__main__":
    main()
