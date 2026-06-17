"""eval/validate_testset.py — Ship E, step 3: validate the labeled test set.

Run this before declaring Ship E done. It enforces the schema from
doc/ship-e-testset.md so Ship F can load eval/testset.jsonl and trust it:

  - every line parses as JSON with the required keys
  - query_id values are unique
  - source in {llm, hand}; type in {in_domain, out_of_domain}
  - out_of_domain  <=>  relevant_article_ids == []
  - in_domain      =>   at least one relevant id
  - every relevant_article_id resolves to a real row in `articles`
  - row count is in the 50-100 target band (reported as a WARNING, not a hard
    failure, so the script stays useful while you build the set up over days)

Exit code 0 = all hard checks passed; 1 = at least one hard failure (or the file
is missing). Warnings alone do not fail the run.

Run:  python -m eval.validate_testset [--path PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from src.storage.database import Database


TESTSET_PATH = "eval/testset.jsonl"
REQUIRED_KEYS = {"query_id", "query", "relevant_article_ids", "source", "type", "notes"}
VALID_SOURCES = {"llm", "hand"}
VALID_TYPES = {"in_domain", "out_of_domain"}
MIN_ROWS, MAX_ROWS = 50, 100


def _load_rows(path: str, errors: List[str]) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((lineno, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON ({e})")
    return rows


def validate(path: str = TESTSET_PATH) -> bool:
    """Return True if all hard checks pass. Prints errors and warnings."""
    errors: List[str] = []
    warnings: List[str] = []

    if not os.path.exists(path):
        print(f"FAIL: no test set at {path}")
        return False

    rows = _load_rows(path, errors)

    # Collect every referenced article id once, then resolve them in a single query.
    referenced_ids: set = set()
    seen_query_ids: set = set()

    for lineno, obj in rows:
        missing = REQUIRED_KEYS - obj.keys()
        if missing:
            errors.append(f"line {lineno}: missing keys {sorted(missing)}")
            continue

        qid = obj["query_id"]
        if qid in seen_query_ids:
            errors.append(f"line {lineno}: duplicate query_id {qid!r}")
        seen_query_ids.add(qid)

        if obj["source"] not in VALID_SOURCES:
            errors.append(f"{qid}: source {obj['source']!r} not in {sorted(VALID_SOURCES)}")
        if obj["type"] not in VALID_TYPES:
            errors.append(f"{qid}: type {obj['type']!r} not in {sorted(VALID_TYPES)}")

        rel = obj["relevant_article_ids"]
        if not isinstance(rel, list) or not all(isinstance(i, int) for i in rel):
            errors.append(f"{qid}: relevant_article_ids must be a list of ints")
            continue

        if obj["type"] == "out_of_domain" and rel:
            errors.append(f"{qid}: out_of_domain must have an empty relevant set (got {rel})")
        if obj["type"] == "in_domain" and not rel:
            errors.append(f"{qid}: in_domain must have >= 1 relevant id")

        referenced_ids.update(rel)

    # Every referenced id must resolve to a real article.
    if referenced_ids:
        db = Database()
        existing = {a.id for a in db.get_articles_by_ids(list(referenced_ids))}
        missing_ids = sorted(referenced_ids - existing)
        if missing_ids:
            errors.append(f"relevant_article_ids not found in `articles`: {missing_ids}")

    # Count is a soft target while the set is being built.
    n = len(rows)
    if n < MIN_ROWS:
        warnings.append(f"only {n} rows — target is {MIN_ROWS}-{MAX_ROWS} (still building?)")
    elif n > MAX_ROWS:
        warnings.append(f"{n} rows — above the {MAX_ROWS} target")

    n_ood = sum(1 for _, o in rows if o.get("type") == "out_of_domain")
    if rows and n_ood == 0:
        warnings.append("no out_of_domain queries — add a few to test the abstention path")

    # Report.
    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"FAIL: {e}")

    if errors:
        print(f"\n{len(errors)} hard failure(s) across {n} row(s).")
        return False
    print(f"\nOK: {n} row(s) valid ({n_ood} out-of-domain). "
          f"{len(warnings)} warning(s).")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Ship E test set.")
    parser.add_argument("--path", type=str, default=TESTSET_PATH)
    args = parser.parse_args()
    sys.exit(0 if validate(args.path) else 1)


if __name__ == "__main__":
    main()
