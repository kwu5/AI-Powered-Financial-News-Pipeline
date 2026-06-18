"""src/evaluation/testset.py — Ship F: load the Ship E ground truth.

Thin typed loader over `eval/testset.jsonl` (schema in doc/ship-e-testset.md). The
harness consumes `TestQuery` objects. `eval/validate_testset.py` is the
authoritative validator — run it first; this loader does only the light parsing the
harness needs and trusts an already-validated file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Set

TESTSET_PATH = "eval/testset.jsonl"


@dataclass(frozen=True)
class TestQuery:
    __test__ = False  # not a pytest test class despite the "Test" prefix

    query_id: str
    query: str
    relevant_article_ids: Set[int]
    source: str
    type: str
    notes: str = ""

    @property
    def is_out_of_domain(self) -> bool:
        return self.type == "out_of_domain"


def load_testset(path: str = TESTSET_PATH) -> List[TestQuery]:
    """Parse the JSONL test set into TestQuery rows (skips blank lines)."""
    rows: List[TestQuery] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} invalid JSON: {e}") from e
            rows.append(
                TestQuery(
                    query_id=o["query_id"],
                    query=o["query"],
                    relevant_article_ids=set(o["relevant_article_ids"]),
                    source=o.get("source", "llm"),
                    type=o.get("type", "in_domain"),
                    notes=o.get("notes", ""),
                )
            )
    return rows
