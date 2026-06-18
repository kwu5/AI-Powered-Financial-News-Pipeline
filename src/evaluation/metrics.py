"""src/evaluation/metrics.py — Ship F: pure retrieval metrics.

No I/O, no retriever/DB imports — every function is a deterministic transform of
(ranked article ids, relevant id set), so they unit-test in isolation. The harness
does the retrieving, timing, and chunk->article dedup; it hands these functions a
ranked list of UNIQUE article ids (best-ranked first) and the query's relevant set.

Relevance is binary and article-level (Ship E labels), so we report
precision / recall / hit-rate / reciprocal-rank rather than graded nDCG.
"""

from __future__ import annotations

from typing import Sequence, Set


def _relevant_hits(ranked: Sequence[int], relevant: Set[int]) -> int:
    return sum(1 for a in ranked if a in relevant)


def precision(ranked: Sequence[int], relevant: Set[int]) -> float:
    """Fraction of the retrieved ARTICLES that are relevant.

    Denominator is the number of distinct articles actually retrieved
    (`len(ranked)`), not the chunk depth k: chunks dedup to a variable number of
    articles, so dividing by k would conflate "few chunks" with "wrong articles".
    Empty retrieval -> 0.0.
    """
    if not ranked:
        return 0.0
    return _relevant_hits(ranked, relevant) / len(ranked)


def recall(ranked: Sequence[int], relevant: Set[int]) -> float:
    """Fraction of the relevant articles that were retrieved. Empty relevant -> 0.0."""
    if not relevant:
        return 0.0
    return _relevant_hits(ranked, relevant) / len(relevant)


def hit_rate(ranked: Sequence[int], relevant: Set[int]) -> float:
    """1.0 if any relevant article was retrieved, else 0.0 (a.k.a. success@k).

    Most Ship E queries have a single relevant article, so this is the
    blunt-but-honest headline: did the right article show up at all?
    """
    return 1.0 if any(a in relevant for a in ranked) else 0.0


def reciprocal_rank(ranked: Sequence[int], relevant: Set[int]) -> float:
    """1/rank of the first relevant article (1-based); 0.0 if none present."""
    for i, a in enumerate(ranked, start=1):
        if a in relevant:
            return 1.0 / i
    return 0.0


def abstention_correct(answered_from_context: bool, is_out_of_domain: bool) -> bool:
    """Did the system make the right answer/abstain call?

    Out-of-domain  -> correct iff it abstained (`answered_from_context` is False).
    In-domain      -> correct iff it answered (`answered_from_context` is True).
    The harness aggregates this over OOD rows for the headline abstention number;
    the in-domain direction is available as a diagnostic.
    """
    if is_out_of_domain:
        return answered_from_context is False
    return answered_from_context is True
