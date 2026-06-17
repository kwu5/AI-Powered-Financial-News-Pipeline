# Ship E — Implementation Review (Labeled Test Set)

**Date:** 2026-06-09
**Status:** Code complete (all files byte-compile); test set still to be generated + labeled by hand.
**Spec:** `doc/ship-e-testset.md` · **Roadmap:** `IMPLEMENTATION_PLAN.md` (Ship E)

Full implementation of the Ship E tooling: the scripts that produce, label, and
validate `eval/testset.jsonl` — the hand-verified ground truth Ships F–H score
retrieval against.

## What was built

| File | Change | Status |
|------|--------|--------|
| `src/storage/database.py` | Added `get_articles_sample(n, min_content_len)` (random, content-filtered) + `get_articles_by_ids(ids)` | ✅ |
| `eval/gen_queries.py` | Filled in all 4 stubbed bodies (kept original docstrings/schema) | ✅ |
| `eval/label_testset.py` | New — resumable pooling + interactive y/n/s/q labeler + spot-check | ✅ |
| `eval/validate_testset.py` | New — full schema / ID-resolution validation, exit 0/1 | ✅ |
| `eval/__init__.py` | New — makes `python -m eval.x` resolve | ✅ |

## Workflow

These are run by the user — `gen_queries` hits the OpenAI API, and the labeler
needs an indexed corpus in ChromaDB.

```powershell
# 1. Generate candidate questions from sampled articles
python -m eval.gen_queries --num-articles 60 --per-article 1

# 2. (manual) Curate eval/queries_candidates.jsonl -> save as eval/queries_curated.jsonl
#    Drop weak/dupe questions; hand-add ~10 hard + 5-10 out-of-domain (set "type":"out_of_domain")

# 3. Label relevance interactively (resumable — stop/restart any time)
python -m eval.label_testset

# 4. Validate before declaring Ship E done
python -m eval.validate_testset
```

## Design decisions to sanity-check

1. **New filename: `eval/queries_curated.jsonl`.** The labeler reads this, not the
   raw `queries_candidates.jsonl`, so curation is non-destructive: gen writes
   candidates → you curate into a separate file → labeler reads that. Input lines
   need `query` (+ optional `source`/`type`/`notes`, defaulting to
   `llm`/`in_domain`/`""`). Point it at another file with `--in`.

2. **Resumability keys on query *text*,** not file position — curating/reordering
   the input won't re-ask labeled queries. `query_id`s (`q001`…) are auto-assigned,
   continuing past the highest already in `testset.jsonl`.

3. **`POOL_DEPTH = 25`** is a local constant in `label_testset.py` (not a `Settings`
   field), per the spec's "eval tooling ≠ runtime config" rule. Larger than
   `RETRIEVAL_TOP_K = 5` to avoid biasing labels toward the served top-k.

4. **Validation count (50–100) is a WARNING, not a hard failure** — keeps the script
   useful while the set is built up over several days. Hard failures are schema /
   ID-resolution issues only (exit code 1).

5. **Quit (`q`) semantics:** mid-query during the *main* pool, the in-progress query
   is discarded (re-labeled next run); mid-query during *spot-check*, what was
   labeled for that query is saved first. Everything already completed is always
   persisted (append-per-query).

## Notes

- `.gitignore` only sweeps `data/` and `output/`, so `eval/testset.jsonl` is tracked
  as ground truth — no change needed.
- Verification so far is byte-compile only (`python -m py_compile`). End-to-end runs
  (API + indexed corpus) are the user's to execute.

## Open follow-ups

- [ ] Run the 4-step workflow to produce a 50–100 row `eval/testset.jsonl`.
- [ ] Mark the Ship E task boxes in `doc/ship-e-testset.md` once the set passes validation.
- [ ] Commit `eval/testset.jsonl` (ground truth belongs in git).
