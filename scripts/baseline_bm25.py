#!/usr/bin/env python3
"""
BM25 text retrieval baseline for mmrag-eval.

Corpus: the reference_answer field of all 198 records.
Query:  the query field of each record.

Tokenization (identical for corpus and queries — documented for reproducibility):
  1. Lowercase the input string.
  2. Replace every character that is not [a-z0-9] or whitespace with a single space.
  3. Split on whitespace; discard empty tokens.
  Intentionally simple: no stemming, no stopword removal, no sub-word splitting.
  This means vocabulary overlap between query and reference_answer drives retrieval.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = REPO_ROOT / "data" / "combined" / "dataset.json"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "bm25_retrieved.json"
TOP_K = 20


def tokenize(text: str) -> list[str]:
    # See module docstring for the exact tokenization contract.
    return [t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if t]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(DATA_JSON) as f:
        records = json.load(f)
    print(f"Records loaded: {len(records)}")

    image_paths = [r["image_path"] for r in records]

    # ── Build BM25 index on reference_answer ─────────────────────────────────
    corpus_tokens = [tokenize(r["reference_answer"]) for r in records]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"BM25 index built: {len(corpus_tokens)} documents")

    # ── Per-query ranking ─────────────────────────────────────────────────────
    retrieved: dict[str, list[str]] = {}

    for r in records:
        query_tokens = tokenize(r["query"])
        scores = bm25.get_scores(query_tokens)              # numpy array, len 198
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        retrieved[r["id"]] = [image_paths[idx] for idx in ranked_idx[:TOP_K]]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(retrieved, f, indent=2)
    print(f"Saved → {OUTPUT_FILE.relative_to(REPO_ROOT)}")

    # ── Ground-truth rank spot-check ──────────────────────────────────────────
    print("\nGround-truth rank (first 5 records):")
    for r in records[:5]:
        gt = r["grounding_labels"][0]
        ranked = retrieved[r["id"]]
        rank = ranked.index(gt) + 1 if gt in ranked else f">{TOP_K}"
        print(f"  {r['id']}  GT rank: {rank:>4}  {Path(gt).name}")


if __name__ == "__main__":
    main()
