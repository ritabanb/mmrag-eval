#!/usr/bin/env python3
"""
Score CLIP and BM25 retrieval baselines using mmrag-eval metrics.

Reads results/clip_retrieved.json and results/bm25_retrieved.json,
scores each with retrieval_quality.score(k=5), aggregates globally
and per category, prints a results table, and runs a sanity-check
over 5 randomly sampled queries.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from mmrag_eval.metrics import retrieval_quality

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = REPO_ROOT / "data" / "combined" / "dataset.json"
RESULTS_DIR = REPO_ROOT / "results"
K = 5
SANITY_SEED = 42
SANITY_N = 5

RETRIEVERS = ["clip", "bm25"]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_retriever(records: list[dict], retrieved_map: dict[str, list[str]]) -> list[dict]:
    per_sample = []
    for r in records:
        ranked = retrieved_map.get(r["id"], [])
        result = retrieval_quality.score(
            retrieved_images=ranked,
            relevant_images=r["grounding_labels"],
            k=K,
        )
        result["id"] = r["id"]
        result["category"] = r.get("category") or "unknown"
        per_sample.append(result)
    return per_sample


def aggregate(per_sample: list[dict]) -> dict:
    n = len(per_sample)
    return {
        "ndcg_at_5": sum(r["ndcg_at_k"] for r in per_sample) / n,
        "recall_at_5": sum(r["recall_at_k"] for r in per_sample) / n,
        "n": n,
    }


def aggregate_by_category(per_sample: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list] = defaultdict(list)
    for r in per_sample:
        by_cat[r["category"]].append(r)
    return {cat: aggregate(samples) for cat, samples in sorted(by_cat.items())}


# ── Printing ──────────────────────────────────────────────────────────────────

def print_overall(agg: dict[str, dict]) -> None:
    print(f"\n{'Retriever':<8}  {'nDCG@5':>8}  {'Recall@5':>10}  {'N':>5}")
    print("-" * 38)
    for name, a in agg.items():
        print(f"{name:<8}  {a['ndcg_at_5']:>8.4f}  {a['recall_at_5']:>10.4f}  {a['n']:>5}")


def print_by_category(agg_by_cat: dict[str, dict[str, dict]]) -> None:
    all_cats = sorted({c for v in agg_by_cat.values() for c in v})
    col = 16  # width per metric pair
    header = f"\n{'Category':<36}"
    for name in RETRIEVERS:
        header += f"  {name + ' nDCG@5':>{col}}  {name + ' Rec@5':>{col}}"
    print(header)
    print("-" * (36 + len(RETRIEVERS) * (col * 2 + 4)))
    for cat in all_cats:
        row = f"{cat:<36}"
        for name in RETRIEVERS:
            a = agg_by_cat[name].get(cat, {"ndcg_at_5": float("nan"), "recall_at_5": float("nan")})
            row += f"  {a['ndcg_at_5']:>{col}.4f}  {a['recall_at_5']:>{col}.4f}"
        print(row)


def print_sanity(records: list[dict], retrievers_data: dict[str, dict]) -> None:
    rng = random.Random(SANITY_SEED)
    sample = rng.sample(records, SANITY_N)
    print(f"\n=== Sanity check: {SANITY_N} random queries (seed={SANITY_SEED}) ===\n")
    for r in sample:
        gt_fn = Path(r["grounding_labels"][0]).name
        print(f"Query : {r['query']}")
        print(f"GT    : {gt_fn}")
        for name in RETRIEVERS:
            ranked = retrievers_data[name].get(r["id"], [])
            top3_names = [Path(p).name for p in ranked[:3]]
            gt_rank = next(
                (i + 1 for i, p in enumerate(ranked) if p == r["grounding_labels"][0]),
                ">20",
            )
            print(f"  {name:<6} top-3: {top3_names}  (GT rank: {gt_rank})")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(DATA_JSON) as f:
        records = json.load(f)

    retrievers_data: dict[str, dict] = {}
    for name in RETRIEVERS:
        path = RESULTS_DIR / f"{name}_retrieved.json"
        with open(path) as f:
            retrievers_data[name] = json.load(f)
        print(f"Loaded {path.name}: {len(retrievers_data[name])} entries")

    # ── Score ─────────────────────────────────────────────────────────────────
    per_sample_all: dict[str, list[dict]] = {}
    for name in RETRIEVERS:
        per_sample_all[name] = evaluate_retriever(records, retrievers_data[name])

    agg = {name: aggregate(ps) for name, ps in per_sample_all.items()}
    agg_by_cat = {name: aggregate_by_category(ps) for name, ps in per_sample_all.items()}

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n=== Baseline Retrieval Results (k=5, 198 queries, leave-one-out) ===")
    print_overall(agg)
    print("\n=== Per-category breakdown ===")
    print_by_category(agg_by_cat)
    print_sanity(records, retrievers_data)

    # ── Save full results ─────────────────────────────────────────────────────
    output = {
        "aggregated": agg,
        "by_category": agg_by_cat,
        "per_sample": per_sample_all,
    }
    out_path = RESULTS_DIR / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
