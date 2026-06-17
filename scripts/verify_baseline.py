#!/usr/bin/env python3
"""
Independent verification of baseline retrieval results.
Steps 2, 3, and 4 of the re-verification pass.
Reads freshly generated retrieved JSONs — does NOT re-run the retrievers.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON  = REPO_ROOT / "data" / "combined" / "dataset.json"
CLIP_JSON  = REPO_ROOT / "results" / "clip_retrieved.json"
BM25_JSON  = REPO_ROOT / "results" / "bm25_retrieved.json"
RANDOM_SEED = 7
RANDOM_OTHER_N = 10   # how many random other records to sample per query


# ── Tokenizer (identical to baseline_bm25.py) ────────────────────────────────
def tokenize(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if t]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


# ── Step 2: BM25 leakage test ─────────────────────────────────────────────────
def step2_leakage(records: list[dict]) -> None:
    print("=" * 68)
    print("STEP 2 — BM25 LEAKAGE: query ↔ reference_answer token overlap")
    print("=" * 68)
    rng = random.Random(RANDOM_SEED)

    own_jaccards   = []
    other_jaccards = []

    for i, r in enumerate(records):
        q_tok  = tokenize(r["query"])
        own_tok = tokenize(r["reference_answer"])
        own_j  = jaccard(q_tok, own_tok)
        own_jaccards.append(own_j)

        # 10 random OTHER records (excluding self)
        others = rng.sample([j for j in range(len(records)) if j != i], RANDOM_OTHER_N)
        other_j_vals = [jaccard(q_tok, tokenize(records[j]["reference_answer"])) for j in others]
        other_jaccards.append(sum(other_j_vals) / len(other_j_vals))

    own_mean   = sum(own_jaccards)   / len(own_jaccards)
    other_mean = sum(other_jaccards) / len(other_jaccards)
    own_sorted   = sorted(own_jaccards)
    other_sorted = sorted(other_jaccards)
    own_median   = own_sorted[len(own_sorted)  // 2]
    other_median = other_sorted[len(other_sorted) // 2]

    print(f"  Own-record  Jaccard:  mean={own_mean:.4f}  median={own_median:.4f}")
    print(f"  Other-avg   Jaccard:  mean={other_mean:.4f}  median={other_median:.4f}")
    ratio = own_mean / other_mean if other_mean > 0 else float("inf")
    print(f"  Ratio (own / other):  {ratio:.1f}x")

    # How many queries have own_jaccard > ALL 10 sampled others?
    own_dominates = sum(
        1 for i, r in enumerate(records)
        if own_jaccards[i] > 0
        and own_jaccards[i] >= other_jaccards[i]
    )
    print(f"  Queries where own overlap ≥ mean-of-10-others: {own_dominates}/{len(records)}")
    print()


# ── Step 3: Specific CLIP failure cases ───────────────────────────────────────
def step3_clip_failures(records: list[dict], clip_map: dict) -> None:
    print("=" * 68)
    print("STEP 3 — CLIP SPECIFIC FAILURE CASES (from freshly generated JSON)")
    print("=" * 68)

    targets = {
        "Precision and Recall Curve.png": None,
        "Multiheaded attention, block diagram.png": None,
    }

    for r in records:
        fn = Path(r["image_path"]).name
        if fn in targets:
            targets[fn] = r["id"]

    for fn, rec_id in targets.items():
        if rec_id is None:
            print(f"  WARNING: could not find record for {fn}")
            continue
        r = next(x for x in records if x["id"] == rec_id)
        gt = r["grounding_labels"][0]
        ranked = clip_map.get(rec_id, [])
        try:
            rank = ranked.index(gt) + 1
        except ValueError:
            rank = f">{ len(ranked)}"
        print(f"  {rec_id} | {fn}")
        print(f"    GT path:   {gt}")
        print(f"    CLIP rank: {rank}")
        print(f"    Top-5:     {[Path(p).name for p in ranked[:5]]}")
        print()


# ── Step 4: Bug checks ────────────────────────────────────────────────────────
def step4_bug_checks(records: list[dict], clip_map: dict, bm25_map: dict) -> None:
    print("=" * 68)
    print("STEP 4 — BUG CHECKS")
    print("=" * 68)

    # 4a: empty or malformed grounding_labels
    empty_gl = [r["id"] for r in records if not r.get("grounding_labels")]
    multi_gl = [r["id"] for r in records if len(r.get("grounding_labels", [])) != 1]
    print(f"  Records with empty grounding_labels:    {len(empty_gl)}")
    print(f"  Records with ≠1 grounding_labels:       {len(multi_gl)}")
    if empty_gl:
        print(f"    IDs: {empty_gl}")
    if multi_gl:
        print(f"    IDs: {multi_gl}")

    # 4b: every ground-truth path present in the full corpus
    corpus_paths = set(r["image_path"] for r in records)
    missing_from_corpus = [
        r["id"] for r in records
        if r["grounding_labels"][0] not in corpus_paths
    ]
    print(f"  GT not present in 198-image corpus:     {len(missing_from_corpus)}")
    if missing_from_corpus:
        print(f"    IDs: {missing_from_corpus[:10]}")

    # 4c: every query's GT appears somewhere in its ranked list
    build_id_to_ip = {r["id"]: r["image_path"] for r in records}
    gt_absent_clip = []
    gt_absent_bm25 = []
    for r in records:
        gt = r["grounding_labels"][0]
        if gt not in clip_map.get(r["id"], []):
            gt_absent_clip.append(r["id"])
        if gt not in bm25_map.get(r["id"], []):
            gt_absent_bm25.append(r["id"])
    print(f"  GT absent from CLIP top-20:             {len(gt_absent_clip)}")
    print(f"  GT absent from BM25 top-20:             {len(gt_absent_bm25)}")

    # 4d: paranoia — does any retriever rank an image path that equals
    #     a totally different record's query image at #1 suspiciously?
    #     More concretely: does any top-1 result appear >5 times across all queries?
    from collections import Counter
    clip_top1 = Counter(clip_map[r["id"]][0] for r in records if clip_map.get(r["id"]))
    bm25_top1 = Counter(bm25_map[r["id"]][0] for r in records if bm25_map.get(r["id"]))
    clip_repeated = [(p, n) for p, n in clip_top1.most_common(5) if n > 5]
    bm25_repeated = [(p, n) for p, n in bm25_top1.most_common(5) if n > 5]
    print(f"  CLIP top-1 images appearing >5× (self-match indicator): {clip_repeated if clip_repeated else 'none'}")
    print(f"  BM25 top-1 images appearing >5× (self-match indicator): {bm25_repeated if bm25_repeated else 'none'}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    with open(DATA_JSON)  as f: records  = json.load(f)
    with open(CLIP_JSON)  as f: clip_map = json.load(f)
    with open(BM25_JSON)  as f: bm25_map = json.load(f)

    print(f"Records: {len(records)}  |  CLIP entries: {len(clip_map)}  |  BM25 entries: {len(bm25_map)}\n")

    step2_leakage(records)
    step3_clip_failures(records, clip_map)
    step4_bug_checks(records, clip_map, bm25_map)


if __name__ == "__main__":
    main()
