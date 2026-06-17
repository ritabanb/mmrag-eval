"""Build the combined public dataset from v1 survivors + v0 sample records.

Steps:
  1. Filter v1 candidates.json to 148 surviving records (clean + recovered + bucket_D)
  2. For each survivor, use the best available reference_answer:
       - reannotate2 pass-2 recoveries → reannotate2_answer
       - pass-1 recoveries and originals → reference_answer (already updated in place)
  3. Strip all internal validation fields; keep only public schema fields
  4. Copy each image to data/combined/images/
  5. Merge with v0 records from data/sample/dataset.json (50 records)
  6. Write data/combined/dataset.json sorted by id
  7. Print audit + per-category breakdown
"""
import json, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
V1_CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
V0_DATASET    = REPO_ROOT / "data" / "sample" / "dataset.json"
V0_IMAGES     = REPO_ROOT / "data" / "sample" / "images"
V1_IMAGES     = REPO_ROOT / "data" / "v1" / "images"
OUT_DIR       = REPO_ROOT / "data" / "combined"
OUT_IMAGES    = OUT_DIR / "images"
OUT_JSON      = OUT_DIR / "dataset.json"

# Public schema fields (ordered for readability)
PUBLIC_FIELDS = [
    "id", "query", "query_type",
    "image_path", "image_filename",
    "reference_answer", "grounding_labels",
    "wikimedia_url", "license", "source",
    "category", "width", "height", "quality_flag",
]

# IDs to drop
DROP_B = {"mmrag-v1-096", "mmrag-v1-200"}
DROP_C = {"mmrag-v1-028", "mmrag-v1-105"}

BUCKET_A_ALL = {
    "mmrag-v1-005","mmrag-v1-008","mmrag-v1-012","mmrag-v1-017","mmrag-v1-019",
    "mmrag-v1-020","mmrag-v1-021","mmrag-v1-027","mmrag-v1-030","mmrag-v1-031",
    "mmrag-v1-041","mmrag-v1-046","mmrag-v1-053","mmrag-v1-065","mmrag-v1-076",
    "mmrag-v1-077","mmrag-v1-087","mmrag-v1-088","mmrag-v1-091","mmrag-v1-092",
    "mmrag-v1-094","mmrag-v1-099","mmrag-v1-101","mmrag-v1-102","mmrag-v1-103",
    "mmrag-v1-111","mmrag-v1-125","mmrag-v1-162","mmrag-v1-165","mmrag-v1-168",
    "mmrag-v1-175","mmrag-v1-180","mmrag-v1-196","mmrag-v1-205","mmrag-v1-208",
    "mmrag-v1-213","mmrag-v1-214","mmrag-v1-216",
    "mmrag-v1-068","mmrag-v1-074","mmrag-v1-081","mmrag-v1-089","mmrag-v1-150",
    "mmrag-v1-198","mmrag-v1-199","mmrag-v1-217",
}

# Bucket-A records recovered in pass 2 (keep with reannotate2_answer)
RECOVERED_PASS2 = {
    "mmrag-v1-005","mmrag-v1-092","mmrag-v1-103","mmrag-v1-111","mmrag-v1-165",
    "mmrag-v1-168","mmrag-v1-180","mmrag-v1-205","mmrag-v1-214",
}

# Still-failing bucket-A (drop)
DROP_A = BUCKET_A_ALL - RECOVERED_PASS2

DROP_ALL = DROP_B | DROP_C | DROP_A


def clean_record(r: dict, new_image_path: str) -> dict:
    out = {}
    for field in PUBLIC_FIELDS:
        if field == "image_path":
            out["image_path"] = new_image_path
        elif field in r:
            out[field] = r[field]
        else:
            out[field] = None
    return out


def main() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # ── v1 survivors ─────────────────────────────────────────────────────────
    with open(V1_CANDIDATES) as f:
        v1_records = json.load(f)

    survivors = []
    skipped = 0
    pass2_used = 0

    for r in v1_records:
        rid = r["id"]
        if rid in DROP_ALL:
            skipped += 1
            continue

        # For pass-2 recoveries, use reannotate2_answer as the reference answer
        if rid in RECOVERED_PASS2:
            r = dict(r)
            r["reference_answer"] = r["reannotate2_answer"]
            pass2_used += 1

        # Copy image
        src = REPO_ROOT / r["image_path"]
        dst = OUT_IMAGES / r["image_filename"]
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: image missing for {rid}: {src}")

        new_path = f"data/combined/images/{r['image_filename']}"
        survivors.append(clean_record(r, new_path))

    print(f"v1 survivors: {len(survivors)}  (skipped {skipped}, pass-2 answers used: {pass2_used})")

    # ── v0 records ────────────────────────────────────────────────────────────
    with open(V0_DATASET) as f:
        v0_records = json.load(f)

    v0_out = []
    for r in v0_records:
        src = REPO_ROOT / r["image_path"]
        dst = OUT_IMAGES / r["image_filename"]
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: v0 image missing: {src}")

        new_path = f"data/combined/images/{r['image_filename']}"
        # v0 records lack width/height/quality_flag/category — pad with None
        out = {}
        for field in PUBLIC_FIELDS:
            if field == "image_path":
                out["image_path"] = new_path
            else:
                out[field] = r.get(field)
        v0_out.append(out)

    print(f"v0 records:   {len(v0_out)}")

    # ── Merge, sort, write ────────────────────────────────────────────────────
    combined = sorted(survivors + v0_out, key=lambda x: x["id"])

    with open(OUT_JSON, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    total = len(combined)
    print(f"Combined total: {total}  (expected {len(survivors)} + {len(v0_out)} = {len(survivors)+len(v0_out)})")
    print(f"Written to: {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Images dir: {OUT_IMAGES.relative_to(REPO_ROOT)}  ({len(list(OUT_IMAGES.iterdir()))} files)")

    # ── Per-category breakdown ────────────────────────────────────────────────
    print("\nPer-category breakdown:")
    from collections import Counter
    cats = Counter(r["category"] for r in combined)
    for cat, n in sorted(cats.items(), key=lambda x: (-x[1], str(x[0]))):
        print(f"  {str(cat):<30} : {n}")
    print(f"  {'TOTAL':<30} : {total}")

    # ── Final reconciliation ──────────────────────────────────────────────────
    print(f"\nReconciliation: {len(survivors)} v1 + {len(v0_out)} v0 = {total}")
    assert total == len(survivors) + len(v0_out), "Count mismatch!"
    assert len(survivors) == 148, f"Expected 148 v1 survivors, got {len(survivors)}"
    print("✓ All assertions pass")


if __name__ == "__main__":
    main()
