"""
Step 1: Remove records with quality_flag in {cull, api_error} + delete their images.
Step 2: Top up classical_ml to 40 records via Wikimedia Commons search.
Step 3: Report final per-category counts and duplicate check.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
IMAGES_DIR = REPO_ROOT / "data" / "v1" / "images"
SAMPLE_JSON = REPO_ROOT / "data" / "sample" / "dataset.json"

TARGET_CAT = "classical_ml"
TARGET_COUNT = 40

SEARCH_TERMS = [
    "decision tree classifier diagram machine learning",
    "random forest feature importance bar chart",
    "AdaBoost algorithm diagram machine learning",
    "XGBoost gradient boosting diagram",
    "k-means algorithm steps diagram",
    "DBSCAN density clustering diagram",
    "hierarchical clustering dendrogram",
    "PCA dimensionality reduction diagram",
    "t-SNE visualization diagram machine learning",
    "linear discriminant analysis diagram",
    "ridge regression regularization path",
    "lasso regression coefficient path",
    "learning rate schedule machine learning",
    "confusion matrix binary classification",
    "feature correlation heatmap machine learning",
]

NOISE_WORDS = [
    "bird", "animal", "wildlife", "species", "ecology",
    "plant", "flower", "fossil", "portrait", "person",
    "building", "map", "geography", "sport", "racing",
    "geograph.org.uk",
]

HEADERS = {"User-Agent": "mmrag-eval-dataset-builder/1.0 (ritabanb@gmail.com)"}


# ── helpers ──────────────────────────────────────────────────────────────────

def noisy(text: str) -> bool:
    t = text.lower()
    return any(n in t for n in NOISE_WORDS)


def wikimedia_search(query: str) -> list:
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "srnamespace": "6", "srlimit": 30, "format": "json",
    })
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{params}", headers=HEADERS
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read()).get("query", {}).get("search", [])


def wikimedia_imageinfo(titles: str) -> dict:
    params = urllib.parse.urlencode({
        "action": "query", "titles": titles,
        "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata",
        "format": "json",
    }, safe="|")
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{params}", headers=HEADERS
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read()).get("query", {}).get("pages", {})


def clean_filename(title: str, mime: str) -> str:
    fn = re.sub(r"^File:", "", title, flags=re.IGNORECASE).strip()
    fn = re.sub(r'[<>:"/\\|?*]', "_", fn)
    if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
        fn += ".png" if mime == "image/png" else ".jpg"
    return fn


def valid_license(lic: str) -> bool:
    return any(tok in lic for tok in ("CC BY-SA 4", "CC BY 4", "CC0"))


# ── STEP 1: remove bad records ────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Removing cull + api_error records")
print("=" * 60)

with open(CANDIDATES) as f:
    records = json.load(f)

keep, remove = [], []
for r in records:
    if r.get("quality_flag") in {"cull", "api_error"}:
        remove.append(r)
    else:
        keep.append(r)

print(f"Total before: {len(records)}")
print(f"Removing    : {len(remove)}")

deleted_images, missing_images = 0, 0
for r in remove:
    img = IMAGES_DIR / r["image_filename"]
    if img.exists():
        img.unlink()
        deleted_images += 1
    else:
        missing_images += 1
        print(f"  (image missing, skipping delete): {r['image_filename']}")

print(f"Images deleted: {deleted_images}  |  already missing: {missing_images}")
print(f"Records kept  : {len(keep)}")

cat_after_step1 = Counter(r["category"] for r in keep)
print("\nPer-category after removal:")
for cat in sorted(cat_after_step1):
    print(f"  {cat:<25}: {cat_after_step1[cat]}")

with open(CANDIDATES, "w") as f:
    json.dump(keep, f, indent=2, ensure_ascii=False)

print("\nSaved cleaned candidates.json")


# ── STEP 2: top up classical_ml ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Topping up classical_ml to 40")
print("=" * 60)

# Build de-dup set from sample + current candidates
try:
    with open(SAMPLE_JSON) as f:
        sample_fns = {r["image_filename"].lower() for r in json.load(f)}
except FileNotFoundError:
    sample_fns = set()

with open(CANDIDATES) as f:
    current = json.load(f)

seen_fns = sample_fns | {r["image_filename"].lower() for r in current}

current_classical = sum(1 for r in current if r["category"] == TARGET_CAT)
need = TARGET_COUNT - current_classical
print(f"classical_ml now: {current_classical}  |  need: {need}")

if need <= 0:
    print("Already at target — nothing to do.")
    new_records = []
else:
    new_records = []

    for term in SEARCH_TERMS:
        if len(new_records) >= need:
            break

        print(f"\nSearching: '{term}'  ({len(new_records)}/{need})")
        try:
            hits = wikimedia_search(term)
            time.sleep(2)
        except Exception as e:
            print(f"  search failed: {e}")
            continue

        if not hits:
            print("  No results.")
            continue

        for i in range(0, len(hits), 10):
            if len(new_records) >= need:
                break
            batch = hits[i:i + 10]
            try:
                pages = wikimedia_imageinfo("|".join(h["title"] for h in batch))
                time.sleep(2)
            except Exception as e:
                print(f"  imageinfo failed: {e}")
                continue

            for _, page in pages.items():
                if len(new_records) >= need:
                    break

                title = page.get("title", "")
                if noisy(title):
                    print(f"  SKIP noise title: {title[:60]}")
                    continue

                ii_list = page.get("imageinfo", [])
                if not ii_list:
                    continue
                ii = ii_list[0]

                mime = ii.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue

                url = ii.get("url", "")
                size = ii.get("size", 0)
                w, h = ii.get("width", 0), ii.get("height", 0)

                if not url or size > 10 * 1024 * 1024:
                    continue
                if w < 200 or h < 200:
                    continue

                lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
                if not valid_license(lic):
                    continue

                fn = clean_filename(title, mime)
                if noisy(fn):
                    print(f"  SKIP noise filename: {fn[:60]}")
                    continue
                if fn.lower() in seen_fns:
                    print(f"  SKIP dup: {fn[:60]}")
                    continue

                dest = IMAGES_DIR / fn
                print(f"  Downloading: {fn[:70]}")
                try:
                    req = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=30) as r:
                        dest.write_bytes(r.read())
                except Exception as e:
                    print(f"  download failed: {e}")
                    continue

                # Validate image
                try:
                    img = PILImage.open(dest)
                    img.verify()
                    img = PILImage.open(dest)
                    iw, ih = img.size
                    if iw < 200 or ih < 200 or dest.stat().st_size > 10 * 1024 * 1024:
                        dest.unlink(missing_ok=True)
                        print(f"  SKIP (too small/large after verify)")
                        continue
                except Exception as e:
                    dest.unlink(missing_ok=True)
                    print(f"  SKIP (PIL verify failed: {e})")
                    continue

                seen_fns.add(fn.lower())
                new_records.append({
                    "filename": fn,
                    "wikimedia_url": url,
                    "license": lic,
                    "width": ii["width"],
                    "height": ii["height"],
                })
                print(f"  OK: {fn[:60]} ({ii['width']}x{ii['height']})")
                time.sleep(2)

    print(f"\nCollected {len(new_records)}/{need} new classical_ml records")


# ── Append new records to candidates.json ────────────────────────────────────

with open(CANDIDATES) as f:
    current = json.load(f)

next_id = max(int(r["id"].split("-")[-1]) for r in current) + 1

for r in new_records:
    current.append({
        "id": f"mmrag-v1-{next_id:03d}",
        "query": "",
        "query_type": "",
        "image_path": f"data/v1/images/{r['filename']}",
        "image_filename": r["filename"],
        "reference_answer": "",
        "grounding_labels": [f"data/v1/images/{r['filename']}"],
        "wikimedia_url": r["wikimedia_url"],
        "license": r["license"],
        "source": "Wikimedia Commons",
        "category": TARGET_CAT,
        "quality_flag": "pending",
        "width": r["width"],
        "height": r["height"],
    })
    next_id += 1

with open(CANDIDATES, "w") as f:
    json.dump(current, f, indent=2, ensure_ascii=False)


# ── STEP 3: report ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Final report")
print("=" * 60)

with open(CANDIDATES) as f:
    final = json.load(f)

cat_counts = Counter(r["category"] for r in final)
flag_counts = Counter(r.get("quality_flag", "missing") for r in final)

print(f"Total records: {len(final)}")
print("\nPer-category:")
for cat in sorted(cat_counts):
    print(f"  {cat:<25}: {cat_counts[cat]}")

print("\nQuality flag breakdown:")
for flag, n in sorted(flag_counts.items()):
    print(f"  {flag:<12}: {n}")

# Duplicate check
all_fns = [r["image_filename"].lower() for r in final]
dupes = [fn for fn, cnt in Counter(all_fns).items() if cnt > 1]
if dupes:
    print(f"\nWARNING: {len(dupes)} duplicate image_filename values found:")
    for d in dupes:
        print(f"  {d}")
else:
    print("\nNo duplicate image_filename values.")

missing_fields = [
    r["id"] for r in final
    if not r.get("wikimedia_url") or not r.get("license") or not r.get("image_filename")
]
if missing_fields:
    print(f"\nWARNING: {len(missing_fields)} records missing required fields:")
    for mid in missing_fields:
        print(f"  {mid}")
else:
    print("All records have wikimedia_url, license, and image_filename.")

print("\nDone. Waiting for instructions.")
