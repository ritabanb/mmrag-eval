"""Top up neural_networks (+11), evaluation_metrics (+3), systems_pipelines (+1)."""

import json
import re
import time
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

TARGETS = {
    "neural_networks": {
        "goal": 40,
        "terms": [
            "convolutional neural network pooling diagram",
            "neural network activation function diagram",
            "recurrent neural network unrolled diagram",
            "encoder decoder neural network diagram",
            "siamese network architecture diagram",
            "capsule network architecture diagram",
            "neural network weight initialization diagram",
            "attention head visualization transformer",
            "embedding layer neural network diagram",
            "neural network hyperparameter tuning diagram",
            "feedforward neural network diagram",
        ],
    },
    "evaluation_metrics": {
        "goal": 40,
        "terms": [
            "mean squared error loss curve machine learning",
            "accuracy versus epochs training curve",
            "interpolated average precision diagram",
        ],
    },
    "systems_pipelines": {
        "goal": 40,
        "terms": [
            "batch inference pipeline machine learning diagram",
        ],
    },
}

NOISE_WORDS = [
    "bird", "animal", "wildlife", "person", "portrait",
    "building", "map", "sport", "racing", "photo",
    "geograph.org.uk",
]

HEADERS = {"User-Agent": "mmrag-eval-dataset-builder/1.0 (ritabanb@gmail.com)"}


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


def collect_for_category(cat: str, need: int, terms: list, seen_fns: set) -> list:
    print(f"\n{'='*60}")
    print(f"Category: {cat}  |  Need: {need}")
    print(f"{'='*60}")
    collected = []

    for term in terms:
        if len(collected) >= need:
            break
        print(f"\n  Searching: '{term}'  ({len(collected)}/{need})")
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
            if len(collected) >= need:
                break
            batch = hits[i:i + 10]
            try:
                pages = wikimedia_imageinfo("|".join(h["title"] for h in batch))
                time.sleep(2)
            except Exception as e:
                print(f"  imageinfo failed: {e}")
                continue

            for _, page in pages.items():
                if len(collected) >= need:
                    break

                title = page.get("title", "")
                if noisy(title):
                    print(f"    SKIP noise: {title[:60]}")
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
                    print(f"    SKIP noise fn: {fn[:60]}")
                    continue
                if fn.lower() in seen_fns:
                    print(f"    SKIP dup: {fn[:60]}")
                    continue

                dest = IMAGES_DIR / fn
                print(f"    Downloading: {fn[:70]}")
                try:
                    req = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=30) as r:
                        dest.write_bytes(r.read())
                except Exception as e:
                    print(f"    download failed: {e}")
                    continue

                try:
                    img = PILImage.open(dest)
                    img.verify()
                    img = PILImage.open(dest)
                    iw, ih = img.size
                    if iw < 200 or ih < 200 or dest.stat().st_size > 10 * 1024 * 1024:
                        dest.unlink(missing_ok=True)
                        print(f"    SKIP (too small/large)")
                        continue
                except Exception as e:
                    dest.unlink(missing_ok=True)
                    print(f"    SKIP (PIL failed: {e})")
                    continue

                seen_fns.add(fn.lower())
                collected.append({
                    "filename": fn,
                    "wikimedia_url": url,
                    "license": lic,
                    "category": cat,
                    "width": ii["width"],
                    "height": ii["height"],
                })
                print(f"    OK: {fn[:60]} ({ii['width']}x{ii['height']})")
                time.sleep(2)

    print(f"\n  Collected {len(collected)}/{need} for {cat}")
    return collected


# ── Main ─────────────────────────────────────────────────────────────────────

try:
    with open(SAMPLE_JSON) as f:
        sample_fns = {r["image_filename"].lower() for r in json.load(f)}
except FileNotFoundError:
    sample_fns = set()

with open(CANDIDATES) as f:
    current = json.load(f)

seen_fns = sample_fns | {r["image_filename"].lower() for r in current}
cat_counts = Counter(r["category"] for r in current)

print("Current per-category counts:")
for cat in sorted(cat_counts):
    print(f"  {cat:<25}: {cat_counts[cat]}")

all_new = []
for cat, cfg in TARGETS.items():
    have = cat_counts.get(cat, 0)
    need = cfg["goal"] - have
    if need <= 0:
        print(f"\n{cat} already at {have} — skipping.")
        continue
    new_recs = collect_for_category(cat, need, cfg["terms"], seen_fns)
    all_new.extend(new_recs)

# Append to candidates.json
with open(CANDIDATES) as f:
    current = json.load(f)

next_id = max(int(r["id"].split("-")[-1]) for r in current) + 1
for r in all_new:
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
        "category": r["category"],
        "quality_flag": "pending",
        "width": r["width"],
        "height": r["height"],
    })
    next_id += 1

with open(CANDIDATES, "w") as f:
    json.dump(current, f, indent=2, ensure_ascii=False)

# Final report
with open(CANDIDATES) as f:
    final = json.load(f)

cat_final = Counter(r["category"] for r in final)
flag_final = Counter(r.get("quality_flag", "missing") for r in final)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"Total records: {len(final)}")
print("\nPer-category:")
for cat in sorted(cat_final):
    n = cat_final[cat]
    goal = TARGETS.get(cat, {}).get("goal", 40)
    status = "✅" if n >= goal else f"⚠️  SHORT {goal - n}"
    print(f"  {cat:<25}: {n}  {status}")

print("\nQuality flag breakdown:")
for flag, n in sorted(flag_final.items()):
    print(f"  {flag:<12}: {n}")

all_fns = [r["image_filename"].lower() for r in final]
dupes = [fn for fn, cnt in Counter(all_fns).items() if cnt > 1]
if dupes:
    print(f"\nWARNING: {len(dupes)} duplicate filenames:")
    for d in dupes:
        print(f"  {d}")
else:
    print("\nNo duplicate image_filename values.")

print("\nDone. Waiting for instructions.")
