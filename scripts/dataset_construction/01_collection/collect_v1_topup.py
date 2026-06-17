"""
Top-up script: fill evaluation_metrics (+9) and systems_pipelines (+13).
Uses tighter search terms to avoid prior contamination patterns.
Appends to data/v1/candidates.json; saves images to data/v1/images/.
"""

import json
import os
import time
import re
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from collections import Counter
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).parent.parent
OUT_JSON = REPO_ROOT / "data" / "v1" / "candidates.json"
OUT_IMAGES = REPO_ROOT / "data" / "v1" / "images"
EXISTING_JSON = REPO_ROOT / "data" / "sample" / "dataset.json"

with open(EXISTING_JSON) as f:
    v0 = json.load(f)
V0_FILENAMES = {r["image_filename"].lower() for r in v0}

with open(OUT_JSON) as f:
    current = json.load(f)
CURRENT_FILENAMES = {r["image_filename"].lower() for r in current}
ALL_SEEN = V0_FILENAMES | CURRENT_FILENAMES

current_counts = Counter(r["category"] for r in current)
TARGETS = {"evaluation_metrics": 40, "systems_pipelines": 40}
NEEDS = {cat: TARGETS[cat] - current_counts.get(cat, 0) for cat in TARGETS}
print("Shortfalls to fill:", NEEDS)

SEARCH_TERMS = {
    "evaluation_metrics": [
        "mean average precision object detection evaluation",
        "cross validation accuracy score plot",
        "training validation accuracy loss curve",
        "interpolated average precision recall",
        "detection error tradeoff curve biometrics",
        "precision at K ranking evaluation diagram",
        "ranking evaluation information retrieval",
        "AUROC receiver operating characteristic diagram",
        "Matthews correlation coefficient diagram",
        "mean reciprocal rank evaluation",
        "normalized discounted cumulative gain",
        "model selection criterion plot",
        "hyperparameter search validation curve",
    ],
    "systems_pipelines": [
        "publish subscribe architecture diagram",
        "Apache Spark cluster architecture",
        "Hadoop HDFS architecture diagram",
        "Kubernetes container orchestration diagram",
        "serverless computing architecture AWS",
        "data workflow pipeline diagram",
        "online model inference serving architecture",
        "feature store architecture machine learning",
        "MLOps continuous training pipeline",
        "real-time stream analytics architecture",
        "model registry deployment diagram",
        "distributed database architecture diagram",
        "API gateway microservice diagram",
        "message broker architecture RabbitMQ",
        "vector database architecture diagram",
    ],
}

# Strict noise filter — catches all prior contamination patterns
NOISE_TERMS = [
    "f1 austria", "formula 1", "formula one", "racing", "motorsport",
    "grand prix", "lauda", "bottas", "verstappen",
    "beaker", "sherd", "bronze age", "archaeological", "fossil",
    "solar panel", "solar energy", "first solar", "photovoltaic",
    "electricity price", "electricity cost", "energy price",
    "battery cost", "battery technology",
    "buddhist", "monk", "monastery", "religion",
    "jeffrey pine", "pine tree", "ecology", "species distribution",
    "oil pipeline", "gas pipeline", "pipeline inspection",
    "pipeline metoda", "pipeline method oil",
    "hulk", "boxer", "wrestling", "palelei",
    "soa island", "geograph.org.uk", "westsee",
    "submanifold", "manifold topology", "spectral geometry",
    "portrait", "photograph of", "photo of person",
    "school building", "hospital", "university campus",
    "ship", "hms", "uss", "aircraft", "airplane", "faa",
    "genetics", "ancestry", "ethnicity",
    "dinosaur",
]

HEADERS = {"User-Agent": "mmrag-eval-dataset-builder/1.0 (ritabanb@gmail.com)"}


def is_noisy(text):
    t = text.lower()
    for term in NOISE_TERMS:
        if term in t:
            return True
    return False


def wikimedia_search(query, limit=50):
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",
        "srlimit": limit,
        "format": "json",
    })
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def get_imageinfo(titles):
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": titles,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "format": "json",
    }, safe="|")
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def safe_filename(title):
    name = re.sub(r"^File:", "", title, flags=re.IGNORECASE).strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name


def download_with_retry(url, dest, max_attempts=4):
    delay = 2.0
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    429 rate limit, waiting {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"    HTTP {e.code}")
                return False
        except Exception as e:
            print(f"    Error: {e}")
            return False
    return False


def validate_image(path):
    try:
        img = PILImage.open(path)
        img.verify()
        img = PILImage.open(path)
        w, h = img.size
        if w < 200 or h < 200:
            return False, f"too small ({w}x{h})"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > 10:
            return False, f"too large ({size_mb:.1f}MB)"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def collect_topup(category, need, seen):
    terms = SEARCH_TERMS[category]
    records = []

    for term in terms:
        if len(records) >= need:
            break
        print(f"  [{category}] Searching: '{term}' ({len(records)}/{need})")
        try:
            result = wikimedia_search(term, limit=50)
            time.sleep(2.0)
        except Exception as e:
            print(f"    Search failed: {e}")
            continue

        hits = result.get("query", {}).get("search", [])
        if not hits:
            print("    No results.")
            continue

        for i in range(0, len(hits), 10):
            if len(records) >= need:
                break
            batch = hits[i:i+10]
            titles = "|".join(h["title"] for h in batch)
            try:
                info = get_imageinfo(titles)
                time.sleep(2.0)
            except Exception as e:
                print(f"    imageinfo failed: {e}")
                continue

            pages = info.get("query", {}).get("pages", {})
            for _, page in pages.items():
                if len(records) >= need:
                    break
                title = page.get("title", "")
                if is_noisy(title):
                    print(f"    SKIP (noise): {title}")
                    continue

                ii_list = page.get("imageinfo", [])
                if not ii_list:
                    continue
                ii = ii_list[0]

                mime = ii.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue

                url = ii.get("url", "")
                if not url or ii.get("size", 0) > 10 * 1024 * 1024:
                    continue

                width = ii.get("width", 0)
                height = ii.get("height", 0)
                if width < 200 or height < 200:
                    continue

                meta = ii.get("extmetadata", {})
                license_short = meta.get("LicenseShortName", {}).get("value", "")
                if not any(lic in license_short for lic in ("CC BY-SA", "CC BY 4", "CC BY-SA 4", "CC0")):
                    continue

                ext = ".png" if mime == "image/png" else ".jpg"
                filename = safe_filename(title)
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    filename += ext

                if is_noisy(filename):
                    print(f"    SKIP (noise filename): {filename}")
                    continue

                fname_lower = filename.lower()
                if fname_lower in seen:
                    print(f"    SKIP (dup): {filename}")
                    continue

                dest = OUT_IMAGES / filename
                print(f"    Downloading: {filename}")
                ok = download_with_retry(url, dest)
                if not ok:
                    continue

                valid, reason = validate_image(dest)
                if not valid:
                    print(f"    SKIP (invalid: {reason})")
                    dest.unlink(missing_ok=True)
                    continue

                seen.add(fname_lower)
                records.append({
                    "filename": filename,
                    "wikimedia_url": url,
                    "license": license_short,
                    "category": category,
                    "width": width,
                    "height": height,
                })
                print(f"    OK: {filename} ({width}x{height})")
                time.sleep(2.0)

    return records


def main():
    seen = set(ALL_SEEN)
    all_new = []

    for category, need in NEEDS.items():
        if need <= 0:
            print(f"\n=== {category}: already at target ===")
            continue
        print(f"\n=== {category}: need {need} more ===")
        recs = collect_topup(category, need, seen)
        for r in recs:
            seen.add(r["filename"].lower())
        all_new.extend(recs)
        print(f"  -> collected {len(recs)}/{need}")
        if len(recs) < need:
            print(f"  *** WARNING: still {need - len(recs)} short ***")

    # Append to candidates.json and re-number
    with open(OUT_JSON) as f:
        existing = json.load(f)

    next_id = len(existing) + 1
    for r in all_new:
        existing.append({
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
            "width": r["width"],
            "height": r["height"],
        })
        next_id += 1

    with open(OUT_JSON, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"TOTAL: {len(existing)} records")
    all_targets = {"neural_networks": 35, "classical_ml": 40, "evaluation_metrics": 40,
                   "systems_pipelines": 40, "statistical_concepts": 45}
    final_counts = Counter(r["category"] for r in existing)
    for cat, target in all_targets.items():
        count = final_counts.get(cat, 0)
        status = "✅" if count >= target else f"⚠️  SHORT by {target - count}"
        print(f"  {cat:<25} {count}/{target}  {status}")
    total = len(existing)
    print(f"\n  {'TOTAL':<25} {total}/200  {'✅' if total == 200 else '⚠️ NOT 200'}")


if __name__ == "__main__":
    main()
