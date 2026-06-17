"""
Supplement script: fill the 4 short categories from the first collection run.
Fixes: removes 'filetype:bitmap' from srsearch (it was treating it as literal text).
Targets: neural_networks (+6), classical_ml (+30), evaluation_metrics (+11), systems_pipelines (+31).
Appends to data/v1/candidates.json and data/v1/images/.
"""

import json
import os
import time
import re
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).parent.parent
OUT_JSON = REPO_ROOT / "data" / "v1" / "candidates.json"
OUT_IMAGES = REPO_ROOT / "data" / "v1" / "images"
EXISTING_JSON = REPO_ROOT / "data" / "sample" / "dataset.json"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)

with open(EXISTING_JSON) as f:
    existing = json.load(f)
EXISTING_FILENAMES = {r["image_filename"].lower() for r in existing}

with open(OUT_JSON) as f:
    current = json.load(f)
CURRENT_FILENAMES = {r["image_filename"].lower() for r in current}

ALL_SEEN = EXISTING_FILENAMES | CURRENT_FILENAMES

from collections import Counter
current_counts = Counter(r["category"] for r in current)

TARGETS = {
    "neural_networks": 35,
    "classical_ml": 40,
    "evaluation_metrics": 40,
    "systems_pipelines": 40,
}

NEEDS = {cat: TARGETS[cat] - current_counts.get(cat, 0) for cat in TARGETS}
print("Shortfalls:", NEEDS)

NOISE = {
    "franz kafka", "kafka museum", "kafka's", "kafkaesque",
    "portrait", "statue", "monument", "memorial", "plaque",
    "building", "school", "hospital", "university", "college",
    "ship", "hms", "uss", "vessel", "yacht",
    "aircraft", "airplane", "aerodynamic", "faa", "airfoil", "lift drag",
    "football", "soccer", "sport", "athletic",
    "genetics", "ancestry", "ethnicity", "population genetics",
    "dinosaur", "fossil", "paleontology",
    "tropical medicine", "liverpool school",
    "solar system", "astrophysics", "telescope",
    "cognitive bias", "bilişsel",
    "manuscript", "handwriting",
    "photograph of", "photo of",
    "news photo", "press photo",
}

SUPPLEMENT_TERMS = {
    "neural_networks": [
        "gated recurrent unit",
        "GRU recurrent network",
        "generative adversarial network",
        "GAN neural network",
        "LSTM cell architecture",
        "long short-term memory",
        "batch normalization layer",
        "dropout regularization neural",
        "ResNet skip connection",
        "residual network deep learning",
        "U-Net segmentation",
        "attention mechanism neural network",
        "backpropagation neural network",
        "neural network layer diagram",
        "deep learning architecture",
    ],
    "classical_ml": [
        "random forest algorithm",
        "decision tree ensemble",
        "gradient boosting",
        "XGBoost diagram",
        "k-nearest neighbor algorithm",
        "KNN classification",
        "naive Bayes classifier",
        "linear regression diagram",
        "logistic regression diagram",
        "support vector machine",
        "SVM hyperplane",
        "k-means clustering",
        "DBSCAN density clustering",
        "AdaBoost boosting",
        "bias variance tradeoff",
        "regularization machine learning",
        "lasso ridge regression",
        "principal component analysis",
        "PCA dimensionality reduction",
        "feature importance machine learning",
        "cross validation machine learning",
        "ensemble learning diagram",
        "boosting bagging diagram",
        "classification algorithm comparison",
        "confusion matrix classifier",
        "overfitting underfitting",
        "train test split",
        "hyperparameter tuning",
        "model selection machine learning",
        "clustering algorithm",
    ],
    "evaluation_metrics": [
        "F1 score",
        "precision recall F1",
        "confusion matrix",
        "ROC AUC curve",
        "BLEU score NLP",
        "NDCG information retrieval",
        "mean average precision",
        "calibration curve",
        "learning curve training",
        "cross entropy loss",
        "loss function machine learning",
        "accuracy precision recall",
        "model performance metrics",
        "benchmark evaluation",
        "elbow method clustering",
        "silhouette analysis",
        "lift curve marketing",
        "gain curve model",
        "error rate machine learning",
        "validation curve",
    ],
    "systems_pipelines": [
        "microservices",
        "service oriented architecture",
        "message queue broker",
        "Apache RabbitMQ",
        "ETL data pipeline",
        "data engineering pipeline",
        "machine learning deployment",
        "MLOps pipeline",
        "stream processing",
        "data streaming architecture",
        "MapReduce Hadoop",
        "distributed computing",
        "data lake warehouse",
        "cloud architecture diagram",
        "CI CD pipeline",
        "continuous integration deployment",
        "load balancing",
        "database replication",
        "API gateway architecture",
        "container orchestration Kubernetes",
        "Docker containerization",
        "serverless architecture",
        "event-driven architecture",
        "pub sub messaging",
        "data warehouse architecture",
        "real-time analytics",
        "batch processing pipeline",
        "feature engineering pipeline",
        "model monitoring drift",
        "A/B testing system",
        "recommendation system architecture",
        "search engine architecture",
        "graph database",
        "caching strategy",
        "software architecture pattern",
    ],
}

HEADERS = {"User-Agent": "mmrag-eval-dataset-builder/1.0 (ritabanb@gmail.com)"}

def wikimedia_search(query, limit=50):
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,  # NO filetype:bitmap — it broke everything
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

def is_noisy(title):
    t = title.lower()
    for word in NOISE:
        if word in t:
            return True
    return False

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

def collect_more(category, need, seen_filenames):
    terms = SUPPLEMENT_TERMS[category]
    records = []
    seen = set(seen_filenames)

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
            print(f"    No results.")
            continue

        batch_size = 10
        for i in range(0, len(hits), batch_size):
            if len(records) >= need:
                break
            batch = hits[i:i+batch_size]
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
                if not url:
                    continue

                if ii.get("size", 0) > 10 * 1024 * 1024:
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

                fname_lower = filename.lower()
                if fname_lower in seen:
                    print(f"    SKIP (dup): {filename}")
                    continue

                if is_noisy(filename):
                    print(f"    SKIP (noise filename): {filename}")
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
    new_records = []

    for category, need in NEEDS.items():
        if need <= 0:
            print(f"\n=== {category}: already at target, skipping ===")
            continue
        print(f"\n=== {category}: need {need} more ===")
        recs = collect_more(category, need, seen)
        for r in recs:
            seen.add(r["filename"].lower())
        new_records.extend(recs)
        print(f"  -> {len(recs)}/{need} collected")
        if len(recs) < need - 5:
            print(f"  *** WARNING: still {need - len(recs)} short ***")

    # Reload current and append
    with open(OUT_JSON) as f:
        existing_records = json.load(f)

    next_id = len(existing_records) + 1
    for r in new_records:
        existing_records.append({
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
        json.dump(existing_records, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"TOTAL: {len(existing_records)} records in {OUT_JSON}")
    final_counts = Counter(r["category"] for r in existing_records)
    all_targets = {**TARGETS, "statistical_concepts": 45}
    for cat, target in all_targets.items():
        count = final_counts.get(cat, 0)
        status = "✅" if count >= target - 5 else "⚠️ SHORT"
        print(f"  {cat}: {count}/{target} {status}")

if __name__ == "__main__":
    main()
