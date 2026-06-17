"""
Collect 200 Wikimedia Commons images for mmrag-eval v0.2 dataset.
Saves to data/v1/candidates.json and data/v1/images/.
Leaves query/query_type/reference_answer blank for human annotation.
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

# Load existing filenames to avoid duplicates
with open(EXISTING_JSON) as f:
    existing = json.load(f)
EXISTING_FILENAMES = {r["image_filename"].lower() for r in existing}

# Red-flag words in title/filename → skip
NOISE = {
    "franz kafka", "kafka museum", "kafka's", "kafkaesque",
    "portrait", "statue", "monument", "memorial", "plaque",
    "building", "school", "hospital", "university", "college",
    "ship", "hms", "uss", "vessel", "yacht",
    "aircraft", "airplane", "aerodynamic", "faa", "airfoil", "lift drag",
    "football", "soccer", "sport", "athletic",
    "genetics", "ancestry", "ethnicity", "population genetics", "somali", "ancient northeast",
    "dinosaur", "fossil", "paleontology",
    "tropical medicine", "liverpool school",
    "solar system", "astrophysics", "telescope",
    "cognitive bias", "bilişsel",
    "manuscript", "handwriting", "notebook",
    "photograph of", "photo of",
}

CATEGORY_TARGETS = {
    "neural_networks": 35,
    "classical_ml": 40,
    "evaluation_metrics": 40,
    "systems_pipelines": 40,
    "statistical_concepts": 45,
}

SEARCH_TERMS = {
    "neural_networks": [
        "gated recurrent unit diagram",
        "long short-term memory cell diagram",
        "generative adversarial network architecture diagram",
        "autoencoder neural network diagram",
        "residual network architecture diagram",
        "U-Net architecture diagram",
        "variational autoencoder diagram",
        "graph neural network architecture",
        "convolutional neural network feature map",
        "transformer neural network architecture",
        "word embedding vector space",
        "neural network gradient descent",
        "dropout neural network diagram",
        "batch normalization diagram",
        "neural network weights diagram",
    ],
    "classical_ml": [
        "random forest decision tree ensemble diagram",
        "gradient boosting machine learning diagram",
        "k-nearest neighbor classification diagram",
        "naive bayes machine learning diagram",
        "principal component analysis biplot",
        "linear regression line fit diagram",
        "logistic regression sigmoid curve",
        "k-means clustering diagram",
        "support vector machine kernel diagram",
        "bias variance tradeoff machine learning",
        "AdaBoost ensemble diagram",
        "DBSCAN clustering diagram",
        "manifold learning diagram",
        "feature selection machine learning diagram",
        "ridge lasso regression regularization diagram",
    ],
    "evaluation_metrics": [
        "receiver operating characteristic curve machine learning",
        "precision recall curve machine learning",
        "confusion matrix machine learning",
        "F1 score precision recall diagram",
        "mean average precision object detection",
        "BLEU score machine translation diagram",
        "normalized discounted cumulative gain diagram",
        "calibration plot machine learning",
        "learning curve machine learning",
        "silhouette score clustering diagram",
        "elbow method k-means diagram",
        "gain chart machine learning evaluation",
        "lift chart model evaluation",
        "AUC ROC machine learning",
        "cross entropy loss curve",
    ],
    "systems_pipelines": [
        "microservices architecture diagram",
        "message queue architecture diagram",
        "ETL pipeline architecture diagram",
        "machine learning pipeline diagram",
        "stream processing architecture diagram",
        "MapReduce distributed computing diagram",
        "feature store machine learning diagram",
        "model serving inference pipeline diagram",
        "data lake architecture diagram",
        "CI CD software pipeline diagram",
        "load balancer architecture diagram",
        "database sharding diagram",
        "event driven architecture diagram",
        "REST API architecture diagram",
        "distributed training machine learning diagram",
    ],
    "statistical_concepts": [
        "normal distribution bell curve diagram",
        "central limit theorem diagram",
        "Bayes theorem probability diagram",
        "hypothesis testing diagram statistics",
        "type I type II error statistics diagram",
        "correlation matrix statistics diagram",
        "probability density function diagram",
        "cumulative distribution function diagram",
        "box plot statistics diagram",
        "histogram frequency distribution",
        "QQ plot normal distribution",
        "Monte Carlo simulation diagram",
        "Markov chain diagram",
        "Poisson distribution diagram",
        "binomial distribution diagram",
        "Student t distribution diagram",
        "chi squared distribution diagram",
        "ANOVA diagram statistics",
        "regression residual plot",
        "kernel density estimation diagram",
    ],
}

HEADERS = {"User-Agent": "mmrag-eval-dataset-builder/1.0 (ritabanb@gmail.com)"}

def wikimedia_search(query, limit=20):
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
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
    # Remove "File:" prefix
    name = re.sub(r"^File:", "", title, flags=re.IGNORECASE).strip()
    # Replace chars unsafe for filesystem
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
                print(f"    HTTP {e.code} for {url}")
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

def collect_category(category, target, already_collected_filenames):
    terms = SEARCH_TERMS[category]
    records = []
    seen_filenames = set(already_collected_filenames)

    for term in terms:
        if len(records) >= target:
            break
        print(f"  [{category}] Searching: '{term}' ({len(records)}/{target})")
        try:
            result = wikimedia_search(term, limit=30)
            time.sleep(2.0)
        except Exception as e:
            print(f"    Search failed: {e}")
            continue

        hits = result.get("query", {}).get("search", [])
        if not hits:
            print(f"    No results.")
            continue

        # Batch imageinfo for up to 10 at a time
        batch_size = 10
        for i in range(0, len(hits), batch_size):
            if len(records) >= target:
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
                if len(records) >= target:
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
                    print(f"    SKIP (mime={mime}): {title}")
                    continue

                url = ii.get("url", "")
                if not url:
                    continue

                # Check file size from API (in bytes)
                api_size = ii.get("size", 0)
                if api_size > 10 * 1024 * 1024:
                    print(f"    SKIP (size={api_size/1e6:.1f}MB): {title}")
                    continue

                # Check dimensions
                width = ii.get("width", 0)
                height = ii.get("height", 0)
                if width < 200 or height < 200:
                    print(f"    SKIP (dim={width}x{height}): {title}")
                    continue

                # Check license
                meta = ii.get("extmetadata", {})
                license_short = meta.get("LicenseShortName", {}).get("value", "")
                if not any(lic in license_short for lic in ("CC BY-SA", "CC BY 4", "CC BY-SA 4", "CC0")):
                    print(f"    SKIP (license={license_short}): {title}")
                    continue

                ext = ".png" if mime == "image/png" else ".jpg"
                filename = safe_filename(title)
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    filename += ext

                # Dedup: check against existing v0 dataset and already-collected v1
                fname_lower = filename.lower()
                if fname_lower in seen_filenames or fname_lower in EXISTING_FILENAMES:
                    print(f"    SKIP (duplicate): {filename}")
                    continue

                dest = OUT_IMAGES / filename
                print(f"    Downloading: {filename}")
                ok = download_with_retry(url, dest)
                if not ok:
                    continue

                valid, reason = validate_image(dest)
                if not valid:
                    print(f"    SKIP (invalid image: {reason}): {filename}")
                    dest.unlink(missing_ok=True)
                    continue

                # Additional noise check on filename itself
                if is_noisy(filename):
                    print(f"    SKIP (noise in filename): {filename}")
                    dest.unlink(missing_ok=True)
                    continue

                seen_filenames.add(fname_lower)
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
    all_records = []
    id_counter = 1
    collected_filenames = set(EXISTING_FILENAMES)

    for category, target in CATEGORY_TARGETS.items():
        print(f"\n=== {category} (target: {target}) ===")
        cat_records = collect_category(category, target, collected_filenames)
        # Update collected_filenames with what we just got
        for r in cat_records:
            collected_filenames.add(r["filename"].lower())
        all_records.extend(cat_records)
        print(f"  -> {len(cat_records)}/{target} collected")
        if len(cat_records) < target - 5:
            print(f"  *** WARNING: {category} is {target - len(cat_records)} short of target ***")

    # Build final JSON
    output = []
    id_counter = 1
    for r in all_records:
        output.append({
            "id": f"mmrag-v1-{id_counter:03d}",
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
        id_counter += 1

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"TOTAL: {len(output)} records saved to {OUT_JSON}")
    print("\nPer-category breakdown:")
    for category, target in CATEGORY_TARGETS.items():
        count = sum(1 for r in output if r["category"] == category)
        status = "✅" if count >= target - 5 else "⚠️ SHORT"
        print(f"  {category}: {count}/{target} {status}")
    print(f"\nImages saved to: {OUT_IMAGES}")

if __name__ == "__main__":
    main()
