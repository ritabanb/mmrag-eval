"""Micro top-up: 3 more evaluation_metrics images. Avoids portrait/person contamination."""

import json, os, time, re, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).parent.parent
OUT_JSON = REPO_ROOT / "data" / "v1" / "candidates.json"
OUT_IMAGES = REPO_ROOT / "data" / "v1" / "images"

with open(REPO_ROOT / "data" / "sample" / "dataset.json") as f:
    V0 = {r["image_filename"].lower() for r in json.load(f)}
with open(OUT_JSON) as f:
    current = json.load(f)
SEEN = V0 | {r["image_filename"].lower() for r in current}
NEED = 3

# Very focused terms that map to well-known Wikimedia ML diagram images
TERMS = [
    "Jaccard index similarity",
    "Cohen kappa coefficient",
    "mean squared error regression",
    "regression error plot",
    "residual plot regression",
    "receiver operating characteristic",
    "binary classification threshold",
    "multiclass confusion matrix",
    "accuracy vs threshold curve",
    "precision recall tradeoff",
]

NOISE = [
    "portrait", "photograph", "person", "people", "spa", "swen",
    "racing", "formula", "beaker", "solar", "electricity", "battery",
    "buddhist", "pine", "ecology", "naval", "ship", "aircraft",
    "army", "military", "geograph.org.uk", "island",
]

HEADERS = {"User-Agent": "mmrag-eval-dataset-builder/1.0 (ritabanb@gmail.com)"}

def noisy(t):
    return any(n in t.lower() for n in NOISE)

def search(q):
    p = urllib.parse.urlencode({"action":"query","list":"search","srsearch":q,
        "srnamespace":"6","srlimit":30,"format":"json"})
    with urllib.request.urlopen(
        urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{p}", headers=HEADERS), timeout=15
    ) as r:
        return json.loads(r.read())

def imageinfo(titles):
    p = urllib.parse.urlencode({"action":"query","titles":titles,"prop":"imageinfo",
        "iiprop":"url|mime|size|extmetadata","format":"json"}, safe="|")
    with urllib.request.urlopen(
        urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{p}", headers=HEADERS), timeout=15
    ) as r:
        return json.loads(r.read())

records = []
seen = set(SEEN)

for term in TERMS:
    if len(records) >= NEED:
        break
    print(f"Searching: '{term}' ({len(records)}/{NEED})")
    try:
        hits = search(term).get("query", {}).get("search", [])
        time.sleep(2)
    except Exception as e:
        print(f"  failed: {e}"); continue
    if not hits:
        print("  No results."); continue

    for i in range(0, len(hits), 10):
        if len(records) >= NEED: break
        try:
            info = imageinfo("|".join(h["title"] for h in hits[i:i+10]))
            time.sleep(2)
        except: continue
        for _, page in info.get("query",{}).get("pages",{}).items():
            if len(records) >= NEED: break
            title = page.get("title","")
            if noisy(title): continue
            ii_list = page.get("imageinfo",[])
            if not ii_list: continue
            ii = ii_list[0]
            if ii.get("mime","") not in ("image/jpeg","image/png"): continue
            url = ii.get("url","")
            if not url or ii.get("size",0) > 10*1024*1024: continue
            if ii.get("width",0) < 200 or ii.get("height",0) < 200: continue
            lic = ii.get("extmetadata",{}).get("LicenseShortName",{}).get("value","")
            if not any(l in lic for l in ("CC BY-SA","CC BY 4","CC BY-SA 4","CC0")): continue

            fn = re.sub(r"^File:","",title,flags=re.IGNORECASE).strip()
            fn = re.sub(r'[<>:"/\\|?*]',"_",fn)
            if not fn.lower().endswith((".png",".jpg",".jpeg")):
                fn += ".png" if ii["mime"]=="image/png" else ".jpg"
            if noisy(fn) or fn.lower() in seen: continue

            dest = OUT_IMAGES / fn
            print(f"  Downloading: {fn}")
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=30) as r:
                    open(dest,"wb").write(r.read())
            except: continue

            try:
                img = PILImage.open(dest); img.verify()
                img = PILImage.open(dest); w,h = img.size
                if w < 200 or h < 200 or os.path.getsize(dest) > 10*1024*1024:
                    dest.unlink(missing_ok=True); continue
            except:
                dest.unlink(missing_ok=True); continue

            seen.add(fn.lower())
            records.append({"filename":fn,"wikimedia_url":url,"license":lic,
                "category":"evaluation_metrics","width":ii["width"],"height":ii["height"]})
            print(f"  OK: {fn} ({ii['width']}x{ii['height']})")
            time.sleep(2)

print(f"\nCollected {len(records)}/{NEED}")

with open(OUT_JSON) as f:
    existing = json.load(f)
next_id = len(existing) + 1
for r in records:
    existing.append({"id":f"mmrag-v1-{next_id:03d}","query":"","query_type":"",
        "image_path":f"data/v1/images/{r['filename']}","image_filename":r["filename"],
        "reference_answer":"","grounding_labels":[f"data/v1/images/{r['filename']}"],
        "wikimedia_url":r["wikimedia_url"],"license":r["license"],
        "source":"Wikimedia Commons","category":r["category"],
        "width":r["width"],"height":r["height"]})
    next_id += 1

with open(OUT_JSON,"w") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

from collections import Counter
final = Counter(r["category"] for r in existing)
targets = {"neural_networks":35,"classical_ml":40,"evaluation_metrics":40,"systems_pipelines":40,"statistical_concepts":45}
print(f"\nFINAL: {len(existing)}/200")
for cat,tgt in targets.items():
    c=final.get(cat,0)
    print(f"  {cat:<25} {c}/{tgt}  {'✅' if c>=tgt else f'⚠️ SHORT {tgt-c}'}")
