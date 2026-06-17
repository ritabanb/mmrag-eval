"""Generate draft annotations for all records in data/v1/candidates.json.

Calls claude-sonnet-4-6 with vision to produce query, query_type,
reference_answer, and quality_flag for each image, then writes results
back to candidates.json and saves flagged_for_review.json.

Uses ANTHROPIC_API_KEY from environment — never hardcodes credentials.
"""

import base64
import json
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
FLAGGED = REPO_ROOT / "data" / "v1" / "flagged_for_review.json"

MODEL = "claude-sonnet-4-6"

PROMPT = """\
You are helping annotate a benchmark dataset for evaluating multimodal RAG systems. \
I will show you an image. Your job is to generate three things:

1. QUERY: A single natural language question a user might ask when this image is retrieved \
as context. Aim for roughly 60% factual questions (about named concepts, labels, \
measurements, relationships shown) and 40% visual_description questions (asking to \
describe, summarize, or explain what is depicted).

2. QUERY_TYPE: exactly "factual" or "visual_description"

3. REFERENCE_ANSWER: 2-4 sentences grounded strictly in what is visually present in the image. \
Do not add outside knowledge that isn't shown.

4. QUALITY_FLAG:
   - "ok"     — image is clearly an ML/data-science diagram, chart, or technical figure
   - "review" — image may be relevant but is ambiguous or low quality
   - "cull"   — image is not a useful ML/data-science technical figure (photo, portrait, unrelated)

Respond in this exact JSON format with no other text:
{
  "query": "...",
  "query_type": "factual" or "visual_description",
  "reference_answer": "...",
  "quality_flag": "ok" or "cull" or "review"
}"""

VALID_FLAGS = {"ok", "cull", "review"}
VALID_TYPES = {"factual", "visual_description"}


def media_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return "image/png" if ext == "png" else "image/jpeg"


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def call_api(client: anthropic.Anthropic, image_path: Path) -> dict:
    img_b64 = encode_image(image_path)
    mt = media_type(image_path.name)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mt,
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def annotate_record(client: anthropic.Anthropic, record: dict) -> dict:
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        print(f"  SKIP (image missing): {record['image_filename']}")
        record["quality_flag"] = "api_error"
        record["query"] = ""
        record["query_type"] = ""
        record["reference_answer"] = ""
        return record

    for attempt in range(2):
        try:
            result = call_api(client, image_path)
            break
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == 0:
                print(f"  parse error ({e}), retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  parse error on retry: {e}")
                record["quality_flag"] = "api_error"
                record["query"] = ""
                record["query_type"] = ""
                record["reference_answer"] = ""
                return record
        except anthropic.APIError as e:
            if attempt == 0:
                print(f"  API error ({e}), retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  API error on retry: {e}")
                record["quality_flag"] = "api_error"
                record["query"] = ""
                record["query_type"] = ""
                record["reference_answer"] = ""
                return record

    # Validate and coerce fields
    query = str(result.get("query", "")).strip()
    query_type = str(result.get("query_type", "")).strip()
    reference_answer = str(result.get("reference_answer", "")).strip()
    quality_flag = str(result.get("quality_flag", "")).strip()

    if query_type not in VALID_TYPES:
        query_type = "factual"
    if quality_flag not in VALID_FLAGS:
        quality_flag = "review"

    record["query"] = query
    record["query_type"] = query_type
    record["reference_answer"] = reference_answer
    record["quality_flag"] = quality_flag
    return record


def save(records: list, path: Path) -> None:
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def main() -> None:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(CANDIDATES) as f:
        records = json.load(f)

    total = len(records)
    already_done = sum(1 for r in records if r.get("query", "").strip())
    todo = total - already_done

    print(f"Records: {total}  |  Already annotated: {already_done}  |  To process: {todo}")
    print(f"Model: {MODEL}\n")

    processed = 0
    skipped = 0

    for i, record in enumerate(records):
        if record.get("query", "").strip():
            skipped += 1
            continue

        idx = i + 1
        print(f"[{idx}/{total}] {record['id']} — {record['image_filename'][:60]}")

        records[i] = annotate_record(client, record)
        processed += 1

        flag = records[i].get("quality_flag", "?")
        print(f"  -> {flag} | {records[i].get('query_type','?')} | {records[i].get('query','')[:80]}")

        save(records, CANDIDATES)

        if processed < todo:
            time.sleep(0.5)

    # Save flagged records
    flagged = [r for r in records if r.get("quality_flag") in {"cull", "review"}]
    if flagged:
        save(flagged, FLAGGED)
        print(f"\nSaved {len(flagged)} flagged records to {FLAGGED.relative_to(REPO_ROOT)}")

    # Final summary
    flag_counts = Counter(r.get("quality_flag", "missing") for r in records)
    cat_counts = Counter(r["category"] for r in records)
    api_errors = flag_counts.get("api_error", 0)

    print("\n=== SUMMARY ===")
    print(f"Total records   : {total}")
    print(f"Processed now   : {processed}")
    print(f"Resumed (skipped): {skipped}")
    print(f"API errors      : {api_errors}")
    print("\nQuality flag breakdown:")
    for flag in ("ok", "review", "cull", "api_error"):
        n = flag_counts.get(flag, 0)
        if n:
            print(f"  {flag:<12}: {n}")
    print("\nPer category:")
    for cat in sorted(cat_counts):
        cat_recs = [r for r in records if r["category"] == cat]
        cat_flags = Counter(r.get("quality_flag", "missing") for r in cat_recs)
        print(f"  {cat:<25}: {cat_counts[cat]}  (ok={cat_flags.get('ok',0)}, "
              f"review={cat_flags.get('review',0)}, cull={cat_flags.get('cull',0)}, "
              f"err={cat_flags.get('api_error',0)})")


if __name__ == "__main__":
    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    main()
