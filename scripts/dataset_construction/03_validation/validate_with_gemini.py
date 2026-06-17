"""Validate draft annotations in candidates.json using Gemini Flash vision API.

Reads GEMINI_API_KEY from environment — never hardcodes credentials.
"""

import base64
import json
import sys
import time
from collections import Counter
from pathlib import Path

import google.genai as genai
from google.genai import types

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
NEEDS_REVIEW = REPO_ROOT / "data" / "v1" / "needs_human_review.json"

MODEL = "gemini-flash-latest"
DELAY = 6.0   # seconds between requests (~10 RPM, under 20 RPM free tier limit)
RETRY_DELAY = 35.0  # respect the retry-after window on 429s

PROMPT_TEMPLATE = """\
I am building a benchmark dataset for evaluating multimodal RAG systems. \
I will show you an image along with a draft query and reference answer written by another AI model.

Your job is to validate whether the annotation is correct based strictly on what is visible in the image.

Draft query: {query}
Draft reference answer: {reference_answer}

Answer these three questions:

1. IMAGE_OK: Is this a clear technical diagram suitable for an ML/statistics/systems benchmark? \
Answer yes or no. A diagram is suitable if it shows a model architecture, algorithm visualization, \
evaluation plot, pipeline, or statistical concept. It is NOT suitable if it is a photo of a person, \
building, animal, or non-technical content.

2. QUERY_OK: Is the draft query answerable strictly from what is visible in this image, without \
requiring external knowledge? Answer yes or no, and if no, explain briefly.

3. ANSWER_OK: Does the draft reference answer accurately describe what is visually present in the \
image? Answer yes or no, and if no, identify specifically what is wrong or missing.

Respond in this exact JSON format with no other text:
{{
  "image_ok": true or false,
  "query_ok": true or false,
  "answer_ok": true or false,
  "notes": "brief note if any answer is false, else empty string"
}}"""


def media_type(filename: str) -> str:
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"


def validate_record(client: genai.Client, record: dict) -> dict:
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return {"gemini_image_ok": False, "gemini_query_ok": False,
                "gemini_answer_ok": False, "gemini_notes": "image file missing",
                "gemini_validated": False}

    image_bytes = image_path.read_bytes()
    mime = media_type(record["image_filename"])
    prompt = PROMPT_TEMPLATE.format(
        query=record["query"],
        reference_answer=record["reference_answer"],
    )

    import re as _re

    parse_errors = 0
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    types.Part.from_text(text=prompt),
                ],
            )
            text = response.text.strip()
            # Strip markdown fences
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(text)
            return {
                "gemini_image_ok": bool(result.get("image_ok", False)),
                "gemini_query_ok": bool(result.get("query_ok", False)),
                "gemini_answer_ok": bool(result.get("answer_ok", False)),
                "gemini_notes": str(result.get("notes", "")),
                "gemini_validated": True,
            }
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            parse_errors += 1
            if parse_errors >= 3:
                print(f"  parse error (giving up after {parse_errors} attempts): {e}")
                break
            print(f"  parse error ({e}), retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            m = _re.search(r"retry in ([\d.]+)s", err_str)
            if is_rate_limit:
                # Always honour the server's retry-after; loop until the window clears
                wait = float(m.group(1)) + 5.0 if m else 65.0
                print(f"  429 rate limit, waiting {wait:.0f}s (attempt {attempt})...")
                time.sleep(wait)
            else:
                # Non-rate-limit error: retry once, then give up
                if attempt >= 2:
                    print(f"  API error (giving up): {type(e).__name__}: {e}")
                    break
                wait = float(m.group(1)) + 5.0 if m else RETRY_DELAY
                print(f"  API error ({type(e).__name__}), retrying in {wait:.0f}s...")
                time.sleep(wait)

    return {"gemini_image_ok": False, "gemini_query_ok": False,
            "gemini_answer_ok": False, "gemini_notes": "validation_error",
            "gemini_validated": False}


def main() -> None:
    api_key = __import__("os").environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    with open(CANDIDATES) as f:
        records = json.load(f)

    to_validate = [r for r in records if r.get("query", "").strip()]
    skip_count = len(records) - len(to_validate)

    print(f"Total records   : {len(records)}")
    print(f"Skipped (no query): {skip_count}")
    print(f"To validate     : {len(to_validate)}")
    print(f"Model           : {MODEL}\n")

    # Build index for fast update
    rec_index = {r["id"]: i for i, r in enumerate(records)}
    processed = 0

    for rec in to_validate:
        # Resume: skip if already validated
        if rec.get("gemini_validated") is True:
            continue

        idx = rec_index[rec["id"]]
        total_idx = idx + 1
        print(f"[{total_idx}/{len(records)}] {rec['id']} — {rec['image_filename'][:55]}")

        result = validate_record(client, rec)
        records[idx].update(result)
        processed += 1

        status = (
            f"img={'✓' if result['gemini_image_ok'] else '✗'}  "
            f"qry={'✓' if result['gemini_query_ok'] else '✗'}  "
            f"ans={'✓' if result['gemini_answer_ok'] else '✗'}"
            + (f"  [{result['gemini_notes'][:60]}]" if result["gemini_notes"] else "")
        )
        print(f"  {status}")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        if processed < len(to_validate):
            time.sleep(DELAY)

    # Save needs_human_review.json
    flagged = [
        r for r in records
        if not r.get("gemini_image_ok", True)
        or not r.get("gemini_query_ok", True)
        or not r.get("gemini_answer_ok", True)
        or not r.get("gemini_validated", True)
    ]
    flagged_sorted = sorted(flagged, key=lambda r: r.get("category", ""))
    with open(NEEDS_REVIEW, "w") as f:
        json.dump(flagged_sorted, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(flagged_sorted)} records to {NEEDS_REVIEW.relative_to(REPO_ROOT)}")

    # Summary
    validated = [r for r in records if r.get("gemini_validated") is True]
    errors = [r for r in records if r.get("gemini_validated") is False and "gemini_notes" in r]
    full_agree = [r for r in validated
                  if r["gemini_image_ok"] and r["gemini_query_ok"] and r["gemini_answer_ok"]]
    full_disagree = [r for r in validated
                     if not r["gemini_image_ok"] and not r["gemini_query_ok"] and not r["gemini_answer_ok"]]
    partial = [r for r in validated if r not in full_agree and r not in full_disagree]

    print("\n=== SUMMARY ===")
    print(f"Total validated          : {len(validated)}")
    print(f"Full agreement (all ✓)   : {len(full_agree)}")
    print(f"Partial disagreement     : {len(partial)}")
    print(f"Full disagreement (all ✗): {len(full_disagree)}")
    print(f"Validation errors        : {len(errors)}")
    print(f"Written to needs_human_review.json: {len(flagged_sorted)}")

    print("\nPer category:")
    for cat in sorted(set(r["category"] for r in records)):
        cat_recs = [r for r in validated if r["category"] == cat]
        ok = sum(1 for r in cat_recs if r["gemini_image_ok"] and r["gemini_query_ok"] and r["gemini_answer_ok"])
        flagged_n = len(cat_recs) - ok
        print(f"  {cat:<25}: {len(cat_recs)} validated, {ok} full-agree, {flagged_n} flagged")


if __name__ == "__main__":
    main()
