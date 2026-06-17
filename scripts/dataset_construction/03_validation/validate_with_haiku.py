"""Validate draft annotations in candidates.json using Claude Haiku 4.5 vision.

Cross-model check: annotations were written by claude-sonnet-4-6;
this validator uses claude-haiku-4-5 to independently assess each one.
Reads ANTHROPIC_API_KEY from environment — never hardcodes credentials.
"""

import base64
import json
import sys
import time
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
NEEDS_REVIEW = REPO_ROOT / "data" / "v1" / "needs_human_review.json"

MODEL = "claude-haiku-4-5"
DELAY = 0.5   # seconds between requests

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


def validate_record(client: anthropic.Anthropic, record: dict) -> dict:
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return {"haiku_image_ok": False, "haiku_query_ok": False,
                "haiku_answer_ok": False, "haiku_notes": "image file missing",
                "haiku_validated": False}

    img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
    mime = media_type(record["image_filename"])
    prompt = PROMPT_TEMPLATE.format(
        query=record["query"],
        reference_answer=record["reference_answer"],
    )

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": mime, "data": img_b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(text)
            return {
                "haiku_image_ok": bool(result.get("image_ok", False)),
                "haiku_query_ok": bool(result.get("query_ok", False)),
                "haiku_answer_ok": bool(result.get("answer_ok", False)),
                "haiku_notes": str(result.get("notes", "")),
                "haiku_validated": True,
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == 0:
                print(f"  parse error ({e}), retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  parse error on retry: {e}")
        except anthropic.RateLimitError as e:
            if attempt == 0:
                print(f"  rate limit, retrying in 10s...")
                time.sleep(10)
            else:
                print(f"  rate limit on retry: {e}")
        except anthropic.APIError as e:
            if attempt == 0:
                print(f"  API error ({e}), retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  API error on retry: {e}")

    return {"haiku_image_ok": False, "haiku_query_ok": False,
            "haiku_answer_ok": False, "haiku_notes": "validation_error",
            "haiku_validated": False}


def main() -> None:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(CANDIDATES) as f:
        records = json.load(f)

    to_validate = [r for r in records if r.get("query", "").strip()]
    already_done = sum(1 for r in records if r.get("haiku_validated") is True)
    todo = len(to_validate) - already_done

    print(f"Total records   : {len(records)}")
    print(f"To validate     : {len(to_validate)}")
    print(f"Already done    : {already_done}")
    print(f"Remaining       : {todo}")
    print(f"Model           : {MODEL}\n")

    rec_index = {r["id"]: i for i, r in enumerate(records)}
    processed = 0

    for rec in to_validate:
        if rec.get("haiku_validated") is True:
            continue

        idx = rec_index[rec["id"]]
        print(f"[{idx+1}/{len(records)}] {rec['id']} — {rec['image_filename'][:55]}")

        result = validate_record(client, rec)
        records[idx].update(result)
        processed += 1

        status = (
            f"img={'✓' if result['haiku_image_ok'] else '✗'}  "
            f"qry={'✓' if result['haiku_query_ok'] else '✗'}  "
            f"ans={'✓' if result['haiku_answer_ok'] else '✗'}"
            + (f"  [{result['haiku_notes'][:70]}]" if result["haiku_notes"] else "")
        )
        print(f"  {status}")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        if processed < todo:
            time.sleep(DELAY)

    # Save needs_human_review.json
    flagged = [
        r for r in records
        if not r.get("haiku_image_ok", True)
        or not r.get("haiku_query_ok", True)
        or not r.get("haiku_answer_ok", True)
        or not r.get("haiku_validated", True)
    ]
    flagged_sorted = sorted(flagged, key=lambda r: r.get("category", ""))
    with open(NEEDS_REVIEW, "w") as f:
        json.dump(flagged_sorted, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(flagged_sorted)} records to {NEEDS_REVIEW.relative_to(REPO_ROOT)}")

    # Summary
    validated = [r for r in records if r.get("haiku_validated") is True]
    errors    = [r for r in records if r.get("haiku_validated") is False]
    full_ok   = [r for r in validated if r["haiku_image_ok"] and r["haiku_query_ok"] and r["haiku_answer_ok"]]
    full_fail = [r for r in validated if not r["haiku_image_ok"] and not r["haiku_query_ok"] and not r["haiku_answer_ok"]]
    partial   = [r for r in validated if r not in full_ok and r not in full_fail]

    print("\n=== SUMMARY ===")
    print(f"Total validated          : {len(validated)}")
    print(f"Full agreement (all ✓)   : {len(full_ok)}")
    print(f"Partial disagreement     : {len(partial)}")
    print(f"Full disagreement (all ✗): {len(full_fail)}")
    print(f"Validation errors        : {len(errors)}")
    print(f"Written to needs_human_review.json: {len(flagged_sorted)}")

    print("\nPer category:")
    for cat in sorted(set(r["category"] for r in records)):
        cat_recs = [r for r in validated if r["category"] == cat]
        ok_n = sum(1 for r in cat_recs if r["haiku_image_ok"] and r["haiku_query_ok"] and r["haiku_answer_ok"])
        print(f"  {cat:<25}: {len(cat_recs)} validated, {ok_n} full-agree, {len(cat_recs)-ok_n} flagged")


if __name__ == "__main__":
    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    main()
