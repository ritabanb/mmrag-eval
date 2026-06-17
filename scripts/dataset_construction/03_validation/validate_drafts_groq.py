"""Third-pass validation using Groq's Llama 3.2 90B Vision.

Adds groq_* fields to candidates.json alongside existing haiku_* fields.
Reads GROQ_API_KEY from environment — never hardcodes credentials.
"""

import base64
import json
import sys
import time
from pathlib import Path

from groq import Groq, RateLimitError, APIError

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
NEEDS_REVIEW = REPO_ROOT / "data" / "v1" / "needs_human_review.json"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DELAY = 4.0
RETRY_DELAY = 20.0

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

Respond in this exact JSON format with no other text. Use JSON boolean literals (true/false), not the words "yes" or "no":
{{
  "image_ok": true,
  "query_ok": true,
  "answer_ok": false,
  "notes": "example note explaining what is wrong"
}}"""


def media_type(filename: str) -> str:
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def validate_record(client: Groq, record: dict) -> dict:
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return {"groq_image_ok": False, "groq_query_ok": False,
                "groq_answer_ok": False, "groq_notes": "image file missing",
                "groq_validated": False}

    img_b64 = encode_image(image_path)
    mime = media_type(record["image_filename"])
    prompt = PROMPT_TEMPLATE.format(
        query=record["query"],
        reference_answer=record["reference_answer"],
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime};base64,{img_b64}",
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=512,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(text)
            return {
                "groq_image_ok": bool(result.get("image_ok", False)),
                "groq_query_ok": bool(result.get("query_ok", False)),
                "groq_answer_ok": bool(result.get("answer_ok", False)),
                "groq_notes": str(result.get("notes", "")),
                "groq_validated": True,
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == 0:
                print(f"  parse error ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  parse error on retry: {e}")
        except RateLimitError as e:
            if attempt == 0:
                print(f"  rate limit, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  rate limit on retry: {e}")
        except APIError as e:
            if attempt == 0:
                print(f"  API error ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  API error on retry: {e}")

    return {"groq_image_ok": False, "groq_query_ok": False,
            "groq_answer_ok": False, "groq_notes": "validation_error",
            "groq_validated": False}


def save_needs_review(records: list) -> int:
    flagged = []
    for r in records:
        h_ok = r.get("haiku_validated") is True
        g_ok = r.get("groq_validated") is True
        if not h_ok and not g_ok:
            continue  # neither validator has run yet for this record

        h_fail = not r.get("haiku_image_ok", True) or not r.get("haiku_query_ok", True) or not r.get("haiku_answer_ok", True)
        g_fail = not r.get("groq_image_ok", True) or not r.get("groq_query_ok", True) or not r.get("groq_answer_ok", True)
        val_fail = not r.get("haiku_validated", True) or not r.get("groq_validated", True)

        disagree = False
        if h_ok and g_ok:
            disagree = (
                r.get("haiku_image_ok") != r.get("groq_image_ok") or
                r.get("haiku_query_ok") != r.get("groq_query_ok") or
                r.get("haiku_answer_ok") != r.get("groq_answer_ok")
            )

        if h_fail or g_fail or val_fail or disagree:
            entry = dict(r)
            entry["disagreement"] = disagree
            flagged.append(entry)

    flagged_sorted = sorted(flagged, key=lambda r: r.get("category", ""))
    with open(NEEDS_REVIEW, "w") as f:
        json.dump(flagged_sorted, f, indent=2, ensure_ascii=False)
    return len(flagged_sorted)


def main() -> None:
    client = Groq()  # reads GROQ_API_KEY from env

    with open(CANDIDATES) as f:
        records = json.load(f)

    to_validate = [r for r in records if r.get("query", "").strip()
                   and r.get("groq_validated") is None]
    already_done = sum(1 for r in records if r.get("groq_validated") is True)

    print(f"Total records   : {len(records)}")
    print(f"Already done    : {already_done}")
    print(f"To validate     : {len(to_validate)}")
    print(f"Model           : {MODEL}\n")

    rec_index = {r["id"]: i for i, r in enumerate(records)}
    processed = 0

    for rec in to_validate:
        idx = rec_index[rec["id"]]
        print(f"[{idx+1}/{len(records)}] {rec['id']} — {rec['image_filename'][:55]}")

        result = validate_record(client, rec)
        records[idx].update(result)
        processed += 1

        status = (
            f"img={'✓' if result['groq_image_ok'] else '✗'}  "
            f"qry={'✓' if result['groq_query_ok'] else '✗'}  "
            f"ans={'✓' if result['groq_answer_ok'] else '✗'}"
            + (f"  [{result['groq_notes'][:70]}]" if result["groq_notes"] else "")
        )
        print(f"  {status}")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        if processed < len(to_validate):
            time.sleep(DELAY)

    # Final combined disagreement list
    n_flagged = save_needs_review(records)
    print(f"\nSaved {n_flagged} records to {NEEDS_REVIEW.relative_to(REPO_ROOT)}")

    # Summary
    both = [r for r in records
            if r.get("haiku_validated") is True and r.get("groq_validated") is True]
    errors = [r for r in records
              if r.get("haiku_validated") is False or r.get("groq_validated") is False]
    full_agree = [r for r in both
                  if r["haiku_image_ok"] == r["groq_image_ok"]
                  and r["haiku_query_ok"] == r["groq_query_ok"]
                  and r["haiku_answer_ok"] == r["groq_answer_ok"]]
    disagree = [r for r in both if r not in full_agree]
    either_flagged = [r for r in both
                      if not r["haiku_image_ok"] or not r["haiku_query_ok"] or not r["haiku_answer_ok"]
                      or not r["groq_image_ok"] or not r["groq_query_ok"] or not r["groq_answer_ok"]]

    print("\n=== SUMMARY ===")
    print(f"Records with both validators complete : {len(both)}")
    print(f"Haiku + Groq fully agree              : {len(full_agree)}")
    print(f"Disagree on at least one flag         : {len(disagree)}")
    print(f"Flagged by either validator           : {len(either_flagged)}")
    print(f"Validation errors (either)            : {len(errors)}")
    print(f"Written to needs_human_review.json    : {n_flagged}")

    print("\nPer category (both validators):")
    for cat in sorted(set(r["category"] for r in records)):
        cat_both = [r for r in both if r["category"] == cat]
        cat_agree = sum(1 for r in cat_both
                        if r["haiku_image_ok"] == r["groq_image_ok"]
                        and r["haiku_query_ok"] == r["groq_query_ok"]
                        and r["haiku_answer_ok"] == r["groq_answer_ok"])
        print(f"  {cat:<25}: {len(cat_both)} both-validated, {cat_agree} agree, {len(cat_both)-cat_agree} disagree")


if __name__ == "__main__":
    if not __import__("os").environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    main()
