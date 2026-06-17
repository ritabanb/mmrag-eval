"""Fourth-pass validation using GPT-4o-mini via OpenRouter.

Adds gpt4omini_* fields to candidates.json alongside haiku_* and groq_* fields.
Tracks token usage and stops before a $1.80 cumulative spend.
Reads OPENROUTER_API_KEY from environment — never hardcodes credentials.
"""

import base64
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"

MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DELAY = 1.0
RETRY_DELAY = 3.0

# Pricing per token (not per million)
PRICE_INPUT = 0.15 / 1_000_000
PRICE_OUTPUT = 0.60 / 1_000_000
COST_CEILING = 1.80  # stop if cumulative cost approaches this

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

Respond in this exact JSON format with no other text. Use JSON boolean literals (true/false), \
not the words "yes" or "no":
{{
  "image_ok": true,
  "query_ok": true,
  "answer_ok": false,
  "notes": "example note explaining what is wrong"
}}"""


def media_type(filename: str) -> str:
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"


def validate_record(api_key: str, record: dict) -> tuple[dict, int, int]:
    """Returns (result_dict, prompt_tokens, completion_tokens)."""
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return ({"gpt4omini_image_ok": False, "gpt4omini_query_ok": False,
                 "gpt4omini_answer_ok": False, "gpt4omini_notes": "image file missing",
                 "gpt4omini_validated": False}, 0, 0)

    img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
    mime = media_type(record["image_filename"])
    prompt = PROMPT_TEMPLATE.format(
        query=record["query"],
        reference_answer=record["reference_answer"],
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ritabanb/mmrag-eval",
        "X-Title": "mmrag-eval annotation validator",
    }
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{img_b64}",
                }},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 512,
    }

    for attempt in range(2):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", RETRY_DELAY))
                if attempt == 0:
                    print(f"  429 rate limit, retrying in {retry_after:.0f}s...")
                    time.sleep(retry_after)
                    continue
                else:
                    print(f"  429 on retry")
                    break

            if resp.status_code != 200:
                msg = resp.json().get("error", {}).get("message", resp.text[:120])
                if attempt == 0:
                    print(f"  HTTP {resp.status_code} ({msg[:80]}), retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  HTTP {resp.status_code} on retry: {msg[:80]}")
                    break

            data = resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            text = data["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            result = json.loads(text)
            return ({
                "gpt4omini_image_ok": bool(result.get("image_ok", False)),
                "gpt4omini_query_ok": bool(result.get("query_ok", False)),
                "gpt4omini_answer_ok": bool(result.get("answer_ok", False)),
                "gpt4omini_notes": str(result.get("notes", "")),
                "gpt4omini_validated": True,
            }, prompt_tokens, completion_tokens)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == 0:
                print(f"  parse error ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  parse error on retry: {e}")
        except requests.RequestException as e:
            if attempt == 0:
                print(f"  request error ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  request error on retry: {e}")

    return ({"gpt4omini_image_ok": False, "gpt4omini_query_ok": False,
             "gpt4omini_answer_ok": False, "gpt4omini_notes": "validation_error",
             "gpt4omini_validated": False}, 0, 0)


def main() -> None:
    api_key = __import__("os").environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    with open(CANDIDATES) as f:
        records = json.load(f)

    all_with_query = [r for r in records if r.get("query", "").strip()]
    already_done = [r for r in all_with_query if r.get("gpt4omini_validated") is True]
    to_validate = [r for r in all_with_query if r.get("gpt4omini_validated") is not True]

    print(f"Total records     : {len(records)}")
    print(f"Already validated : {len(already_done)}")
    print(f"To validate now   : {len(to_validate)}")
    print(f"Model             : {MODEL}")
    print(f"Cost ceiling      : ${COST_CEILING:.2f}\n")

    rec_index = {r["id"]: i for i, r in enumerate(records)}
    processed = 0
    errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    cumulative_cost = 0.0

    for rec in to_validate:
        cost_so_far = total_prompt_tokens * PRICE_INPUT + total_completion_tokens * PRICE_OUTPUT
        if cost_so_far >= COST_CEILING:
            print(f"\nCOST CEILING REACHED (${cost_so_far:.4f} >= ${COST_CEILING:.2f})")
            print(f"Stopping after {processed} records to avoid exceeding budget.")
            break

        idx = rec_index[rec["id"]]
        print(f"[{idx+1}/{len(records)}] {rec['id']} — {rec['image_filename'][:55]}")

        result, pt, ct = validate_record(api_key, rec)
        records[idx].update(result)
        processed += 1
        total_prompt_tokens += pt
        total_completion_tokens += ct
        cumulative_cost = total_prompt_tokens * PRICE_INPUT + total_completion_tokens * PRICE_OUTPUT

        if not result["gpt4omini_validated"]:
            errors += 1

        status = (
            f"img={'✓' if result['gpt4omini_image_ok'] else '✗'}  "
            f"qry={'✓' if result['gpt4omini_query_ok'] else '✗'}  "
            f"ans={'✓' if result['gpt4omini_answer_ok'] else '✗'}  "
            f"[${cumulative_cost:.4f}]"
            + (f"  [{result['gpt4omini_notes'][:50]}]" if result["gpt4omini_notes"] else "")
        )
        print(f"  {status}")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        if processed < len(to_validate):
            time.sleep(DELAY)

    # Reload fresh for summary
    with open(CANDIDATES) as f:
        records = json.load(f)

    validated = [r for r in records if r.get("gpt4omini_validated") is True]
    img_ok  = sum(1 for r in validated if r["gpt4omini_image_ok"])
    qry_ok  = sum(1 for r in validated if r["gpt4omini_query_ok"])
    ans_ok  = sum(1 for r in validated if r["gpt4omini_answer_ok"])

    # Cross-validator agreement: flag records where 2+ validators disagree with the original draft
    # (original draft assumed "ok" = True for all three)
    # Count validators that flag each field as NOT ok
    flagged_cross = []
    for r in validated:
        validators_flagging = 0
        for prefix in ("haiku_", "groq_", "gpt4omini_"):
            if r.get(f"{prefix}validated") is True:
                if not r.get(f"{prefix}image_ok", True) or not r.get(f"{prefix}query_ok", True) or not r.get(f"{prefix}answer_ok", True):
                    validators_flagging += 1
        if validators_flagging >= 2:
            flagged_cross.append(r)

    print("\n=== SUMMARY ===")
    print(f"Validated this run    : {processed}")
    print(f"Validation errors     : {errors}")
    print(f"Total prompt tokens   : {total_prompt_tokens:,}")
    print(f"Total completion tok  : {total_completion_tokens:,}")
    print(f"Estimated cost        : ${cumulative_cost:.4f}")
    print(f"\nimage_ok  : {img_ok}/{len(validated)}")
    print(f"query_ok  : {qry_ok}/{len(validated)}")
    print(f"answer_ok : {ans_ok}/{len(validated)}")
    print(f"\nRecords flagged by 2+ validators: {len(flagged_cross)}")
    print("(these most need human review)")


if __name__ == "__main__":
    main()
