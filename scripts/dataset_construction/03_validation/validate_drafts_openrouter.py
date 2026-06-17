"""Continue Llama 4 Scout validation via OpenRouter, picking up where Groq left off.

Writes into the same groq_* fields; sets groq_provider="openrouter" for provenance.
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
NEEDS_REVIEW = REPO_ROOT / "data" / "v1" / "needs_human_review.json"

MODEL = "meta-llama/llama-4-scout"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DELAY = 4.0
RETRY_DELAY = 8.0

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


def validate_record(api_key: str, record: dict) -> dict:
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return {"groq_image_ok": False, "groq_query_ok": False,
                "groq_answer_ok": False, "groq_notes": "image file missing",
                "groq_validated": False, "groq_provider": "openrouter"}

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
                    print(f"  429 rate limit on retry")
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
            text = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences
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
                "groq_provider": "openrouter",
            }

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

    return {"groq_image_ok": False, "groq_query_ok": False,
            "groq_answer_ok": False, "groq_notes": "validation_error",
            "groq_validated": False, "groq_provider": "openrouter"}


def save_needs_review(records: list) -> int:
    flagged = []
    for r in records:
        h_ok = r.get("haiku_validated") is True
        g_ok = r.get("groq_validated") is True
        if not h_ok and not g_ok:
            continue

        h_fail = (not r.get("haiku_image_ok", True) or not r.get("haiku_query_ok", True)
                  or not r.get("haiku_answer_ok", True))
        g_fail = (not r.get("groq_image_ok", True) or not r.get("groq_query_ok", True)
                  or not r.get("groq_answer_ok", True))
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
    api_key = __import__("os").environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    with open(CANDIDATES) as f:
        records = json.load(f)

    all_with_query = [r for r in records if r.get("query", "").strip()]
    groq_done = [r for r in all_with_query if r.get("groq_validated") is True]
    to_validate = [r for r in all_with_query if r.get("groq_validated") is not True]

    print(f"Total records         : {len(records)}")
    print(f"Already groq_validated: {len(groq_done)}")
    print(f"To validate now       : {len(to_validate)}")
    print(f"Model                 : {MODEL}\n")

    rec_index = {r["id"]: i for i, r in enumerate(records)}
    processed = 0
    errors = 0

    for rec in to_validate:
        idx = rec_index[rec["id"]]
        print(f"[{idx+1}/{len(records)}] {rec['id']} — {rec['image_filename'][:55]}")

        result = validate_record(api_key, rec)
        records[idx].update(result)
        processed += 1
        if not result["groq_validated"]:
            errors += 1

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

    # Save combined review list
    n_flagged = save_needs_review(records)
    print(f"\nSaved {n_flagged} records to {NEEDS_REVIEW.relative_to(REPO_ROOT)}")

    # Summary
    by_provider = {}
    for r in records:
        if r.get("groq_validated") is True:
            p = r.get("groq_provider", "groq")
            by_provider[p] = by_provider.get(p, 0) + 1

    total_groq_done = sum(1 for r in records if r.get("groq_validated") is True)
    total_errors = sum(1 for r in records if r.get("groq_validated") is False)

    print("\n=== SUMMARY ===")
    print(f"Validated this run    : {processed}")
    print(f"Validation errors     : {errors}")
    print(f"Combined total (groq_validated=true): {total_groq_done}/189")
    print(f"Remaining errors      : {total_errors}")
    print("\nBy provider:")
    for provider, n in sorted(by_provider.items()):
        print(f"  {provider:<12}: {n}")
    print(f"\nneeds_human_review.json: {n_flagged} records")


if __name__ == "__main__":
    main()
