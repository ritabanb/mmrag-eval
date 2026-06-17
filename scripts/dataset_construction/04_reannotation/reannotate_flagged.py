"""Re-annotate records doubly-flagged on answer_ok, then re-validate with Haiku.

Step 1: Load records where haiku_answer_ok=False AND gpt4omini_answer_ok=False
Step 2: Re-annotate via claude-sonnet-4-6, incorporating reviewer feedback
Step 3: Save corrected answers (original preserved in original_reference_answer)
Step 4: Re-validate the 77 via claude-haiku-4-5 (haiku_recheck_* fields)
Step 5: Print summary report

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

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"

DELAY = 0.5
RETRY_DELAY = 2.0

# Pricing per token
SONNET_IN  = 3.00  / 1_000_000
SONNET_OUT = 15.00 / 1_000_000
HAIKU_IN   = 1.00  / 1_000_000
HAIKU_OUT  = 5.00  / 1_000_000

REANNOTATE_PROMPT = """\
Here is an image, the current query, the current reference answer, and feedback \
from two independent reviewers who found factual errors in the reference answer.

Query: {query}
Current reference answer: {reference_answer}
Reviewer 1 (Haiku) notes: {haiku_notes}
Reviewer 2 (GPT-4o-mini) notes: {gpt4omini_notes}

Look at the image very carefully. Write a corrected reference answer that:
1. Fixes only the specific factual errors identified by the reviewers (wrong numbers, \
wrong colors, wrong positions, wrong labels)
2. States only what is directly visible in the image — no inferred formulas, named \
techniques, or domain knowledge not shown in the image itself
3. Keeps the same length and level of detail as the original (2-4 sentences)

If, after careful re-inspection, you believe the reviewers were incorrect and the \
original answer was already accurate, set corrected_answer to the original answer, \
set changed to false, and explain in disagreement_note."""

RECHECK_PROMPT = """\
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


def image_content(record: dict) -> dict:
    image_path = REPO_ROOT / record["image_path"]
    img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
    return {"type": "image", "source": {
        "type": "base64",
        "media_type": media_type(record["image_filename"]),
        "data": img_b64,
    }}


def reannotate_record(client: anthropic.Anthropic, record: dict) -> tuple[dict, int, int]:
    """Returns (result_dict, input_tokens, output_tokens)."""
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return ({"reannotated": False, "reannotation_changed": False,
                 "reannotation_disagreement_note": "image file missing"}, 0, 0)

    prompt = REANNOTATE_PROMPT.format(
        query=record["query"],
        reference_answer=record["reference_answer"],
        haiku_notes=record.get("haiku_notes", ""),
        gpt4omini_notes=record.get("gpt4omini_notes", ""),
    )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=SONNET,
                max_tokens=1024,
                messages=[{"role": "user", "content": [
                    image_content(record),
                    {"type": "text", "text": prompt},
                ]}],
                extra_body={"output_config": {"format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "corrected_answer": {"type": "string"},
                            "changed": {"type": "boolean"},
                            "disagreement_note": {"type": "string"},
                        },
                        "required": ["corrected_answer", "changed", "disagreement_note"],
                        "additionalProperties": False,
                    },
                }}},
            )
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text.strip()
                    break
            if not text:
                raise ValueError(f"empty content (stop_reason={response.stop_reason})")
            result = json.loads(text)
            pt = response.usage.input_tokens
            ct = response.usage.output_tokens
            return ({
                "reannotated": True,
                "reannotation_changed": bool(result.get("changed", False)),
                "reannotation_disagreement_note": str(result.get("disagreement_note", "")),
                "corrected_answer": str(result.get("corrected_answer", record["reference_answer"])),
            }, pt, ct)

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            delay = 5.0 * (attempt + 1)
            if attempt < 2:
                print(f"  parse/empty error ({e}), retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                print(f"  parse/empty error on final attempt: {e}")
        except anthropic.RateLimitError:
            delay = 15.0 * (attempt + 1)
            if attempt < 2:
                print(f"  rate limit, retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                print(f"  rate limit on final attempt")
        except anthropic.APIError as e:
            delay = 5.0 * (attempt + 1)
            if attempt < 2:
                print(f"  API error ({e}), retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                print(f"  API error on final attempt: {e}")

    return ({"reannotated": False, "reannotation_changed": False,
             "reannotation_disagreement_note": "api_error"}, 0, 0)


def recheck_record(client: anthropic.Anthropic, record: dict) -> tuple[dict, int, int]:
    """Returns (result_dict, input_tokens, output_tokens)."""
    image_path = REPO_ROOT / record["image_path"]
    if not image_path.exists():
        return ({"haiku_recheck_image_ok": False, "haiku_recheck_query_ok": False,
                 "haiku_recheck_answer_ok": False, "haiku_recheck_notes": "image file missing"}, 0, 0)

    prompt = RECHECK_PROMPT.format(
        query=record["query"],
        reference_answer=record["reference_answer"],
    )

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=HAIKU,
                max_tokens=512,
                messages=[{"role": "user", "content": [
                    image_content(record),
                    {"type": "text", "text": prompt},
                ]}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(text)
            pt = response.usage.input_tokens
            ct = response.usage.output_tokens
            return ({
                "haiku_recheck_image_ok": bool(result.get("image_ok", False)),
                "haiku_recheck_query_ok": bool(result.get("query_ok", False)),
                "haiku_recheck_answer_ok": bool(result.get("answer_ok", False)),
                "haiku_recheck_notes": str(result.get("notes", "")),
            }, pt, ct)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == 0:
                print(f"  recheck parse error ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  recheck parse error on retry: {e}")
        except anthropic.RateLimitError:
            if attempt == 0:
                print(f"  recheck rate limit, retrying in 10s...")
                time.sleep(10)
            else:
                print(f"  recheck rate limit on retry")
        except anthropic.APIError as e:
            if attempt == 0:
                print(f"  recheck API error ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  recheck API error on retry: {e}")

    return ({"haiku_recheck_image_ok": False, "haiku_recheck_query_ok": False,
             "haiku_recheck_answer_ok": False, "haiku_recheck_notes": "validation_error"}, 0, 0)


def main() -> None:
    client = anthropic.Anthropic()

    with open(CANDIDATES) as f:
        records = json.load(f)

    # ── STEP 1: Load flagged records ──────────────────────────────────────────
    flagged = [
        r for r in records
        if r.get("haiku_answer_ok") is False and r.get("gpt4omini_answer_ok") is False
    ]
    print(f"=== STEP 1: LOAD FLAGGED RECORDS ===")
    print(f"Total records in candidates.json : {len(records)}")
    print(f"Doubly-flagged on answer_ok      : {len(flagged)}")
    if len(flagged) != 77:
        print(f"WARNING: expected 77, got {len(flagged)}")
    print(f"\nFlagged IDs:")
    for r in flagged:
        print(f"  {r['id']}")

    rec_index = {r["id"]: i for i, r in enumerate(records)}

    # Resume: skip already reannotated
    to_reannotate = [r for r in flagged if r.get("reannotated") is not True]
    already_done = len(flagged) - len(to_reannotate)
    if already_done:
        print(f"\n{already_done} already reannotated — resuming from where we left off.")

    # ── STEP 2 + 3: Re-annotate via Sonnet ───────────────────────────────────
    print(f"\n=== STEP 2-3: RE-ANNOTATE VIA {SONNET} ===")
    sonnet_in_tokens = sonnet_out_tokens = 0
    reannotated_count = 0

    for i, rec in enumerate(to_reannotate, 1):
        idx = rec_index[rec["id"]]
        print(f"[{i}/{len(to_reannotate)}] {rec['id']} — {rec['image_filename'][:55]}")

        result, pt, ct = reannotate_record(client, rec)
        sonnet_in_tokens += pt
        sonnet_out_tokens += ct

        if result["reannotated"]:
            records[idx]["original_reference_answer"] = records[idx]["reference_answer"]
            records[idx]["reference_answer"] = result["corrected_answer"]
            records[idx]["reannotated"] = True
            records[idx]["reannotation_changed"] = result["reannotation_changed"]
            records[idx]["reannotation_disagreement_note"] = result["reannotation_disagreement_note"]
            reannotated_count += 1
            changed_str = "changed" if result["reannotation_changed"] else "unchanged (disagreed)"
            print(f"  {changed_str}")
            if result["reannotation_disagreement_note"]:
                print(f"  note: {result['reannotation_disagreement_note'][:100]}")
        else:
            print(f"  ERROR: reannotation failed")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        if i < len(to_reannotate):
            time.sleep(DELAY)

    sonnet_cost = sonnet_in_tokens * SONNET_IN + sonnet_out_tokens * SONNET_OUT
    print(f"\nSonnet pass complete: {reannotated_count}/{len(to_reannotate)} reannotated")
    print(f"Tokens: {sonnet_in_tokens:,} in / {sonnet_out_tokens:,} out — ${sonnet_cost:.4f}")

    # ── STEP 4: Re-validate via Haiku ────────────────────────────────────────
    print(f"\n=== STEP 4: RE-VALIDATE VIA {HAIKU} ===")
    haiku_in_tokens = haiku_out_tokens = 0

    # Reload to get current state (includes any previously reannotated records)
    with open(CANDIDATES) as f:
        records = json.load(f)

    to_recheck = [
        r for r in records
        if r.get("reannotated") is True and "haiku_recheck_answer_ok" not in r
    ]
    print(f"Records to recheck: {len(to_recheck)}")

    for i, rec in enumerate(to_recheck, 1):
        idx = rec_index[rec["id"]]
        print(f"[{i}/{len(to_recheck)}] {rec['id']}")

        result, pt, ct = recheck_record(client, rec)
        haiku_in_tokens += pt
        haiku_out_tokens += ct
        records[idx].update(result)

        status = (
            f"img={'✓' if result['haiku_recheck_image_ok'] else '✗'}  "
            f"qry={'✓' if result['haiku_recheck_query_ok'] else '✗'}  "
            f"ans={'✓' if result['haiku_recheck_answer_ok'] else '✗'}"
            + (f"  [{result['haiku_recheck_notes'][:70]}]" if result["haiku_recheck_notes"] else "")
        )
        print(f"  {status}")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        if i < len(to_recheck):
            time.sleep(DELAY)

    haiku_cost = haiku_in_tokens * HAIKU_IN + haiku_out_tokens * HAIKU_OUT

    # ── STEP 5: Report ────────────────────────────────────────────────────────
    print(f"\n=== STEP 5: REPORT ===")

    with open(CANDIDATES) as f:
        records = json.load(f)

    all_flagged = [
        r for r in records
        if r.get("haiku_answer_ok") is False and r.get("gpt4omini_answer_ok") is False
    ]
    reannotated   = [r for r in all_flagged if r.get("reannotated") is True]
    changed       = [r for r in reannotated if r.get("reannotation_changed") is True]
    disagreed     = [r for r in reannotated if r.get("reannotation_changed") is False]

    recheck_done  = [r for r in reannotated if "haiku_recheck_answer_ok" in r]
    now_pass      = [r for r in recheck_done if r.get("haiku_recheck_answer_ok") is True]
    still_fail    = [r for r in recheck_done if r.get("haiku_recheck_answer_ok") is False]

    total_cost = sonnet_cost + haiku_cost

    print(f"Doubly-flagged records           : {len(all_flagged)}")
    print(f"Successfully reannotated         : {len(reannotated)}")
    print(f"  Answer changed by Sonnet       : {len(changed)}")
    print(f"  Sonnet disagreed (kept original): {len(disagreed)}")
    print(f"\nOf {len(changed)} changed answers, Haiku recheck:")
    changed_pass = [r for r in now_pass if r.get("reannotation_changed") is True]
    changed_fail = [r for r in still_fail if r.get("reannotation_changed") is True]
    print(f"  Now pass answer_ok             : {len(changed_pass)}")
    print(f"  Still fail answer_ok           : {len(changed_fail)}")

    print(f"\nOf {len(disagreed)} Sonnet-disagreed (original kept), Haiku recheck:")
    disagree_pass = [r for r in now_pass if r.get("reannotation_changed") is False]
    disagree_fail = [r for r in still_fail if r.get("reannotation_changed") is False]
    print(f"  Pass answer_ok                 : {len(disagree_pass)}")
    print(f"  Still fail answer_ok           : {len(disagree_fail)}")

    print(f"\nTotal now passing recheck        : {len(now_pass)}/{len(recheck_done)}")
    print(f"Still failing — need manual review: {len(still_fail)}")

    if still_fail:
        print(f"\nIDs still failing after reannotation:")
        for r in still_fail:
            print(f"  {r['id']}  ({r.get('reannotation_disagreement_note','')[:60]})")

    print(f"\n=== COST ESTIMATE ===")
    print(f"Sonnet re-annotation : {sonnet_in_tokens:,} in / {sonnet_out_tokens:,} out — ${sonnet_cost:.4f}")
    print(f"Haiku recheck        : {haiku_in_tokens:,} in / {haiku_out_tokens:,} out — ${haiku_cost:.4f}")
    print(f"Total estimated cost : ${total_cost:.4f}")


if __name__ == "__main__":
    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    main()
