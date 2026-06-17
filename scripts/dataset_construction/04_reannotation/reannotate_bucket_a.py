"""Second reannotation pass for Bucket-A (correction_failed) records.

Fresh-start prompt: no existing answer shown, reviewer notes used only to
flag the type of error to avoid — Sonnet must independently re-read the image.
Reads ANTHROPIC_API_KEY from environment — never hardcodes credentials.
"""
import base64, json, sys, time
from pathlib import Path
import anthropic

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES = REPO_ROOT / "data" / "v1" / "candidates.json"
SONNET = "claude-sonnet-4-6"
HAIKU  = "claude-haiku-4-5"
DELAY  = 0.5
SONNET_IN  = 3.00 / 1_000_000
SONNET_OUT = 15.00 / 1_000_000
HAIKU_IN   = 1.00 / 1_000_000
HAIKU_OUT  = 5.00 / 1_000_000

BUCKET_A_IDS = {
    # non-numeric
    "mmrag-v1-005","mmrag-v1-008","mmrag-v1-012","mmrag-v1-017","mmrag-v1-019",
    "mmrag-v1-020","mmrag-v1-021","mmrag-v1-027","mmrag-v1-030","mmrag-v1-031",
    "mmrag-v1-041","mmrag-v1-046","mmrag-v1-053","mmrag-v1-065","mmrag-v1-076",
    "mmrag-v1-077","mmrag-v1-087","mmrag-v1-088","mmrag-v1-091","mmrag-v1-092",
    "mmrag-v1-094","mmrag-v1-099","mmrag-v1-101","mmrag-v1-102","mmrag-v1-103",
    "mmrag-v1-111","mmrag-v1-125","mmrag-v1-162","mmrag-v1-165","mmrag-v1-168",
    "mmrag-v1-175","mmrag-v1-180","mmrag-v1-196","mmrag-v1-205","mmrag-v1-208",
    "mmrag-v1-213","mmrag-v1-214","mmrag-v1-216",
    # numeric
    "mmrag-v1-068","mmrag-v1-074","mmrag-v1-081","mmrag-v1-089","mmrag-v1-150",
    "mmrag-v1-198","mmrag-v1-199","mmrag-v1-217",
}

REANNOTATE2_PROMPT = """\
Look at this image carefully, as if for the first time. Ignore \
any prior answer you may have seen for this query — start fresh.

Query: {query}

Write a reference answer (2-4 sentences) that describes only \
what is directly and unambiguously visible in the image: exact \
labels, counts, colors, positions, and values as you read them \
directly from the image. If a detail is too small, blurry, or \
ambiguous to confirm with confidence, do not state it — describe \
only what you can verify with high confidence.

Two prior reviewers identified specific problems with earlier \
attempts at this answer. Their notes are provided ONLY so you \
know what kind of error to avoid — do not anchor on their stated \
"correct" values, independently verify everything yourself:
Reviewer notes: {haiku_recheck_notes}"""

HAIKU_PROMPT = """\
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


def reannotate2(client: anthropic.Anthropic, record: dict) -> tuple[dict, int, int]:
    """Fresh-start reannotation. Returns (result, input_tokens, output_tokens)."""
    prompt = REANNOTATE2_PROMPT.format(
        query=record["query"],
        haiku_recheck_notes=record.get("haiku_recheck_notes", ""),
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
                            "new_reference_answer": {"type": "string"},
                            "confidence": {"type": "string"},
                        },
                        "required": ["new_reference_answer", "confidence"],
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
            parsed = json.loads(text)
            if "new_reference_answer" not in parsed or "confidence" not in parsed:
                raise KeyError(f"missing keys in: {list(parsed.keys())}")
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            return parsed, in_tok, out_tok
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            delay = 5.0 * (attempt + 1)
            if attempt < 2:
                print(f"    parse error ({e}), retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                print(f"    parse error on final attempt: {e}")
    return {"new_reference_answer": "", "confidence": "low"}, 0, 0


def haiku_validate(client: anthropic.Anthropic, record: dict,
                   new_answer: str) -> tuple[dict, int, int]:
    """Run Haiku validation on new_reference_answer."""
    prompt = HAIKU_PROMPT.format(
        query=record["query"],
        reference_answer=new_answer,
    )
    for attempt in range(3):
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
            parsed = json.loads(text)
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            return {
                "reannotate2_haiku_image_ok":  bool(parsed.get("image_ok", False)),
                "reannotate2_haiku_query_ok":  bool(parsed.get("query_ok", False)),
                "reannotate2_haiku_answer_ok": bool(parsed.get("answer_ok", False)),
                "reannotate2_haiku_notes":     str(parsed.get("notes", "")),
            }, in_tok, out_tok
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            delay = 5.0 * (attempt + 1)
            if attempt < 2:
                print(f"    haiku parse error ({e}), retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                print(f"    haiku parse error on final attempt: {e}")
    return {
        "reannotate2_haiku_image_ok": False,
        "reannotate2_haiku_query_ok": False,
        "reannotate2_haiku_answer_ok": False,
        "reannotate2_haiku_notes": "validation_error",
    }, 0, 0


def main() -> None:
    client = anthropic.Anthropic()

    with open(CANDIDATES) as f:
        records = json.load(f)
    rec_index = {r["id"]: i for i, r in enumerate(records)}

    bucket_a = [r for r in records if r["id"] in BUCKET_A_IDS]
    already_done = [r for r in bucket_a if r.get("reannotate2_done")]
    todo = [r for r in bucket_a if not r.get("reannotate2_done")]

    print(f"Bucket-A total  : {len(bucket_a)}")
    print(f"Already done    : {len(already_done)}")
    print(f"Remaining       : {len(todo)}")
    print()

    total_in = total_out = 0
    haiku_in = haiku_out = 0

    # ── STEP 1: Sonnet fresh-start reannotation ──────────────────────────────
    print("=== STEP 1: Sonnet fresh-start reannotation ===")
    for n, rec in enumerate(todo, 1):
        idx = rec_index[rec["id"]]
        print(f"[{n}/{len(todo)}] {rec['id']}")
        result, in_tok, out_tok = reannotate2(client, rec)
        total_in  += in_tok
        total_out += out_tok

        new_ans    = result["new_reference_answer"]
        confidence = result["confidence"]
        print(f"  confidence={confidence}  answer={new_ans[:80]}...")

        records[idx]["reannotate2_answer"]     = new_ans
        records[idx]["reannotate2_confidence"] = confidence
        records[idx]["reannotate2_done"]       = bool(new_ans)

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY)

    # ── STEP 2: Haiku validation on new answers ───────────────────────────────
    print("\n=== STEP 2: Haiku validation ===")
    to_validate = [r for r in records if r["id"] in BUCKET_A_IDS
                   and r.get("reannotate2_done")
                   and "reannotate2_haiku_answer_ok" not in r]

    for n, rec in enumerate(to_validate, 1):
        idx = rec_index[rec["id"]]
        new_ans = rec.get("reannotate2_answer", "")
        if not new_ans:
            print(f"[{n}/{len(to_validate)}] {rec['id']}  SKIP (empty answer)")
            continue
        print(f"[{n}/{len(to_validate)}] {rec['id']}")
        val, h_in, h_out = haiku_validate(client, rec, new_ans)
        haiku_in  += h_in
        haiku_out += h_out

        records[idx].update(val)
        status = (
            f"img={'✓' if val['reannotate2_haiku_image_ok'] else '✗'}  "
            f"qry={'✓' if val['reannotate2_haiku_query_ok'] else '✗'}  "
            f"ans={'✓' if val['reannotate2_haiku_answer_ok'] else '✗'}"
            + (f"  [{val['reannotate2_haiku_notes'][:70]}]"
               if val["reannotate2_haiku_notes"] else "")
        )
        print(f"  {status}")

        with open(CANDIDATES, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY)

    # ── STEP 3: Report ────────────────────────────────────────────────────────
    print("\n=== STEP 3: REPORT ===")
    bucket_a_now = [r for r in records if r["id"] in BUCKET_A_IDS]

    now_pass     = [r for r in bucket_a_now if r.get("reannotate2_haiku_answer_ok")]
    still_fail   = [r for r in bucket_a_now if r.get("reannotate2_done")
                    and not r.get("reannotate2_haiku_answer_ok")]
    low_conf     = [r for r in bucket_a_now if r.get("reannotate2_confidence") == "low"]

    print(f"Bucket-A records processed : {len(bucket_a_now)}")
    print(f"Now pass haiku answer_ok   : {len(now_pass)}")
    print(f"Still fail                 : {len(still_fail)}")
    print(f"Marked confidence=low      : {len(low_conf)}")

    if low_conf:
        print(f"\n  Low-confidence IDs: {', '.join(r['id'] for r in low_conf)}")

    if still_fail:
        print(f"\n  Still-failing IDs ({len(still_fail)}):")
        for r in sorted(still_fail, key=lambda x: x["id"]):
            conf = r.get("reannotate2_confidence", "?")
            print(f"    {r['id']}  [conf={conf}]  {r.get('reannotate2_haiku_notes','')[:80]}")

    sonnet_cost = total_in  * SONNET_IN  + total_out  * SONNET_OUT
    haiku_cost  = haiku_in  * HAIKU_IN   + haiku_out  * HAIKU_OUT
    print(f"\nEstimated cost:")
    print(f"  Sonnet  ({total_in:,} in / {total_out:,} out)  : ${sonnet_cost:.4f}")
    print(f"  Haiku   ({haiku_in:,} in / {haiku_out:,} out)   : ${haiku_cost:.4f}")
    print(f"  Total                               : ${sonnet_cost + haiku_cost:.4f}")


if __name__ == "__main__":
    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    main()
