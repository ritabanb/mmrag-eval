# Dataset Construction Pipeline

This directory contains the full pipeline used to build the mmrag-eval v0.2 dataset
(198 records) as described in Section IV of the IEEE Access paper. Scripts are
numbered by stage; run them in order within each stage.

## Cost

This pipeline makes real paid API calls across multiple providers. Anyone
reproducing it end-to-end should expect a small but real API cost:

- **Anthropic** (`ANTHROPIC_API_KEY`) — stages 2, 3a, and 4 (draft generation,
  Haiku validation, Sonnet reannotation passes)
- **OpenRouter** (`OPENROUTER_API_KEY`) — stage 3b fallback and stage 3c;
  GPT-4o-mini validation (stage 3c) cost approximately **$0.71** for the full
  198-record pass at the time of writing
- **Groq** (`GROQ_API_KEY`) — stage 3b primary (free tier, but rate-limited)

Running the pipeline from scratch is not free. Budget roughly $5–10 for the
full v0.2 scale depending on current provider pricing.

## Prerequisites

```bash
pip install anthropic requests Pillow
# For Groq validation: pip install groq
# For Gemini validation (optional, not used in paper): pip install google-genai
```

Required environment variables (set before running each stage):
- `ANTHROPIC_API_KEY` — stages 2, 3a, 4
- `OPENROUTER_API_KEY` — stages 3b (fallback), 3c
- `GROQ_API_KEY` — stage 3b (primary)
- `GEMINI_API_KEY` — stage 3d (optional, not used in paper)

## Stage 1 — Collection (`01_collection/`)

Scrapes Wikimedia Commons for CC-licensed ML/statistics/systems diagrams.
No API key required; uses the Wikimedia Commons public API with a polite
`User-Agent` header and 2-second delays between requests.

Collection was iterative — run these scripts in order:

1. `collect_v1.py` — initial 200-record sweep across 5 categories
2. `collect_v1_supplement.py` — fills short categories (also fixed a bug where
   `filetype:bitmap` was being treated as literal search text rather than a filter)
3. `collect_v1_topup.py` — fills `evaluation_metrics` (+9) and `systems_pipelines` (+13)
4. `cleanup_and_topup.py` — culls `quality_flag=cull/api_error` records; tops up `classical_ml` to 40
5. `collect_v1_micro3.py` — 3 additional `evaluation_metrics` images with focused search terms
6. `collect_v1_final7.py` — final 7 `evaluation_metrics` images
7. `topup_v2.py` — tops up `neural_networks` (+11), `evaluation_metrics` (+3), `systems_pipelines` (+1)

Output: `data/v1/candidates.json`, `data/v1/images/`

## Stage 2 — Draft Generation (`02_draft_generation/`)

`generate_drafts.py` — calls `claude-sonnet-4-6` with vision to write
`query`, `query_type`, `reference_answer`, and `quality_flag` for each image.
Resumes automatically if interrupted (skips records that already have a query).
Requires `ANTHROPIC_API_KEY`.

Output: updates `data/v1/candidates.json` in place

## Stage 3 — Dual-Model Validation (`03_validation/`)

Independent cross-validation of each annotation by models that did not generate
the draft. Run in this order:

1. `validate_with_haiku.py` — validator 1: `claude-haiku-4-5` adds `haiku_*`
   fields (`haiku_image_ok`, `haiku_query_ok`, `haiku_answer_ok`, `haiku_notes`).
   Requires `ANTHROPIC_API_KEY`.

2. `validate_drafts_groq.py` — validator 2a: Llama 3.2 90B via Groq adds
   `groq_*` fields. Requires `GROQ_API_KEY`.

3. `validate_drafts_openrouter.py` — validator 2b: fallback for records not
   reached by Groq (Groq's own quota was hit mid-run); writes to the same
   `groq_*` fields with `groq_provider="openrouter"` for provenance.
   Requires `OPENROUTER_API_KEY`.

4. `validate_drafts_gpt4omini.py` — validator 3: GPT-4o-mini via OpenRouter
   adds `gpt4omini_*` fields. Has a built-in $1.80 cost ceiling to prevent
   runaway spend. Requires `OPENROUTER_API_KEY`.

5. `validate_with_gemini.py` — **not used in paper's final pipeline.**
   Google AI Studio's free tier was limited to 20 requests per day, making it
   impractical to validate 198 records. Groq (and OpenRouter as a fallback)
   was used instead. Retained here for completeness. Requires `GEMINI_API_KEY`.

Records where both validator 1 (Haiku) and validator 3 (GPT-4o-mini) flagged
`answer_ok=False` were sent to Stage 4 for reannotation.

## Stage 4 — Reannotation (`04_reannotation/`)

Two-pass reannotation for doubly-flagged records (77 records entered this stage):

1. `reannotate_flagged.py` — pass 1: `claude-sonnet-4-6` rewrites the reference
   answer with reviewer feedback (Haiku and GPT-4o-mini notes) incorporated;
   `claude-haiku-4-5` rechecks the new answer. 17 of 77 records recovered.

2. `reannotate_bucket_a.py` — pass 2: fresh-start prompt for 46 remaining
   bucket-A records. Sonnet re-reads the image independently with no prior
   answer shown — only the type of error to avoid (from reviewer notes).
   9 more records recovered. 37 records that could not be reliably annotated
   after two passes were dropped.

Requires `ANTHROPIC_API_KEY`.

## Stage 5 — Merge & Build (`05_merge/`)

`build_final_dataset.py` — filters v1 survivors (148 records), applies
pass-2 answers where applicable (`reannotate2_answer` field), strips all
internal validation fields, merges with the v0 sample (50 records from
`data/sample/dataset.json`), and writes the public dataset.

Output: `data/combined/dataset.json`, `data/combined/images/` (198 records total)

This script includes reconciliation assertions: it will exit with an error if
the survivor count or total record count does not match the expected values,
preventing silent data corruption from pipeline changes.
