# mmrag-eval v0.1 Sample Dataset

A sample dataset of 50 annotated image–query pairs for evaluating multimodal RAG grounding quality.
All images are sourced from [Wikimedia Commons](https://commons.wikimedia.org/) under CC BY-SA 4.0.

---

## Purpose

This dataset provides a small, human-inspectable benchmark for the three mmrag-eval metrics:

| Metric | What it tests |
|---|---|
| `grounding_fidelity` | Does the generated answer stay within what the image actually shows? |
| `retrieval_quality` | Is the correct image retrieved at high rank (nDCG@K, Recall@K)? |
| `diversity` | Are the retrieved images visually distinct from each other? |

---

## Dataset Schema

Each record in `dataset.json` has the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record identifier (`mmrag-v0-NNN`) |
| `query` | string | Natural language question about the image |
| `query_type` | string | `"factual"` or `"visual_description"` |
| `image_path` | string | Relative path to the image file |
| `image_filename` | string | Bare filename |
| `reference_answer` | string | 2–4 sentence ground-truth answer grounded in the image |
| `grounding_labels` | list[str] | Image paths that are ground-truth evidence for the query |
| `wikimedia_url` | string | Full Wikimedia CDN URL of the original image |
| `license` | string | Image license (`CC BY-SA 4.0`) |
| `source` | string | `"Wikimedia Commons"` |
| `needs_review` | bool | Optional; `true` for records flagged for manual verification |

### Query types

- **factual** — asks about a specific named concept, count, label, or relationship directly visible in the diagram. The answer must be verifiable from the image alone without prior domain knowledge.
- **visual_description** — open-ended, asks what is shown. The answer describes visible components without requiring external facts.

---

## Category Breakdown

| Category | Count | Description |
|---|---|---|
| `neural_networks` | 15 | CNN, transformer, RNN, encoder-decoder architectures |
| `classical_ml` | 10 | Decision trees, SVM, k-means, logistic regression, RL |
| `evaluation_metrics` | 10 | Confusion matrix, ROC, precision-recall, cross-validation |
| `systems_pipelines` | 10 | ETL/ELT, cloud, data warehouse, AutoML, anomaly detection |
| `statistical_concepts` | 5 | Normal distribution, box plots, scatter plots |
| **Total** | **50** | |

---

## Annotation Quality

All 50 records in v0.1 were manually reviewed by the dataset author using a purpose-built review tool that displayed each image alongside its query and reference answer side by side.

4 records were corrected during review:

- **mmrag-v0-021:** reference answer rewritten to name the exact regression methods visible in the image (Linear Regression, Logistic Regression, Survival Analysis Regression)
- **mmrag-v0-022:** reference answer updated to name all four test recommendations verbatim from the image (Chi-square / Fisher's Exact Test, Logistic regression, T-test / ANOVA, Pearson / Spearman Correlation)
- **mmrag-v0-026:** query and answer completely rewritten — the Wikimedia Commons filename ("Confusion matrix") did not match the image content; the image is a scatter plot of 5 models by false negative rate vs mean accuracy, and the original annotation described TP/FP/TN/FN cells that do not exist in the image
- **mmrag-v0-040:** answer corrected to describe the actual visible architecture (Public Cloud, Hybrid Cloud, and Private Cloud connected through an Internet node to SME servers) rather than IaaS/PaaS/SaaS service layers that are not present in the image

Record 026 represents the most important catch: the Wikimedia Commons filename did not match the image content, and the original annotation described diagram elements that do not exist in the image. In a grounding benchmark, a fabricated reference answer silently corrupts every fidelity score computed against that record. Manual review is the only reliable way to catch this class of error in image datasets.

---

## Query Type Breakdown

| Query type | Count |
|---|---|
| `factual` | 29 |
| `visual_description` | 21 |

---

## Loading the Dataset

```python
from mmrag_eval.dataset.loader import load_from_json

samples = load_from_json("data/sample/dataset.json")
print(len(samples))  # 50
```

Or directly:

```python
import json

with open("data/sample/dataset.json") as f:
    records = json.load(f)

# Filter by category (field not in MMRagSample — access raw JSON)
nn_records = [r for r in records if "neural_network" in r["image_filename"].lower()
              or r.get("category") == "neural_networks"]
```

---

## License and Attribution

All images are sourced from **Wikimedia Commons** and licensed under **CC BY-SA 4.0**
(Creative Commons Attribution-ShareAlike 4.0 International).

The `wikimedia_url` field in each record links to the original file on Wikimedia Commons,
where individual attribution information and exact license terms can be found.

You are free to use, share, and adapt these images for any purpose, including commercial,
as long as you give appropriate credit and distribute derivatives under the same license.

The query text and reference answers in `dataset.json` are original annotations by the
mmrag-eval project and are released under the same MIT license as the rest of the repository.

---

## Roadmap

| Version | Samples | Status | Notes |
|---|---|---|---|
| **v0.1** | 50 | ✅ Released | Wikimedia Commons images, CC BY-SA 4.0, all 50 records manually reviewed by author |
| v0.2 | 200 | Planned | Human-verified grounding labels; multi-image queries |
| v1.0 | 1000 | Planned | HuggingFace Hub release; public leaderboard |

---

## Known Limitations (v0.1)

- All reference answers were based on direct visual inspection of each image; no answers were generated from filenames alone.
- Images are from a single source (Wikimedia Commons) and skew toward English-language diagrams, with a small number in Hindi, Malay, and Spanish.
