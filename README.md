# mmrag-eval

[![CI](https://github.com/ritabanb/mmrag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/ritabanb/mmrag-eval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ritabanb/mmrag-eval/pulls)

**mmrag-eval** is an open-source benchmark for evaluating the grounding quality of multimodal Retrieval-Augmented Generation (RAG) systems — measuring not just whether a system retrieves images, but whether its generated answers stay within, and faithfully reflect, the visual evidence it retrieved.

---

## Problem Statement

Multimodal RAG systems retrieve images to ground their responses, but standard benchmarks only measure answer correctness against a reference string. This leaves two critical failure modes undetected:

1. **Hallucination** — answers that are plausible but not supported by the retrieved images.
2. **Redundancy** — retrieval pipelines that pad results with near-duplicate images, inflating apparent recall while reducing evidence diversity.

mmrag-eval provides three independent, composable metrics that together give a principled grounding score for any multimodal RAG system.

---

## Dataset

A sample dataset of 50 annotated image–query pairs is available on HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("ritaban-b/mmrag-eval", split="train")
```

→ [ritaban-b/mmrag-eval on HuggingFace](https://huggingface.co/datasets/ritaban-b/mmrag-eval)

All 50 records were manually reviewed by the author.
See the [dataset card](https://huggingface.co/datasets/ritaban-b/mmrag-eval) for schema, category breakdown, and annotation quality notes.

---

## Metrics

### 1. Grounding Fidelity (`grounding_fidelity`)

Measures how well a generated answer is grounded in the retrieved image rather than hallucinated. Uses CLIP image–text similarity as a proxy score. A higher score means the answer text is more consistent with the image's semantic content.

**Hook for GPT-4V:** pass a custom `grounding_fn(image_path, text) -> float` to replace CLIP with any vision-language judge.

### 2. Retrieval Quality (`retrieval_quality`)

Measures the standard IR effectiveness of the image retrieval step:

- **nDCG@K** — normalized Discounted Cumulative Gain at K: rewards retrieving relevant images at higher ranks.
- **Recall@K** — fraction of ground-truth relevant images recovered in the top-K results.

### 3. Diversity (`diversity`)

Penalizes retrieval pipelines that return near-duplicate images. Uses perceptual hashing (pHash) to detect visually similar images and returns a score in [0, 1], where 1.0 means all retrieved images are visually distinct.

---

## Quickstart

```bash
pip install git+https://github.com/ritabanb/mmrag-eval.git
```

```python
from mmrag_eval import evaluate
from mmrag_eval.dataset.loader import MMRagSample

samples = [
    MMRagSample(
        query="What is shown in the diagram?",
        image_path="data/fig1.png",
        reference_answer="A flowchart depicting the training pipeline.",
        grounding_labels=["data/fig1.png"],  # relevant images for this query
    )
]

retrieved_images = [["data/fig1.png", "data/fig2.png"]]
generated_answers = ["The diagram shows a machine learning training pipeline."]

results = evaluate(
    samples=samples,
    retrieved_images=retrieved_images,
    generated_answers=generated_answers,
    k=5,
)

print(results["aggregated"])
# {
#   "grounding_fidelity": 0.83,
#   "ndcg_at_k": 1.0,
#   "recall_at_k": 1.0,
#   "diversity_score": 1.0,
#   "num_samples": 1
# }
```

### CLI

```bash
python scripts/run_eval.py dataset.json \
  --retrieved retrieved.json \
  --answers answers.json \
  --k 5 \
  --output results.json
```

---

## Dataset Format

JSON list of objects with four required fields:

```json
[
  {
    "query": "string",
    "image_path": "path/to/image.png",
    "reference_answer": "string",
    "grounding_labels": ["path/to/relevant_image.png"]
  }
]
```

Also loadable from HuggingFace datasets:

```python
from mmrag_eval.dataset.loader import load_from_hf
samples = load_from_hf("your-hf-org/mmrag-dataset", split="test")
```

---

## Custom Grounding Function (GPT-4V)

```python
import openai, base64

def gpt4v_grounding(image_path: str, text: str) -> float:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": f"On a scale of 0–1, how well does this text describe only what is visible in the image? Text: '{text}'. Reply with a single float."},
            ],
        }],
        max_tokens=10,
    )
    return float(response.choices[0].message.content.strip())

results = evaluate(samples, retrieved_images, generated_answers, grounding_fn=gpt4v_grounding)
```

---

## Contributing

Contributions are welcome — especially new metrics, dataset loaders, and evaluation integrations.

1. Fork the repo and create a feature branch.
2. Add tests for any new metric in `tests/test_metrics.py`.
3. Run `pytest` and make sure all tests pass.
4. Open a pull request with a clear description of what you added and why.

Please open an issue first for significant changes so we can discuss the approach.

---

## Citation

If you use mmrag-eval in your research, please cite:

```bibtex
@software{mmrag_eval,
  author  = {Bhattacharya, Ritaban},
  title   = {{mmrag-eval}: Benchmark for Evaluating Grounding Quality in Multimodal RAG Systems},
  year    = {2026},
  url     = {https://github.com/ritabanb/mmrag-eval},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
