from __future__ import annotations

from typing import Callable, Optional

from mmrag_eval.dataset.loader import MMRagSample
from mmrag_eval.metrics import diversity, grounding_fidelity, retrieval_quality


def evaluate(
    samples: list[MMRagSample],
    retrieved_images: list[list[str]],
    generated_answers: list[str],
    k: int = 5,
    diversity_threshold: int = 10,
    grounding_fn: Optional[Callable[[str, str], float]] = None,
    clip_model: str = "openai/clip-vit-base-patch32",
) -> dict:
    """
    Run the full evaluation pipeline.

    Args:
        samples: Ground-truth samples from the dataset.
        retrieved_images: Per-sample list of retrieved image paths (in rank order).
        generated_answers: Per-sample generated answers to evaluate.
        k: Cutoff for retrieval metrics.
        diversity_threshold: Hamming distance threshold for near-duplicate detection.
        grounding_fn: Optional callable replacing CLIP; signature (image_path, text) -> float.
        clip_model: HuggingFace model ID for CLIP (ignored when grounding_fn is set).

    Returns:
        {"aggregated": {metric: mean, ...}, "per_sample": [per-sample dicts]}
    """
    per_sample = []
    for sample, retrieved, answer in zip(samples, retrieved_images, generated_answers):
        gf = grounding_fidelity.score(
            image_path=sample.image_path,
            answer=answer,
            reference_answer=sample.reference_answer,
            grounding_fn=grounding_fn,
            model_name=clip_model,
        )
        rq = retrieval_quality.score(
            retrieved_images=retrieved,
            relevant_images=sample.grounding_labels,
            k=k,
        )
        div = diversity.score(image_paths=retrieved, threshold=diversity_threshold)
        per_sample.append({**gf, **rq, **div})

    numeric_keys = [key for key, val in per_sample[0].items() if isinstance(val, float)]
    aggregated = {
        key: sum(s[key] for s in per_sample) / len(per_sample) for key in numeric_keys
    }
    aggregated["num_samples"] = len(samples)

    return {"aggregated": aggregated, "per_sample": per_sample}
