from __future__ import annotations

from typing import Callable, Optional

_clip_cache: dict[str, tuple] = {}


def _load_clip(model_name: str):
    if model_name not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        _clip_cache[model_name] = (model, processor)
    return _clip_cache[model_name]


def clip_similarity(
    image_path: str,
    text: str,
    model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """Cosine similarity between image and text embeddings, mapped to [0, 1] via sigmoid."""
    import torch
    from PIL import Image

    model, processor = _load_clip(model_name)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
        sim = torch.sigmoid(out.logits_per_image).item()
    return sim


def score(
    image_path: str,
    answer: str,
    reference_answer: Optional[str] = None,
    grounding_fn: Optional[Callable[[str, str], float]] = None,
    model_name: str = "openai/clip-vit-base-patch32",
) -> dict:
    """
    Score how well an answer is grounded in an image.

    grounding_fn: optional override (e.g. GPT-4V) with signature (image_path, text) -> float.
    When provided, CLIP is not called.
    """
    method = "custom" if grounding_fn else "clip"
    _score = grounding_fn if grounding_fn else lambda p, t: clip_similarity(p, t, model_name)

    result: dict = {
        "grounding_fidelity": _score(image_path, answer),
        "method": method,
    }

    if reference_answer is not None:
        ref = _score(image_path, reference_answer)
        result["reference_fidelity"] = ref
        result["relative_fidelity"] = result["grounding_fidelity"] / ref if ref > 0 else 0.0

    return result
