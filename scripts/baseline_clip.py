#!/usr/bin/env python3
"""
CLIP-based dense retrieval baseline for mmrag-eval.

Uses openai/clip-vit-base-patch32 — the same model as
mmrag_eval/metrics/grounding_fidelity.py — for consistency.
Embeds all 198 images once, caches embeddings to
results/clip_image_embeddings.pt, then ranks the full 198-image
corpus by cosine similarity for each query text.

Leave-one-out design: the ground-truth image stays in the corpus,
so a perfect retriever would rank it #1.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = REPO_ROOT / "data" / "combined" / "dataset.json"
RESULTS_DIR = REPO_ROOT / "results"
EMBED_CACHE = RESULTS_DIR / "clip_image_embeddings.pt"
OUTPUT_FILE = RESULTS_DIR / "clip_retrieved.json"
MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_K = 20
BATCH_SIZE = 16


def embed_images(model, processor, image_paths: list[Path]) -> torch.Tensor:
    all_feats = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in image_paths[i : i + BATCH_SIZE]]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            # transformers 5.x returns BaseModelOutputWithPooling from vision_model;
            # project through visual_projection to get the shared CLIP embedding space.
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            feats = model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats)
        print(f"  images embedded: {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)}", end="\r")
    print()
    return torch.cat(all_feats, dim=0)  # (N, D)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(DATA_JSON) as f:
        records = json.load(f)
    print(f"Records loaded: {len(records)}")

    image_paths = [REPO_ROOT / r["image_path"] for r in records]

    print(f"Loading {MODEL_NAME} ...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # ── Image embeddings (cached) ─────────────────────────────────────────────
    if EMBED_CACHE.exists():
        print(f"Loading cached embeddings from {EMBED_CACHE.name} ...")
        image_embeddings = torch.load(EMBED_CACHE, weights_only=True)
    else:
        print("Embedding all images ...")
        image_embeddings = embed_images(model, processor, image_paths)
        torch.save(image_embeddings, EMBED_CACHE)
        print(f"Embeddings cached → {EMBED_CACHE.name}")

    print(f"Image embedding matrix: {image_embeddings.shape}")  # expect (198, 512)

    # ── Per-query ranking ─────────────────────────────────────────────────────
    print("Ranking queries ...")
    retrieved: dict[str, list[str]] = {}

    for r in records:
        text_inputs = processor(text=[r["query"]], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_out = model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            text_feats = model.text_projection(text_out.pooler_output)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sims = (text_feats @ image_embeddings.T).squeeze(0)   # (198,)
        ranked_idx = sims.argsort(descending=True).tolist()

        # Paths stored relative to REPO_ROOT to match grounding_labels format
        retrieved[r["id"]] = [
            str(image_paths[j].relative_to(REPO_ROOT))
            for j in ranked_idx[:TOP_K]
        ]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(retrieved, f, indent=2)
    print(f"Saved → {OUTPUT_FILE.relative_to(REPO_ROOT)}")

    # ── Ground-truth rank spot-check ──────────────────────────────────────────
    print("\nGround-truth rank (first 5 records):")
    for r in records[:5]:
        gt = r["grounding_labels"][0]
        ranked = retrieved[r["id"]]
        rank = ranked.index(gt) + 1 if gt in ranked else f">{ TOP_K}"
        print(f"  {r['id']}  GT rank: {rank:>4}  {Path(gt).name}")


if __name__ == "__main__":
    main()
