from __future__ import annotations

from pathlib import Path


def _phash(image_path: str):
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.open(image_path).convert("RGB"))


def diversity_score(image_paths: list[str], threshold: int = 10) -> float:
    """
    Score in [0, 1] penalizing near-duplicate images.
    Two images are near-duplicates when their perceptual hash Hamming distance < threshold.
    Returns 1.0 when all images are distinct, 0.0 when every pair is a duplicate.
    """
    n = len(image_paths)
    if n <= 1:
        return 1.0

    hashes = [_phash(p) for p in image_paths]
    total_pairs = n * (n - 1) / 2
    duplicate_pairs = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if hashes[i] - hashes[j] < threshold
    )
    return 1.0 - duplicate_pairs / total_pairs


def score(image_paths: list[str], threshold: int = 10) -> dict:
    """Return diversity score for a set of retrieved images."""
    return {
        "diversity_score": diversity_score(image_paths, threshold),
        "num_images": len(image_paths),
        "duplicate_threshold": threshold,
    }
