"""Tests for mmrag-eval metrics. All tests use synthetic data — no network calls."""
from __future__ import annotations

import io
import math
from unittest.mock import patch

import pytest
from PIL import Image

from mmrag_eval.metrics import diversity, grounding_fidelity, retrieval_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    img = Image.new("RGB", size, color=color)
    return img


def _make_checkerboard(color_a: tuple, color_b: tuple, size: int = 64, cell: int = 8) -> Image.Image:
    """Two-color checkerboard — gives a structurally distinct pHash."""
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = color_a if ((x // cell) + (y // cell)) % 2 == 0 else color_b
    return img


def _save_image(tmp_path, name: str, color: tuple[int, int, int]) -> str:
    p = tmp_path / name
    _make_image(color).save(p)
    return str(p)


# ---------------------------------------------------------------------------
# retrieval_quality
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_retrieval(self):
        retrieved = ["img_a", "img_b", "img_c"]
        relevant = ["img_a", "img_b", "img_c"]
        assert retrieval_quality.ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_no_hits(self):
        assert retrieval_quality.ndcg_at_k(["x", "y"], ["a", "b"], k=2) == pytest.approx(0.0)

    def test_partial_hit_ranking(self):
        # First hit at position 1 (best) vs position 2
        score_top = retrieval_quality.ndcg_at_k(["a", "x"], ["a"], k=2)
        score_low = retrieval_quality.ndcg_at_k(["x", "a"], ["a"], k=2)
        assert score_top > score_low

    def test_empty_relevant(self):
        assert retrieval_quality.ndcg_at_k(["a"], [], k=1) == pytest.approx(0.0)


class TestRecallAtK:
    def test_full_recall(self):
        assert retrieval_quality.recall_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert retrieval_quality.recall_at_k(["x"], ["a"], k=1) == pytest.approx(0.0)

    def test_partial_recall(self):
        assert retrieval_quality.recall_at_k(["a", "x"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert retrieval_quality.recall_at_k(["a"], [], k=1) == pytest.approx(0.0)


class TestRetrievalScoreDict:
    def test_returns_required_keys(self):
        result = retrieval_quality.score(["a"], ["a"], k=1)
        assert set(result.keys()) == {"ndcg_at_k", "recall_at_k", "k"}

    def test_k_stored(self):
        result = retrieval_quality.score([], [], k=7)
        assert result["k"] == 7


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------

class TestDiversityScore:
    def test_single_image_is_fully_diverse(self, tmp_path):
        img = _save_image(tmp_path, "a.png", (255, 0, 0))
        assert diversity.diversity_score([img]) == pytest.approx(1.0)

    def test_identical_images_score_zero(self, tmp_path):
        a = _save_image(tmp_path, "a.png", (0, 128, 0))
        b = _save_image(tmp_path, "b.png", (0, 128, 0))
        score = diversity.diversity_score([a, b], threshold=100)
        assert score == pytest.approx(0.0)

    def test_distinct_images_score_one(self, tmp_path):
        # Solid-color images have identical pHashes (all DCT energy at DC).
        # Use structurally different checkerboards so the hashes actually differ.
        a_img = _make_checkerboard((255, 0, 0), (0, 0, 0))
        b_img = _make_checkerboard((0, 0, 255), (255, 255, 255))
        a = str(tmp_path / "checker_rb.png"); a_img.save(a)
        b = str(tmp_path / "checker_bw.png"); b_img.save(b)
        score = diversity.diversity_score([a, b], threshold=1)
        assert score == pytest.approx(1.0)

    def test_score_dict_keys(self, tmp_path):
        img = _save_image(tmp_path, "x.png", (10, 20, 30))
        result = diversity.score([img])
        assert set(result.keys()) == {"diversity_score", "num_images", "duplicate_threshold"}


# ---------------------------------------------------------------------------
# grounding_fidelity
# ---------------------------------------------------------------------------

class TestGroundingFidelityWithCustomFn:
    """Tests using a deterministic stub — no CLIP download required."""

    def _stub(self, image_path: str, text: str) -> float:
        return 0.75

    def test_score_uses_grounding_fn(self, tmp_path):
        img = _save_image(tmp_path, "img.png", (100, 100, 100))
        result = grounding_fidelity.score(img, "some answer", grounding_fn=self._stub)
        assert result["grounding_fidelity"] == pytest.approx(0.75)
        assert result["method"] == "custom"

    def test_relative_fidelity_computed(self, tmp_path):
        img = _save_image(tmp_path, "img.png", (100, 100, 100))
        result = grounding_fidelity.score(
            img, "answer", reference_answer="ref", grounding_fn=self._stub
        )
        assert "reference_fidelity" in result
        assert result["relative_fidelity"] == pytest.approx(1.0)

    def test_no_reference_answer_omits_keys(self, tmp_path):
        img = _save_image(tmp_path, "img.png", (100, 100, 100))
        result = grounding_fidelity.score(img, "answer", grounding_fn=self._stub)
        assert "reference_fidelity" not in result
        assert "relative_fidelity" not in result

    def test_method_is_clip_by_default(self, tmp_path):
        img = _save_image(tmp_path, "img.png", (100, 100, 100))
        with patch("mmrag_eval.metrics.grounding_fidelity.clip_similarity", return_value=0.6):
            result = grounding_fidelity.score(img, "text")
        assert result["method"] == "clip"
        assert result["grounding_fidelity"] == pytest.approx(0.6)
