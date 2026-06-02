from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MMRagSample:
    query: str
    image_path: str
    reference_answer: str
    grounding_labels: list[str]


def load_from_hf(path_or_name: str, split: str = "test") -> list[MMRagSample]:
    """Load from a HuggingFace dataset. Expected columns: query, image_path, reference_answer, grounding_labels."""
    from datasets import load_dataset

    ds = load_dataset(path_or_name, split=split)
    return [
        MMRagSample(
            query=row["query"],
            image_path=row["image_path"],
            reference_answer=row["reference_answer"],
            grounding_labels=row["grounding_labels"],
        )
        for row in ds
    ]


def load_from_json(path: str | Path) -> list[MMRagSample]:
    """Load from a local JSON file (list of dicts with the four required fields)."""
    with open(path) as f:
        data = json.load(f)
    return [MMRagSample(**item) for item in data]
