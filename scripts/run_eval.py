#!/usr/bin/env python3
"""CLI entry point for mmrag-eval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate multimodal RAG grounding quality with mmrag-eval."
    )
    p.add_argument("dataset", help="Path to local JSON dataset file")
    p.add_argument(
        "--retrieved",
        required=True,
        help="Path to JSON file mapping each sample index to its retrieved image paths",
    )
    p.add_argument(
        "--answers",
        required=True,
        help="Path to JSON file containing generated answers (list, same order as dataset)",
    )
    p.add_argument("--k", type=int, default=5, help="Cutoff for retrieval metrics (default: 5)")
    p.add_argument(
        "--diversity-threshold",
        type=int,
        default=10,
        help="Hamming distance threshold for near-duplicate detection (default: 10)",
    )
    p.add_argument(
        "--output", default=None, help="Write JSON results to this file (default: stdout)"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from mmrag_eval import evaluate
    from mmrag_eval.dataset.loader import load_from_json

    samples = load_from_json(args.dataset)

    with open(args.retrieved) as f:
        retrieved_images = json.load(f)

    with open(args.answers) as f:
        generated_answers = json.load(f)

    results = evaluate(
        samples=samples,
        retrieved_images=retrieved_images,
        generated_answers=generated_answers,
        k=args.k,
        diversity_threshold=args.diversity_threshold,
    )

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
