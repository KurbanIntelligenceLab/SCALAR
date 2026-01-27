#!/usr/bin/env python3
"""Build 1-shot prompts (train R examples) for CIF -> properties."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from experiments.generating_from_cif.cif_to_properties_zeroshot import (
    build_fewshot_tasks,
    run_openrouter,
)
from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 1-shot prompts for CIF -> nanoparticle properties."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build 1-shot prompts and gold labels.")
    run = sub.add_parser("run", help="Run 1-shot prompts on OpenRouter.")

    # cif/xyz roots and r-values are fixed defaults in the shared builder
    build.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for material sampling and example selection.",
    )
    build.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated prompts per material.",
    )
    build.add_argument(
        "--prompts-out",
        type=Path,
        default=Path("results/task_1_llm/1shot/prompts.jsonl"),
        help="Output JSONL with prompts.",
    )
    build.add_argument(
        "--gold-out",
        type=Path,
        default=Path("results/task_1_llm/1shot/gold.jsonl"),
        help="Output JSONL with ground-truth labels.",
    )

    run.add_argument(
        "--prompts",
        type=Path,
        default=Path("results/task_1_llm/1shot/prompts.jsonl"),
        help="Prompts JSONL created by build.",
    )
    run.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated OpenRouter model ids (defaults to registry).",
    )
    run.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/task_1_llm/1shot"),
        help="Directory to write predictions JSONL per model.",
    )
    run.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY).",
    )
    run.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Limit number of prompts to run (0 = all).",
    )
    run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max tokens for completion (0 = provider default/max).",
    )
    run.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between requests.",
    )
    run.add_argument(
        "--referer",
        type=str,
        default="",
        help="Optional HTTP Referer header for OpenRouter.",
    )
    run.add_argument(
        "--title",
        type=str,
        default="SCALAR CIF->properties (1-shot)",
        help="Optional X-Title header for OpenRouter.",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for OpenRouter calls per model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cif_root = Path("scalar/unit_cells")
    args.xyz_root = Path("scalar/quaternions")
    args.r_values = ""
    args.max_materials = 0
    if not hasattr(args, "repeats"):
        args.repeats = 5
    if args.command == "build":
        return build_fewshot_tasks(args, shots=1)
    if args.command == "run":
        return run_openrouter(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
