#!/usr/bin/env python3
"""
Inverse Retrieval Task (3 candidates):
Given properties -> identify the correct CIF among 3 candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from experiments.generating_from_cif.cif_to_properties_zeroshot import (
    available_r_values,
    compute_properties_from_xyz,
    iter_materials,
    run_openrouter,
    split_for_r,
)

load_dotenv()

NUM_CANDIDATES = 3


def _format_float(value: Any, ndigits: int = 5) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return "N/A"


def make_inverse_prompt(
    target_props: Dict[str, Any],
    candidates: List[Tuple[str, str]],
) -> str:
    candidate_text = ""
    for idx, (name, text) in enumerate(candidates, start=1):
        candidate_text += f"--- Candidate {idx} ({name}) ---\n{text}\n\n"

    return (
        "I have a nanoparticle with these properties:\n"
        f"Mass: {_format_float(target_props.get('mass_amu'))} amu\n"
        f"Density: {_format_float(target_props.get('density'))}\n"
        f"Mean NN Dist: {_format_float(target_props.get('mean_nn_distance'))}\n\n"
        "Which of the following Crystal Files (CIF) best matches these properties?\n"
        "Think step-by-step, then return ONLY valid JSON in this format:\n"
        '{ "candidate_index": <int>, "reasoning": "<brief reasoning>" }\n\n'
        f"{candidate_text}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inverse retrieval task: properties -> CIF (3 candidates)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build inverse prompts and gold labels.")
    build.add_argument(
        "--cif-root",
        type=Path,
        default=Path("scalar/unit_cells"),
        help="Root directory containing CIF files.",
    )
    build.add_argument(
        "--xyz-root",
        type=Path,
        default=Path("scalar/quaternions"),
        help="Root directory containing nanoparticle XYZ folders.",
    )
    build.add_argument(
        "--r-values",
        type=str,
        default="",
        help="Comma-separated list of R values to include (e.g. 6,7,8).",
    )
    build.add_argument(
        "--max-materials",
        type=int,
        default=0,
        help="Limit number of materials (0 = no limit).",
    )
    build.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling.",
    )
    build.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated prompts per (material, R).",
    )
    build.add_argument(
        "--prompts-out",
        type=Path,
        default=Path("results/task_inverse_llm/3cand/prompts.jsonl"),
        help="Output JSONL with prompts.",
    )
    build.add_argument(
        "--gold-out",
        type=Path,
        default=Path("results/task_inverse_llm/3cand/gold.jsonl"),
        help="Output JSONL with ground-truth labels.",
    )

    run = sub.add_parser("run", help="Run inverse prompts on OpenRouter.")
    run.add_argument(
        "--prompts",
        type=Path,
        default=Path("results/task_inverse_llm/3cand/prompts.jsonl"),
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
        default=Path("results/task_inverse_llm/3cand"),
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
        default="SCALAR inverse retrieval (3 candidates)",
        help="Optional X-Title header for OpenRouter.",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for OpenRouter calls per model.",
    )
    return parser.parse_args()


def build_inverse_tasks(args: argparse.Namespace) -> int:
    cif_root = args.cif_root
    xyz_root = args.xyz_root
    r_values: List[int] = []
    if args.r_values:
        r_values = [int(x.strip()) for x in args.r_values.split(",") if x.strip()]

    materials = list(iter_materials(cif_root, xyz_root))
    if not materials:
        print(
            f"[ERROR] No materials found under {cif_root} matching {xyz_root}",
            file=sys.stderr,
        )
        return 1

    if len(materials) < NUM_CANDIDATES:
        print("[ERROR] Not enough materials to build candidate sets.", file=sys.stderr)
        return 1

    if args.max_materials and args.max_materials > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            len(materials),
            size=min(args.max_materials, len(materials)),
            replace=False,
        )
        materials = [materials[i] for i in sorted(idx)]

    cif_cache: Dict[str, str] = {}
    for material, cif_path, _ in materials:
        cif_cache[material] = cif_path.read_text(encoding="utf-8", errors="ignore")

    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    args.gold_out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    total = 0
    skipped = 0

    with (
        args.prompts_out.open("w", encoding="utf-8") as pf,
        args.gold_out.open("w", encoding="utf-8") as gf,
    ):
        for material, _, xyz_dir in tqdm(
            materials, desc="Building inverse prompts", unit="material"
        ):
            available = r_values or available_r_values(xyz_dir)
            if not available:
                continue

            for r_val in available:
                if split_for_r(r_val) == "train":
                    continue
                xyz_path = xyz_dir / f"R{r_val}" / "xyz" / "rot_0.xyz"
                if not xyz_path.exists():
                    skipped += 1
                    continue

                gt = compute_properties_from_xyz(xyz_path)
                if (
                    gt.get("mass_amu") is None
                    or gt.get("density") is None
                    or gt.get("mean_nn_distance") is None
                ):
                    skipped += 1
                    continue

                distractors = [m for m, _, _ in materials if m != material]
                if len(distractors) < NUM_CANDIDATES - 1:
                    skipped += 1
                    continue
                sampled = list(
                    rng.choice(distractors, size=NUM_CANDIDATES - 1, replace=False)
                )

                candidates = [(material, cif_cache[material])]
                candidates += [(m, cif_cache[m]) for m in sampled]
                rng.shuffle(candidates)

                correct_index = 1 + [name for name, _ in candidates].index(material)
                prompt = make_inverse_prompt(gt, candidates)

                item_id = f"{material}__R{int(r_val)}"
                gf.write(
                    json.dumps(
                        {
                            "id": item_id,
                            "material": material,
                            "r_value": int(r_val),
                            "split": split_for_r(r_val),
                            "correct_index": correct_index,
                            "candidates": [name for name, _ in candidates],
                            "ground_truth": gt,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

                for rep in range(1, args.repeats + 1):
                    rep_id = f"{item_id}__rep{rep}"
                    pf.write(
                        json.dumps(
                            {
                                "id": rep_id,
                                "base_id": item_id,
                                "rep": rep,
                                "material": material,
                                "r_value": int(r_val),
                                "split": split_for_r(r_val),
                                "prompt": prompt,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    total += 1

    print(f"[OK] Wrote {total} prompts to {args.prompts_out}")
    if skipped:
        print(f"[WARN] Skipped {skipped} items (missing xyz or properties)")
    print(f"[OK] Wrote gold labels to {args.gold_out}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build_inverse_tasks(args)
    if args.command == "run":
        return run_openrouter(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
