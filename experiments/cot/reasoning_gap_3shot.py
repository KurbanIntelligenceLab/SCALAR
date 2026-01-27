#!/usr/bin/env python3
"""
Chain-of-Thought Task (3-shot only):
Build/run 3-shot prompts for CIF -> properties using CoT reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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

CoT_TEMPLATE = """You are a crystallographer. Use explicit physics-based reasoning before giving final numbers.
Focus on the following steps:
1) Parse unit-cell parameters and estimate the unit-cell volume.
2) Determine stoichiometry and approximate bulk atomic/mass density.
3) Approximate nanoparticle volume from radius R and scale counts/mass/volume.
4) Use geometry/nearest-neighbor intuition to estimate distance statistics.

Do NOT output XYZ or coordinates. Use the provided examples as guidance.
Use exactly 5 decimal places for all floating-point values.
Provide brief, structured reasoning, then output the final JSON only.

JSON format:
{{
  "material": "{material_id}",
  "results": {{
    "R10": {{
      "num_atoms": <int>,
      "composition": {{"El": <count>, "...": <count>}},
      "min_nn_distance": <float>,
      "mean_nn_distance": <float>,
      "median_nn_distance": <float>,
      "mass_amu": <float>,
      "convex_hull_volume": <float>,
      "density": <float>
    }},
    "R11": {{ ... }}
  }}
}}

CIF:
<<<CIF
{cif_text}
CIF
>>>

Examples:
{examples_json}

Target R values:
{r_list}

JSON:"""


def make_fewshot_prompt(
    material: str,
    cif_text: str,
    example_map: Dict[int, Dict[str, Any]],
    target_r_values: List[int],
) -> str:
    r_lines = [f"- R{r}: split={split_for_r(r)}" for r in target_r_values]
    r_list = "\n".join(r_lines)
    examples_json = json.dumps(
        {f"R{r}": example_map[r] for r in sorted(example_map.keys())},
        indent=2,
        ensure_ascii=True,
    )
    return CoT_TEMPLATE.format(
        material_id=material,
        cif_text=cif_text.strip(),
        examples_json=examples_json,
        r_list=r_list,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought task (3-shot only) for CIF -> properties."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build 3-shot prompts and gold labels.")
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
        default=Path("results/task_1_cot_llm/3shot/prompts.jsonl"),
        help="Output JSONL with prompts.",
    )
    build.add_argument(
        "--gold-out",
        type=Path,
        default=Path("results/task_1_cot_llm/3shot/gold.jsonl"),
        help="Output JSONL with ground-truth labels.",
    )

    run = sub.add_parser("run", help="Run prompts on OpenRouter.")
    run.add_argument(
        "--prompts",
        type=Path,
        default=Path("results/task_1_cot_llm/3shot/prompts.jsonl"),
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
        default=Path("results/task_1_cot_llm/3shot"),
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
        default="SCALAR CIF->properties (3-shot CoT)",
        help="Optional X-Title header for OpenRouter.",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for OpenRouter calls per model.",
    )
    return parser.parse_args()


def build_cot(args: argparse.Namespace, shots: int = 3) -> int:
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

    if args.max_materials and args.max_materials > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            len(materials),
            size=min(args.max_materials, len(materials)),
            replace=False,
        )
        materials = [materials[i] for i in sorted(idx)]

    if args.prompts_out:
        args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    if args.gold_out:
        args.gold_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    prompt_count = 0
    skipped = 0
    rng = np.random.default_rng(args.seed)

    with (
        args.prompts_out.open("w", encoding="utf-8") as pf,
        args.gold_out.open("w", encoding="utf-8") as gf,
    ):
        for material, cif_path, xyz_dir in tqdm(
            materials,
            desc="Building CoT prompts",
            unit="material",
        ):
            available = r_values or available_r_values(xyz_dir)
            if not available:
                continue
            train_rs = [r for r in available if split_for_r(r) == "train"]
            if len(train_rs) < shots:
                skipped += 1
                continue

            target_rs = [r for r in available if split_for_r(r) != "train"]
            if not target_rs:
                skipped += 1
                continue

            example_rs = [
                int(r) for r in rng.choice(train_rs, size=shots, replace=False)
            ]
            example_map: Dict[int, Dict[str, Any]] = {}
            for r in example_rs:
                xyz_path = xyz_dir / f"R{r}" / "xyz" / "rot_0.xyz"
                if not xyz_path.exists():
                    continue
                gt = compute_properties_from_xyz(xyz_path)
                example_map[r] = {"split": split_for_r(r), **gt}

            if len(example_map) < shots:
                skipped += 1
                continue

            items: List[Dict[str, Any]] = []
            for target_r in target_rs:
                xyz_path = xyz_dir / f"R{target_r}" / "xyz" / "rot_0.xyz"
                if not xyz_path.exists():
                    skipped += 1
                    continue
                gt = compute_properties_from_xyz(xyz_path)
                items.append(
                    {
                        "r_value": int(target_r),
                        "split": split_for_r(target_r),
                        "xyz_path": str(xyz_path),
                        "ground_truth": gt,
                    }
                )

            if not items:
                skipped += 1
                continue

            cif_text = cif_path.read_text(encoding="utf-8", errors="ignore")
            prompt = make_fewshot_prompt(
                material,
                cif_text,
                example_map,
                [i["r_value"] for i in items],
            )

            gf.write(
                json.dumps(
                    {
                        "id": material,
                        "material": material,
                        "example_r_values": sorted(int(k) for k in example_map.keys()),
                        "items": items,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

            for rep in range(1, args.repeats + 1):
                rep_id = f"{material}__rep{rep}"
                pf.write(
                    json.dumps(
                        {
                            "id": rep_id,
                            "base_id": material,
                            "rep": rep,
                            "material": material,
                            "cif_path": str(cif_path),
                            "example_r_values": sorted(
                                int(k) for k in example_map.keys()
                            ),
                            "prompt": prompt,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                total += 1
                prompt_count += 1

    print(f"[OK] Wrote {prompt_count} prompts to {args.prompts_out}")
    print(f"[OK] Wrote {total} gold items to {args.gold_out}")
    if skipped:
        print(
            f"[WARN] Skipped {skipped} materials (insufficient train examples or missing xyz)"
        )
    return 0


def main() -> int:
    args = parse_args()
    args.cif_root = Path("scalar/unit_cells")
    args.xyz_root = Path("scalar/quaternions")
    args.r_values = ""
    args.max_materials = 0
    if not hasattr(args, "repeats"):
        args.repeats = 5
    if args.command == "build":
        return build_cot(args, shots=3)
    if args.command == "run":
        return run_openrouter(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
