#!/usr/bin/env python3
"""
Benchmark Task 1: CIF -> nanoparticle property prediction at target R values.

This script builds LLM prompts from CIF files and computes ground-truth
properties from the corresponding nanoparticle XYZ (rot_0.xyz).

Outputs:
  - prompts JSONL: input prompts for any LLM
  - gold JSONL: ground-truth properties for scoring

Scoring:
  - Compare model JSON outputs against gold for each (material, R)
  - Exact match for discrete fields, MAE/relative errors for floats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from ase.data import atomic_masses
from ase.io import read as ase_read
from scipy.spatial import ConvexHull, cKDTree
from dotenv import load_dotenv
from experiments.benchmark_models import OpenRouterModelRegistry
from tqdm import tqdm

load_dotenv()

R_DIR_RE = re.compile(r"^R(\d+)$")
REP_ID_RE = re.compile(r"^(?P<base>.+)__rep(?P<rep>\d+)$")


# Split mapping used for Task 1 labels
R_SPLITS = {
    "ID": [13, 15, 17, 20, 24, 27],
    "OOD": [10, 11, 29, 30],
}

PROMPT_TEMPLATE = """You are given a crystal CIF and multiple target nanoparticle radii R (Å).
Do NOT output XYZ or coordinates. For each target R, give me the following properties for the nanoparticle.
Use exactly 5 decimal places for all floating-point values.
Return ONLY valid JSON in this format:
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

Target R values (with split labels):
{r_list}

JSON:"""

FEWSHOT_TEMPLATE = """You are given a crystal CIF and multiple target nanoparticle radii R (Å).
Do NOT output XYZ or coordinates. Use the provided examples as guidance. Give me the following properties for the nanoparticle.
Use exactly 5 decimal places for all floating-point values.
Return ONLY valid JSON in this format:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and score LLM benchmark tasks for CIF -> properties."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build prompts and gold labels.")
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
        help="Random seed for material sampling.",
    )
    build.add_argument(
        "--prompts-out",
        type=Path,
        default=Path("results/task_1_llm/zeroshot/prompts.jsonl"),
        help="Output JSONL with prompts.",
    )
    build.add_argument(
        "--gold-out",
        type=Path,
        default=Path("results/task_1_llm/zeroshot/gold.jsonl"),
        help="Output JSONL with ground-truth labels.",
    )
    build.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated prompts per (material, R) for ID/OOD.",
    )

    score = sub.add_parser("score", help="Score model outputs against gold labels.")
    score.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Gold JSONL created by build.",
    )
    score.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predictions JSONL with fields: id and prediction (JSON or dict).",
    )
    score.add_argument(
        "--out",
        type=Path,
        default=Path("results/task_1_llm/score.json"),
        help="Where to write summary JSON (also printed).",
    )

    run = sub.add_parser("run", help="Run prompts through OpenRouter.")
    run.add_argument(
        "--prompts",
        type=Path,
        default=Path("results/task_1_llm/zeroshot/prompts.jsonl"),
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
        default=Path("results/task_1_llm/zeroshot"),
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
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for OpenRouter calls per model.",
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
        default="SCALAR CIF->properties",
        help="Optional X-Title header for OpenRouter.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def available_r_values(material_dir: Path) -> List[int]:
    r_vals = []
    for child in material_dir.iterdir():
        if not child.is_dir():
            continue
        m = R_DIR_RE.match(child.name)
        if m:
            r_vals.append(int(m.group(1)))
    return sorted(set(r_vals))


def split_for_r(r_val: int) -> str:
    if r_val in R_SPLITS["ID"]:
        return "ID"
    if r_val in R_SPLITS["OOD"]:
        return "OOD"
    return "train"


def _round_float(value: Optional[float], ndigits: int = 5) -> Optional[float]:
    if value is None:
        return None
    return float(round(value, ndigits))


def compute_properties_from_xyz(xyz_path: Path) -> Dict[str, Any]:
    atoms = ase_read(str(xyz_path))
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    num_atoms = len(symbols)

    composition: Dict[str, int] = {}
    for s in symbols:
        composition[s] = composition.get(s, 0) + 1

    if num_atoms >= 2:
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        nn_dists = dists[:, 1]
        min_nn = float(np.min(nn_dists))
        mean_nn = float(np.mean(nn_dists))
        median_nn = float(np.median(nn_dists))
    else:
        min_nn = None
        mean_nn = None
        median_nn = None

    mass_amu = float(np.sum(atomic_masses[numbers]))

    volume = None
    density = None
    if num_atoms >= 4:
        try:
            hull = ConvexHull(positions)
            volume = float(hull.volume)  # A^3
            if volume > 0:
                mass_g = mass_amu * 1.66053906660e-24
                density = float(mass_g / (volume * 1.0e-24))  # g/cm^3
        except Exception:
            volume = None
            density = None

    return {
        "num_atoms": int(num_atoms),
        "composition": composition,
        "min_nn_distance": _round_float(min_nn),
        "mean_nn_distance": _round_float(mean_nn),
        "median_nn_distance": _round_float(median_nn),
        "mass_amu": _round_float(mass_amu),
        "convex_hull_volume": _round_float(volume),
        "density": _round_float(density),
    }


def make_prompt(material: str, cif_text: str, r_values: List[int]) -> str:
    r_lines = [f"- R{r}: split={split_for_r(r)}" for r in r_values]
    r_list = "\n".join(r_lines)
    return PROMPT_TEMPLATE.format(
        material_id=material,
        cif_text=cif_text.strip(),
        r_list=r_list,
    )


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
    return FEWSHOT_TEMPLATE.format(
        material_id=material,
        cif_text=cif_text.strip(),
        examples_json=examples_json,
        r_list=r_list,
    )


def iter_materials(cif_root: Path, xyz_root: Path) -> Iterable[Tuple[str, Path, Path]]:
    for cif_path in sorted(cif_root.glob("*.cif")):
        material = cif_path.stem
        xyz_dir = xyz_root / material
        if not xyz_dir.exists():
            continue
        yield material, cif_path, xyz_dir


def build_tasks(args: argparse.Namespace) -> int:
    cif_root = args.cif_root
    xyz_root = args.xyz_root
    r_values = []
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
            len(materials), size=min(args.max_materials, len(materials)), replace=False
        )
        materials = [materials[i] for i in sorted(idx)]

    prompts_out = args.prompts_out
    gold_out = args.gold_out
    prompts_out.parent.mkdir(parents=True, exist_ok=True)
    gold_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    prompt_count = 0
    skipped = 0
    with (
        prompts_out.open("w", encoding="utf-8") as pf,
        gold_out.open("w", encoding="utf-8") as gf,
    ):
        for material, cif_path, xyz_dir in tqdm(
            materials, desc="Building prompts", unit="material"
        ):
            r_list = r_values or available_r_values(xyz_dir)
            if not r_list:
                continue
            cif_text = read_text(cif_path)
            items = []
            for r in r_list:
                if split_for_r(r) == "train":
                    continue
                xyz_path = xyz_dir / f"R{r}" / "xyz" / "rot_0.xyz"
                if not xyz_path.exists():
                    skipped += 1
                    continue

                ground_truth = compute_properties_from_xyz(xyz_path)
                items.append(
                    {
                        "r_value": int(r),
                        "split": split_for_r(r),
                        "xyz_path": str(xyz_path),
                        "ground_truth": ground_truth,
                    }
                )

            if not items:
                continue

            prompt = make_prompt(material, cif_text, [i["r_value"] for i in items])
            gf.write(
                json.dumps(
                    {
                        "id": material,
                        "material": material,
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
                            "prompt": prompt,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                total += 1
                prompt_count += 1

    print(f"[OK] Wrote {prompt_count} prompts to {prompts_out}")
    print(f"[OK] Wrote {total} gold items to {gold_out}")
    if skipped:
        print(f"[WARN] Skipped {skipped} items missing rot_0.xyz")
    return 0


def build_fewshot_tasks(args: argparse.Namespace, shots: int) -> int:
    cif_root = args.cif_root
    xyz_root = args.xyz_root
    r_values = []
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
            len(materials), size=min(args.max_materials, len(materials)), replace=False
        )
        materials = [materials[i] for i in sorted(idx)]

    prompts_out = args.prompts_out
    gold_out = args.gold_out
    prompts_out.parent.mkdir(parents=True, exist_ok=True)
    gold_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    prompt_count = 0
    skipped = 0
    rng = np.random.default_rng(args.seed)

    with (
        prompts_out.open("w", encoding="utf-8") as pf,
        gold_out.open("w", encoding="utf-8") as gf,
    ):
        for material, cif_path, xyz_dir in tqdm(
            materials, desc="Building few-shot prompts", unit="material"
        ):
            available = r_values or available_r_values(xyz_dir)
            if not available:
                continue
            train_rs = [r for r in available if split_for_r(r) == "train"]
            if len(train_rs) < shots:
                skipped += 1
                continue
            cif_text = read_text(cif_path)

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
                example_map[r] = {
                    "split": split_for_r(r),
                    **gt,
                }

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

    print(f"[OK] Wrote {prompt_count} prompts to {prompts_out}")
    print(f"[OK] Wrote {total} gold items to {gold_out}")
    if skipped:
        print(
            f"[WARN] Skipped {skipped} materials (insufficient train examples or missing xyz)"
        )
    return 0


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _flatten_gold_items(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    gold_items: Dict[str, Dict[str, Any]] = {}
    for item in load_jsonl(gold_path):
        if "items" in item:
            material = item.get("id") or item.get("material")
            for entry in item["items"]:
                r_val = entry["r_value"]
                item_id = f"{material}__R{r_val}"
                gt = dict(entry["ground_truth"])
                gt["split"] = entry.get("split")
                gold_items[item_id] = gt
        else:
            item_id = item["id"]
            gt = dict(item["ground_truth"])
            if "split" in item:
                gt["split"] = item["split"]
            gold_items[item_id] = gt
    return gold_items


def _base_id(item_id: str) -> str:
    m = REP_ID_RE.match(item_id)
    if m:
        return m.group("base")
    return item_id


def _parse_r_key(key: str) -> Optional[int]:
    if key is None:
        return None
    key = str(key).strip()
    if key.startswith("R") and key[1:].isdigit():
        return int(key[1:])
    if key.isdigit():
        return int(key)
    return None


def _flatten_predictions(pred_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    preds: Dict[str, List[Dict[str, Any]]] = {}
    for item in load_jsonl(pred_path):
        item_id = item.get("id")
        if not item_id:
            continue
        base_id = item.get("base_id") or _base_id(item_id)
        pred = item.get("prediction") or item.get("output") or item.get("response")
        if isinstance(pred, str):
            pred = parse_json_from_text(pred)
        if not isinstance(pred, dict):
            continue

        results = pred.get("results")
        if isinstance(results, dict):
            for r_key, r_pred in results.items():
                r_val = _parse_r_key(r_key)
                if r_val is None or not isinstance(r_pred, dict):
                    continue
                rid = f"{base_id}__R{r_val}"
                preds.setdefault(rid, []).append(r_pred)
            continue

        r_val = pred.get("r_value")
        if r_val is not None:
            rid = f"{base_id}__R{int(r_val)}"
            preds.setdefault(rid, []).append(pred)
        else:
            preds.setdefault(base_id, []).append(pred)
    return preds


def score_predictions(args: argparse.Namespace) -> int:
    gold_items = _flatten_gold_items(args.gold)
    gold_items = {k: v for k, v in gold_items.items() if v.get("split") != "train"}
    preds = _flatten_predictions(args.predictions)

    numeric_fields = [
        "min_nn_distance",
        "mean_nn_distance",
        "median_nn_distance",
        "mass_amu",
        "convex_hull_volume",
        "density",
    ]
    discrete_fields = ["num_atoms", "split"]

    results = {
        "total": len(gold_items),
        "scored": 0,
        "missing_predictions": 0,
        "field_metrics": {},
    }

    field_errors: Dict[str, List[float]] = {k: [] for k in numeric_fields}
    field_rel_errors: Dict[str, List[float]] = {k: [] for k in numeric_fields}
    field_exact: Dict[str, List[int]] = {
        k: [] for k in discrete_fields + ["composition"]
    }  # 1/0

    # Track per-base-id errors for mean/std reporting
    per_id_errors: Dict[str, Dict[str, List[float]]] = {k: {} for k in numeric_fields}

    for item_id, gt in gold_items.items():
        pred_list = preds.get(item_id)
        if not pred_list:
            results["missing_predictions"] += 1
            continue
        for pred in pred_list:
            results["scored"] += 1

            # discrete: num_atoms
            for field in discrete_fields:
                field_exact[field].append(1 if pred.get(field) == gt.get(field) else 0)

            # composition exact match
            field_exact["composition"].append(
                1 if pred.get("composition") == gt.get("composition") else 0
            )

            # numeric fields
            for field in numeric_fields:
                gt_val = gt.get(field)
                pred_val = pred.get(field)
                if gt_val is None or pred_val is None:
                    continue
                try:
                    gt_f = float(gt_val)
                    pred_f = float(pred_val)
                except (TypeError, ValueError):
                    continue
                err = abs(pred_f - gt_f)
                field_errors[field].append(err)
                if gt_f != 0:
                    field_rel_errors[field].append(err / abs(gt_f))
                per_id_errors[field].setdefault(item_id, []).append(err)

    for field in discrete_fields + ["composition"]:
        vals = field_exact[field]
        results["field_metrics"][field] = {
            "accuracy": float(np.mean(vals)) if vals else None,
            "count": len(vals),
        }

    for field in numeric_fields:
        errs = field_errors[field]
        rels = field_rel_errors[field]
        per_id = per_id_errors[field]
        per_id_means = [float(np.mean(v)) for v in per_id.values() if v]
        per_id_stds = [float(np.std(v)) for v in per_id.values() if v]
        results["field_metrics"][field] = {
            "mae": float(np.mean(errs)) if errs else None,
            "rel_mae": float(np.mean(rels)) if rels else None,
            "per_id_mae_mean": float(np.mean(per_id_means)) if per_id_means else None,
            "per_id_mae_std": float(np.mean(per_id_stds)) if per_id_stds else None,
            "count": len(errs),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return 0


def _openrouter_request(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    referer: str,
    title: str,
    timeout_s: int = 120,
) -> str:
    if not api_key:
        raise RuntimeError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key."
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    parsed = json.loads(body)
    choices = parsed.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content", "") or ""


def run_openrouter(args: argparse.Namespace) -> int:
    prompts = list(load_jsonl(args.prompts))
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        models = OpenRouterModelRegistry.all()
    if not models:
        print("[ERROR] No models provided.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        safe_model = model.replace("/", "_").replace(":", "_")
        out_path = args.out_dir / f"predictions_{safe_model}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                future_map = {}
                for item in prompts:
                    fut = ex.submit(
                        _openrouter_request,
                        api_key=args.api_key,
                        model=model,
                        prompt=item["prompt"],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        referer=args.referer,
                        title=args.title,
                    )
                    future_map[fut] = item

                for fut in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc=f"Running {model}",
                    unit="prompt",
                ):
                    item = future_map[fut]
                    item_id = item.get("id", "")
                    base_id = item.get("base_id") or _base_id(item_id)
                    try:
                        response = fut.result()
                    except urllib.error.HTTPError as err:
                        response = f"[HTTPError {err.code}] {err.read().decode('utf-8', errors='ignore')}"
                    except Exception as err:  # noqa: BLE001 - surface any failure
                        response = f"[Error] {err}"

                    f.write(
                        json.dumps(
                            {
                                "id": item_id,
                                "base_id": base_id,
                                "model": model,
                                "response": response,
                                "prediction": parse_json_from_text(response),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

                    if args.sleep > 0:
                        time.sleep(args.sleep)

        print(f"[OK] Wrote {len(prompts)} predictions to {out_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build_tasks(args)
    if args.command == "score":
        return score_predictions(args)
    if args.command == "run":
        return run_openrouter(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
