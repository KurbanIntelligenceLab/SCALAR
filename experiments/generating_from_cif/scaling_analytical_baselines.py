#!/usr/bin/env python3
"""
CIF-only baseline for nanoparticle property prediction.

Uses only CIF files and given R values: no gold, no XYZ, no other inputs.
Physics-based analytical formulas from CIF (unit cell, lattice, composition)
predict properties at each (material, R).

Usage:
  # All materials in CIF directory, given R values:
  python -m experiments.generating_from_cif.scaling_analytical_baselines run \\
    --cif-root scalar/unit_cells \\
    --r-values 10,11,13,15,17,20,24,27,29,30 \\
    --out results/predictions_analytical.jsonl

  # Single CIF, given R values:
  python -m experiments.generating_from_cif.scaling_analytical_baselines run \\
    --cif path/to/Ag_P-1.cif \\
    --r-values 10,13,20 \\
    --out results/predictions_analytical.jsonl

  # With gold: run baseline, then evaluate and report errors:
  python -m experiments.generating_from_cif.scaling_analytical_baselines run \\
    --cif-root scalar/unit_cells --r-values 10,11,...,30 \\
    --out results/predictions_analytical.jsonl \\
    --gold results/task_1_cot_llm/1shot/gold.jsonl \\
    [--score-out results/score_analytical.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    pdist = None
    warnings.warn("scipy not available – NN distance from CIF may use fallback")

try:
    from ase.io import read as ase_read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available – CIF parsing uses fallback")


def _round_float(value: Optional[float], ndigits: int = 5) -> Optional[float]:
    if value is None:
        return None
    return float(round(value, ndigits))


# -----------------------------------------------------------------------------
# Analytical baseline: CIF + R → properties (no other inputs)
# -----------------------------------------------------------------------------

ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95, "K": 39.10, "Ca": 40.08,
    "Sc": 44.96, "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Mn": 54.94,
    "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
    "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.97, "Br": 79.90,
    "Kr": 83.80, "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22,
    "Nb": 92.91, "Mo": 95.95, "Tc": 98.00, "Ru": 101.1, "Rh": 102.9,
    "Pd": 106.4, "Ag": 107.9, "Cd": 112.4, "In": 114.8, "Sn": 118.7,
    "Sb": 121.8, "Te": 127.6, "I": 126.9, "Xe": 131.3, "Cs": 132.9,
    "Ba": 137.3, "La": 138.9, "Ce": 140.1, "Pr": 140.9, "Nd": 144.2,
    "Pm": 145.0, "Sm": 150.4, "Eu": 152.0, "Gd": 157.3, "Tb": 158.9,
    "Dy": 162.5, "Ho": 164.9, "Er": 167.3, "Tm": 168.9, "Yb": 173.0,
    "Lu": 175.0, "Hf": 178.5, "Ta": 180.9, "W": 183.8, "Re": 186.2,
    "Os": 190.2, "Ir": 192.2, "Pt": 195.1, "Au": 197.0, "Hg": 200.6,
    "Tl": 204.4, "Pb": 207.2, "Bi": 209.0, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0,
    "Pa": 231.0, "U": 238.0, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
}


class AnalyticalBaseline:
    """
    Physics-based formulas from CIF only:

    - N_atoms ≈ (4/3 π R³) / V_cell * Z * packing
    - Convex hull volume ≈ (4/3) π R³
    - Mass from composition; density = mass/volume (g/cm³)
    - NN distances from unit-cell structure (constant in R).
    """

    def __init__(self) -> None:
        self.cif_data: Dict[str, Dict[str, Any]] = {}

    def load_cif(self, material: str, cif_path: Path) -> bool:
        if ASE_AVAILABLE and self._load_cif_ase(material, cif_path):
            return True
        return self._parse_cif_fallback(material, cif_path)

    def _load_cif_ase(self, material: str, cif_path: Path) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=r"ase\.spacegroup\.spacegroup",
                )
                atoms = ase_read(str(cif_path))
            cell = atoms.get_cell()
            volume = float(atoms.get_volume())
            symbols = list(atoms.get_chemical_symbols())
            comp: Dict[str, int] = defaultdict(int)
            for s in symbols:
                comp[s] += 1

            pos = atoms.get_positions()
            if len(pos) > 1 and pdist is not None:
                d = pdist(pos)
                nn = float(np.min(d)) if d.size else 2.5
            else:
                nn = 2.5

            self.cif_data[material] = {
                "volume": volume,
                "n_atoms_unit_cell": len(atoms),
                "composition": dict(comp),
                "nn_distance": nn,
                "lattice_params": cell.lengths().tolist(),
            }
            return True
        except Exception as e:
            warnings.warn(f"ASE CIF parse failed {cif_path}: {e}")
            return False

    def _parse_cif_fallback(self, material: str, cif_path: Path) -> bool:
        try:
            text = cif_path.read_text()
            a = b = c = 3.0
            for line in text.splitlines():
                if "_cell_length_a" in line:
                    m = re.search(r"[\d.]+", line.split()[-1])
                    if m:
                        a = float(m.group())
                elif "_cell_length_b" in line:
                    m = re.search(r"[\d.]+", line.split()[-1])
                    if m:
                        b = float(m.group())
                elif "_cell_length_c" in line:
                    m = re.search(r"[\d.]+", line.split()[-1])
                    if m:
                        c = float(m.group())

            vol = a * b * c
            elem = material.split("_")[0] if "_" in material else material
            self.cif_data[material] = {
                "volume": vol,
                "n_atoms_unit_cell": 4,
                "composition": {elem: 1},
                "nn_distance": a / np.sqrt(2),
                "lattice_params": [a, b, c],
            }
            return True
        except Exception as e:
            warnings.warn(f"Fallback CIF parse failed {material}: {e}")
            return False

    def predict(self, material: str, r_value: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if material not in self.cif_data:
            return out

        cif = self.cif_data[material]
        sphere_vol = (4 / 3) * np.pi * (r_value ** 3)
        v_cell = cif["volume"]
        z = cif["n_atoms_unit_cell"]
        packing = 0.74

        num_atoms = (sphere_vol / v_cell) * z * packing
        n = int(round(max(0, num_atoms)))
        out["num_atoms"] = n

        total_uc = sum(cif["composition"].values())
        comp: Dict[str, int] = {}
        remaining = n
        elems = list(cif["composition"].keys())
        for e in elems[:-1]:
            c = int(round(n * cif["composition"][e] / total_uc))
            c = max(0, min(c, remaining))
            comp[e] = c
            remaining -= c
        comp[elems[-1]] = max(0, remaining)
        out["composition"] = comp

        nn = cif["nn_distance"]
        out["min_nn_distance"] = _round_float(nn)
        out["mean_nn_distance"] = _round_float(nn)
        out["median_nn_distance"] = _round_float(nn)

        mass = sum(comp[e] * ATOMIC_MASSES.get(e, 100.0) for e in comp)
        out["mass_amu"] = _round_float(mass)
        out["convex_hull_volume"] = _round_float(sphere_vol)

        if sphere_vol > 0:
            density = mass * 1.66053906660 / sphere_vol
            out["density"] = _round_float(density)
        else:
            out["density"] = None

        return out


def props_to_scoreable(props: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in props.items():
        if k == "num_atoms" and v is not None:
            out[k] = int(v)
        elif k == "composition":
            out[k] = v
        elif v is not None and isinstance(v, (int, float)):
            out[k] = _round_float(float(v)) if isinstance(v, float) else v
        else:
            out[k] = v
    return out


def discover_cif_materials(cif_root: Path) -> List[tuple[str, Path]]:
    """Return [(material_id, cif_path), ...] from *.cif in cif_root."""
    out: List[tuple[str, Path]] = []
    for p in sorted(cif_root.glob("*.cif")):
        out.append((p.stem, p))
    return out


def run_cif_only(
    cif_paths: List[tuple[str, Path]],
    r_values: List[int],
    out_path: Path,
    verbose: bool = False,
) -> int:
    """
    Run CIF-only analytical baseline.
    Input: CIF files + R values. No gold, no XYZ, nothing else.
    """
    baseline = AnalyticalBaseline()
    for material, path in cif_paths:
        if not path.exists():
            if verbose:
                print(f"[skip] missing {path}", file=sys.stderr)
            continue
        baseline.load_cif(material, path)

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for material, _ in cif_paths:
            if material not in baseline.cif_data:
                continue
            results: Dict[str, Dict[str, Any]] = {}
            for r in r_values:
                pred = baseline.predict(material, float(r))
                if not pred:
                    continue
                pred = props_to_scoreable(pred)
                results[f"R{r}"] = pred
            if not results:
                continue
            rec = {
                "id": material,
                "base_id": material,
                "prediction": {"material": material, "results": results},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    return written


# -----------------------------------------------------------------------------
# Analysis: compare predictions to gold, report errors
# -----------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _flatten_gold(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    """Gold items as {material__Rr: {ground_truth fields, split}}; exclude train."""
    flat: Dict[str, Dict[str, Any]] = {}
    for item in _load_jsonl(gold_path):
        if "items" in item:
            material = item.get("id") or item.get("material")
            for entry in item["items"]:
                r_val = entry["r_value"]
                sid = f"{material}__R{r_val}"
                gt = dict(entry.get("ground_truth") or {})
                gt["split"] = entry.get("split")
                flat[sid] = gt
        else:
            sid = item.get("id", "")
            gt = dict(item.get("ground_truth") or {})
            if "split" in item:
                gt["split"] = item["split"]
            flat[sid] = gt
    return {k: v for k, v in flat.items() if v.get("split") != "train"}


def _parse_r_key(key: Any) -> Optional[int]:
    if key is None:
        return None
    s = str(key).strip()
    if s.startswith("R") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    return None


def _flatten_predictions(pred_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Predictions as {material__Rr: [pred_dict, ...]}."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for item in _load_jsonl(pred_path):
        base_id = item.get("base_id") or item.get("id")
        if not base_id:
            continue
        pred = item.get("prediction") or item.get("output") or item.get("response")
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except json.JSONDecodeError:
                continue
        if not isinstance(pred, dict):
            continue
        results = pred.get("results")
        if not isinstance(results, dict):
            continue
        for r_key, r_pred in results.items():
            r_val = _parse_r_key(r_key)
            if r_val is None or not isinstance(r_pred, dict):
                continue
            rid = f"{base_id}__R{r_val}"
            out.setdefault(rid, []).append(r_pred)
    return out


def _analyze_and_report_errors(
    gold_path: Path,
    pred_path: Path,
    score_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare predictions to gold; return metrics dict, print summary, optionally write JSON."""
    gold = _flatten_gold(gold_path)
    preds = _flatten_predictions(pred_path)

    numeric_fields = [
        "min_nn_distance",
        "mean_nn_distance",
        "median_nn_distance",
        "mass_amu",
        "convex_hull_volume",
        "density",
    ]
    discrete_fields = ["num_atoms", "split"]

    report: Dict[str, Any] = {
        "total": len(gold),
        "scored": 0,
        "missing_predictions": 0,
        "field_metrics": {},
    }
    field_errors: Dict[str, List[float]] = {k: [] for k in numeric_fields}
    field_rel_errors: Dict[str, List[float]] = {k: [] for k in numeric_fields}
    field_exact: Dict[str, List[int]] = {
        k: [] for k in discrete_fields + ["composition"]
    }
    per_id_errors: Dict[str, Dict[str, List[float]]] = {k: {} for k in numeric_fields}

    for item_id, gt in gold.items():
        pred_list = preds.get(item_id)
        if not pred_list:
            report["missing_predictions"] += 1
            continue
        for pred in pred_list:
            report["scored"] += 1
            for f in discrete_fields:
                field_exact[f].append(1 if pred.get(f) == gt.get(f) else 0)
            field_exact["composition"].append(
                1 if pred.get("composition") == gt.get("composition") else 0
            )
            for f in numeric_fields:
                gv, pv = gt.get(f), pred.get(f)
                if gv is None or pv is None:
                    continue
                try:
                    gf, pf = float(gv), float(pv)
                except (TypeError, ValueError):
                    continue
                err = abs(pf - gf)
                field_errors[f].append(err)
                if gf != 0:
                    field_rel_errors[f].append(err / abs(gf))
                per_id_errors[f].setdefault(item_id, []).append(err)

    for f in discrete_fields + ["composition"]:
        v = field_exact[f]
        report["field_metrics"][f] = {
            "accuracy": float(np.mean(v)) if v else None,
            "count": len(v),
        }
    for f in numeric_fields:
        e, r = field_errors[f], field_rel_errors[f]
        per_id = per_id_errors[f]
        per_means = [float(np.mean(x)) for x in per_id.values() if x]
        per_stds = [float(np.std(x)) for x in per_id.values() if x]
        report["field_metrics"][f] = {
            "mae": float(np.mean(e)) if e else None,
            "rel_mae": float(np.mean(r)) if r else None,
            "per_id_mae_mean": float(np.mean(per_means)) if per_means else None,
            "per_id_mae_std": float(np.mean(per_stds)) if per_stds else None,
            "count": len(e),
        }

    print("\n--- Analysis (predictions vs gold) ---")
    print(f"  total gold: {report['total']}  scored: {report['scored']}  missing: {report['missing_predictions']}")
    for f, m in report["field_metrics"].items():
        if "accuracy" in m and m["accuracy"] is not None:
            print(f"  {f}: accuracy = {m['accuracy']:.4f}  (n={m['count']})")
        elif "mae" in m and m["mae"] is not None:
            print(f"  {f}: MAE = {m['mae']:.4f}  rel_MAE = {m['rel_mae']:.4f}  (n={m['count']})")
    print("---")

    if score_out is not None:
        score_out.parent.mkdir(parents=True, exist_ok=True)
        with score_out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Wrote score JSON to {score_out}")

    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CIF-only baseline: predict nanoparticle properties from CIF + R values (no gold, no XYZ).",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run CIF-only baseline and write prediction JSONL.")
    run.add_argument(
        "--cif-root",
        type=Path,
        help="Directory of Material.cif files. Omit if using --cif.",
    )
    run.add_argument(
        "--cif",
        type=Path,
        help="Single CIF file (material = stem). Overrides --cif-root.",
    )
    run.add_argument(
        "--r-values",
        type=str,
        required=True,
        help="Comma-separated R values (Å), e.g. 10,11,13,15,17,20,24,27,29,30",
    )
    run.add_argument("--out", type=Path, required=True, help="Output prediction JSONL.")
    run.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold JSONL to evaluate against. If set, run analysis and report errors.",
    )
    run.add_argument(
        "--score-out",
        type=Path,
        default=None,
        help="Write score JSON here when using --gold.",
    )
    run.add_argument("--verbose", action="store_true")

    score = sub.add_parser("score", help="Analyze predictions vs gold only (no CIF run).")
    score.add_argument("--gold", type=Path, required=True, help="Gold JSONL.")
    score.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL.")
    score.add_argument("--out", type=Path, default=None, help="Write score JSON here.")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "score":
        if not args.gold.exists():
            print(f"[ERROR] Gold not found: {args.gold}", file=sys.stderr)
            return 1
        if not args.predictions.exists():
            print(f"[ERROR] Predictions not found: {args.predictions}", file=sys.stderr)
            return 1
        _analyze_and_report_errors(args.gold, args.predictions, score_out=args.out)
        return 0

    if args.command != "run":
        return 1

    r_values: List[int] = []
    for s in args.r_values.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            r_values.append(int(s))
        except ValueError:
            print(f"[ERROR] Invalid --r-values: {args.r_values}", file=sys.stderr)
            return 1
    if not r_values:
        print("[ERROR] No R values provided.", file=sys.stderr)
        return 1

    cif_paths: List[tuple[str, Path]] = []
    if getattr(args, "cif", None) and args.cif is not None:
        p = Path(args.cif)
        if not p.exists():
            print(f"[ERROR] CIF not found: {p}", file=sys.stderr)
            return 1
        cif_paths = [(p.stem, p)]
    elif getattr(args, "cif_root", None) and args.cif_root is not None:
        root = Path(args.cif_root)
        if not root.is_dir():
            print(f"[ERROR] CIF root not found or not a directory: {root}", file=sys.stderr)
            return 1
        cif_paths = discover_cif_materials(root)
        if not cif_paths:
            print(f"[ERROR] No *.cif files in {root}", file=sys.stderr)
            return 1
    else:
        print("[ERROR] Provide either --cif (single file) or --cif-root (directory).", file=sys.stderr)
        return 1

    n = run_cif_only(cif_paths, r_values, args.out, verbose=args.verbose)
    print(f"[OK] Wrote {n} prediction lines to {args.out}")

    if getattr(args, "gold", None) and args.gold is not None:
        if not args.gold.exists():
            print(f"[ERROR] Gold not found: {args.gold}", file=sys.stderr)
            return 1
        _analyze_and_report_errors(args.gold, args.out, score_out=args.score_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
