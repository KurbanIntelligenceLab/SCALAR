import json
import sys
import tempfile
import traceback
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_scalar import carve

RAW_DIR = Path(__file__).resolve().parents[1] / "scalar_raw"
CKPT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "carve_validation"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_xyz(p):
    lines = [l.strip() for l in open(p) if l.strip()]
    n = int(lines[0])
    atom_lines = lines[2:2 + n]
    els, coords = [], []
    for a in atom_lines:
        parts = a.split()
        els.append(parts[0])
        coords.append(list(map(float, parts[1:4])))
    return els, np.array(coords, dtype=np.float64)


def compare_one(dep_els, dep_coords, gen_els, gen_coords):
    n_dep = len(dep_els)
    n_gen = len(gen_els)
    if n_dep == 0 and n_gen == 0:
        return {"n_dep": 0, "n_gen": 0, "delta": 0, "max_coord_dev": 0.0, "species_match": True}
    off = dep_coords.mean(axis=0) - gen_coords.mean(axis=0) if n_gen > 0 else np.zeros(3)
    gen_aligned = gen_coords + off
    max_dev = None
    species_match = None
    if n_dep == n_gen:
        tree = cKDTree(gen_aligned)
        dist, idx = tree.query(dep_coords)
        max_dev = float(dist.max())
        species_match = bool(all(dep_els[i] == gen_els[idx[i]] for i in range(n_dep)))
    return {
        "n_dep": n_dep,
        "n_gen": n_gen,
        "delta": n_dep - n_gen,
        "max_coord_dev": max_dev,
        "species_match": species_match,
    }


def process_material(material):
    ckpt_path = CKPT_DIR / f"{material}.json"
    if ckpt_path.exists():
        return json.loads(ckpt_path.read_text())

    result = {"material": material, "radii": {}, "error": None}
    try:
        cif_path = RAW_DIR / material / f"{material}.cif"
        with tempfile.TemporaryDirectory() as tmp:
            report = carve.carve_material(cif_path, tmp, material)
            for r in range(10, 31):
                gen_path = Path(tmp) / f"{material}_R{r}.xyz"
                dep_path = RAW_DIR / material / f"{material}_R{r}.xyz"
                gen_els, gen_coords = load_xyz(gen_path)
                if dep_path.exists():
                    dep_els, dep_coords = load_xyz(dep_path)
                else:
                    dep_els, dep_coords = [], np.zeros((0, 3))
                result["radii"][str(r)] = compare_one(dep_els, dep_coords, gen_els, gen_coords)
            result["replica_counts"] = report["replica_counts"]
            result["margins"] = report["margins"]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    ckpt_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    materials = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    with Pool(processes=8) as pool:
        results = pool.map(process_material, materials)
    summary_path = Path(__file__).resolve().parents[1] / "outputs" / "carve_validation_report.json"
    exact = 0
    total = 0
    max_dev_overall = 0.0
    max_atom_dev = 0
    non_reproducing = []
    for res in results:
        if res["error"]:
            non_reproducing.append({"material": res["material"], "reason": res["error"][:300]})
            continue
        for r, comp in res["radii"].items():
            total += 1
            if comp["delta"] == 0:
                exact += 1
                if comp["max_coord_dev"] is not None:
                    max_dev_overall = max(max_dev_overall, comp["max_coord_dev"])
            else:
                max_atom_dev = max(max_atom_dev, abs(comp["delta"]))
                non_reproducing.append({
                    "material": res["material"], "radius": r,
                    "reason": f"atom count mismatch dep-gen={comp['delta']} "
                              f"(dep={comp['n_dep']}, gen={comp['n_gen']})",
                })
    summary = {
        "pairs_checked": total,
        "exact_atom_count_matches": exact,
        "mean_atom_count_deviation": float(np.mean([
            abs(comp["delta"]) for res in results if not res["error"]
            for comp in res["radii"].values()
        ])),
        "max_atom_count_deviation": max_atom_dev,
        "max_coord_deviation_angstrom": max_dev_overall,
        "non_reproducing": non_reproducing,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
