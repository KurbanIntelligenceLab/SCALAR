from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from scipy import stats
from scipy.spatial import ConvexHull, cKDTree

R_VALUES = list(range(10, 31))
ID_RADII = {13, 15, 17, 20, 24, 27}
OOD_RADII = {10, 11, 29, 30}
TRAIN_RADII = sorted(set(R_VALUES) - ID_RADII - OOD_RADII)


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols = []
    coords = np.empty((n, 3), dtype=float)
    for i, line in enumerate(lines[2:2 + n]):
        parts = line.split()
        symbols.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return symbols, coords


def parse_cif_bulk(path: Path):
    text = path.read_text(errors="replace")
    lines = [l.rstrip("\r") for l in text.splitlines()]
    a = b = c = None
    for l in lines:
        s = l.strip()
        if s.startswith("_cell_length_a"):
            a = float(s.split()[1])
        elif s.startswith("_cell_length_b"):
            b = float(s.split()[1])
        elif s.startswith("_cell_length_c"):
            c = float(s.split()[1])
    n = len(lines)
    i = 0
    tags = []
    data_start = None
    while i < n:
        if lines[i].strip().lower() == "loop_":
            j = i + 1
            block_tags = []
            while j < n and lines[j].strip().startswith("_"):
                block_tags.append(lines[j].strip())
                j += 1
            if any("_atom_site_" in t for t in block_tags):
                tags = block_tags
                data_start = j
                break
            i = j
        else:
            i += 1
    if data_start is None:
        raise ValueError(f"no atom_site loop in {path}")
    label_idx = type_idx = occ_idx = None
    for idx, t in enumerate(tags):
        tl = t.lower()
        if tl == "_atom_site_type_symbol":
            type_idx = idx
        elif tl == "_atom_site_label":
            label_idx = idx
        elif tl == "_atom_site_occupancy":
            occ_idx = idx
    ncols = len(tags)
    counts = {}
    j = data_start
    while j < n:
        line = lines[j].strip()
        if line == "" or line.startswith("_") or line.lower() == "loop_" or line.startswith("#"):
            break
        parts = line.split()
        if len(parts) < ncols:
            j += 1
            continue
        if type_idx is not None:
            elsym = parts[type_idx]
        else:
            raw = parts[label_idx]
            m = re.match(r"[A-Za-z]+", raw)
            elsym = m.group(0)
        elsym = elsym[0].upper() + elsym[1:].lower() if len(elsym) > 1 else elsym.upper()
        occ = float(parts[occ_idx]) if occ_idx is not None else 1.0
        counts[elsym] = counts.get(elsym, 0.0) + occ
        j += 1
    total = sum(counts.values())
    frac = {k: v / total for k, v in counts.items()}
    return a, b, c, frac


def hull_volume(coords: np.ndarray) -> float:
    return float(ConvexHull(coords).volume)


def mass_amu(symbols) -> float:
    return float(sum(atomic_masses[atomic_numbers[s]] for s in symbols))


def entropy_from_counts(counts: dict) -> tuple[float, float]:
    total = sum(counts.values())
    p = np.array([v / total for v in counts.values() if v > 0])
    h = float(-np.sum(p * np.log(p)))
    return h, float(np.exp(h))


def power_law_fit(r: np.ndarray, y: np.ndarray) -> dict:
    log_fit = stats.linregress(np.log(r), np.log(y))
    dof = len(r) - 2
    t_crit = stats.t.ppf(0.975, dof)
    ci = (
        float(log_fit.slope - t_crit * log_fit.stderr),
        float(log_fit.slope + t_crit * log_fit.stderr),
    )
    return {
        "exponent": float(log_fit.slope),
        "exponent_ci95": ci,
        "log_r2": float(log_fit.rvalue ** 2),
        "n_points": len(r),
    }


def surface_fraction(coords: np.ndarray, cutoff_factor: float = 1.25) -> tuple[float, int, int]:
    n = len(coords)
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    median_nn = float(np.median(d[:, 1]))
    cutoff = cutoff_factor * median_nn
    neighbor_counts = np.array([len(tree.query_ball_point(coords[i], cutoff)) - 1 for i in range(n)])
    vals, cnts = np.unique(neighbor_counts, return_counts=True)
    bulk_coord = int(vals[np.argmax(cnts)])
    n_surface = int(np.sum(neighbor_counts < bulk_coord))
    n_bulk = n - n_surface
    ratio = float(n_surface / n_bulk) if n_bulk > 0 else float("nan")
    return ratio, n_surface, n_bulk


def load_material_record(raw_root: Path, mat: str) -> dict:
    cif_path = raw_root / mat / f"{mat}.cif"
    a, b, c, bulk_frac = parse_cif_bulk(cif_path)
    per_radius = {}
    for r in R_VALUES:
        xyz_path = raw_root / mat / f"{mat}_R{r}.xyz"
        symbols, coords = read_xyz(xyz_path)
        counts = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1
        n_atoms = len(symbols)
        vol = hull_volume(coords)
        mass = mass_amu(symbols)
        density = mass * 1.66053906660 / vol
        h, eff = entropy_from_counts(counts)
        sr_ratio, n_surf, n_bulk = surface_fraction(coords)
        all_elems = set(counts.keys()) | set(bulk_frac.keys())
        frac_r = {e: counts.get(e, 0) / n_atoms for e in all_elems}
        mad = float(np.mean([abs(frac_r[e] - bulk_frac.get(e, 0.0)) for e in all_elems]))
        per_radius[r] = {
            "num_atoms": n_atoms,
            "hull_volume": vol,
            "mass_amu": mass,
            "density_g_cm3": density,
            "element_counts": counts,
            "entropy_nats": h,
            "effective_elements": eff,
            "surface_to_bulk_ratio": sr_ratio,
            "n_surface_atoms": n_surf,
            "n_bulk_atoms": n_bulk,
            "stoichiometry_mad_from_bulk": mad,
        }
    return {
        "material": mat,
        "lattice_a": a,
        "lattice_b": b,
        "lattice_c": c,
        "bulk_composition_fraction": bulk_frac,
        "per_radius": per_radius,
    }


def build_split_metrics(records: dict, radii: list[int]) -> dict:
    element_counts_total = {}
    atom_counts = []
    hull_volumes = []
    material_atoms_containing = {}
    n_structures = 0
    for mat, rec in records.items():
        for r in radii:
            pr = rec["per_radius"][r]
            n_structures += 1
            atom_counts.append(pr["num_atoms"])
            hull_volumes.append(pr["hull_volume"])
            for e, c in pr["element_counts"].items():
                element_counts_total[e] = element_counts_total.get(e, 0) + c
    total_atoms = sum(element_counts_total.values())
    h, eff = entropy_from_counts(element_counts_total)
    atom_arr = np.array(atom_counts, dtype=float)
    vol_arr = np.array(hull_volumes, dtype=float)

    la = np.array([rec["lattice_a"] for rec in records.values()])
    lb = np.array([rec["lattice_b"] for rec in records.values()])
    lc = np.array([rec["lattice_c"] for rec in records.values()])

    def stat_block(x):
        return {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "cv_percent": float(100 * np.std(x, ddof=1) / np.mean(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "range": float(np.max(x) - np.min(x)),
        }

    lattice_stats = {"a": stat_block(la), "b": stat_block(lb), "c": stat_block(lc)}
    lattice_corr = {
        "a_b": float(np.corrcoef(la, lb)[0, 1]),
        "a_c": float(np.corrcoef(la, lc)[0, 1]),
        "b_c": float(np.corrcoef(lb, lc)[0, 1]),
    }

    mean_n_by_r = np.array([np.mean([records[m]["per_radius"][r]["num_atoms"] for m in records]) for r in radii])
    mean_v_by_r = np.array([np.mean([records[m]["per_radius"][r]["hull_volume"] for m in records]) for r in radii])
    r_arr = np.array(radii, dtype=float)
    fit_n = power_law_fit(r_arr, mean_n_by_r) if len(radii) >= 3 else None
    fit_v = power_law_fit(r_arr, mean_v_by_r) if len(radii) >= 3 else None

    elements_by_atom_fraction = sorted(
        ({"element": e, "atom_fraction": c / total_atoms} for e, c in element_counts_total.items()),
        key=lambda d: -d["atom_fraction"],
    )
    n_structures_per_element = {}
    for mat, rec in records.items():
        for r in radii:
            pr = rec["per_radius"][r]
            for e in pr["element_counts"]:
                n_structures_per_element[e] = n_structures_per_element.get(e, 0) + 1
    elements_by_n_structures = sorted(
        ({"element": e, "n_structures": n, "fraction_structures": n / n_structures} for e, n in n_structures_per_element.items()),
        key=lambda d: -d["n_structures"],
    )

    return {
        "radii": radii,
        "n_radii": len(radii),
        "n_structures": n_structures,
        "n_unique_elements": len(element_counts_total),
        "hydrogen_atom_fraction_percent": float(100 * element_counts_total.get("H", 0) / total_atoms),
        "composition_entropy_nats": h,
        "effective_element_count": eff,
        "elements_by_atom_fraction_top10": elements_by_atom_fraction[:10],
        "elements_by_n_structures_top10": elements_by_n_structures[:10],
        "lattice_stats": lattice_stats,
        "lattice_pairwise_pearson_r": lattice_corr,
        "atom_count_stats": {
            "min": float(atom_arr.min()),
            "max": float(atom_arr.max()),
            "mean": float(atom_arr.mean()),
            "cv_percent": float(100 * atom_arr.std(ddof=1) / atom_arr.mean()),
        },
        "hull_volume_stats": {
            "min": float(vol_arr.min()),
            "max": float(vol_arr.max()),
            "mean": float(vol_arr.mean()),
            "cv_percent": float(100 * vol_arr.std(ddof=1) / vol_arr.mean()),
        },
        "atom_count_power_law_fit_vs_R": fit_n,
        "hull_volume_power_law_fit_vs_R": fit_v,
    }


def step1(records: dict) -> dict:
    splits = {
        "training": TRAIN_RADII,
        "ID": sorted(ID_RADII),
        "OOD": sorted(OOD_RADII),
        "full": R_VALUES,
    }
    per_split_table = {name: build_split_metrics(records, radii) for name, radii in splits.items()}

    la_vals = np.array([rec["lattice_a"] for rec in records.values()])
    lb_vals = np.array([rec["lattice_b"] for rec in records.values()])
    lc_vals = np.array([rec["lattice_c"] for rec in records.values()])
    invariant = True
    for name in ("training", "ID", "OOD"):
        block = per_split_table[name]["lattice_stats"]
        if not (
            np.isclose(block["a"]["mean"], np.mean(la_vals))
            and np.isclose(block["b"]["mean"], np.mean(lb_vals))
            and np.isclose(block["c"]["mean"], np.mean(lc_vals))
        ):
            invariant = False

    return {
        "split_radii": {
            "training": TRAIN_RADII,
            "ID": sorted(ID_RADII),
            "OOD": sorted(OOD_RADII),
            "n_training": len(TRAIN_RADII),
            "n_ID": len(ID_RADII),
            "n_OOD": len(OOD_RADII),
        },
        "lattice_parameters_split_invariant": invariant,
        "per_split_table": per_split_table,
    }


def step2(records: dict) -> dict:
    n_materials = len(records)
    n_structures_total = n_materials * len(R_VALUES)
    element_data = {}
    per_material_element_fraction = {mat: {} for mat in records}

    for mat, rec in records.items():
        mat_counts = {}
        for r in R_VALUES:
            for e, c in rec["per_radius"][r]["element_counts"].items():
                mat_counts[e] = mat_counts.get(e, 0) + c
        mat_total = sum(mat_counts.values())
        for e, c in mat_counts.items():
            per_material_element_fraction[mat][e] = c / mat_total

    all_elements = sorted({e for rec in records.values() for e in rec["bulk_composition_fraction"]} |
                           {e for mat in per_material_element_fraction for e in per_material_element_fraction[mat]})

    total_atom_counts = {}
    n_materials_with = {}
    n_structures_with = {}
    sum_atoms_when_present = {}
    n_structures_when_present = {}
    for e in all_elements:
        total_atom_counts[e] = 0
        n_materials_with[e] = 0
        n_structures_with[e] = 0
        sum_atoms_when_present[e] = 0
        n_structures_when_present[e] = 0

    for mat, rec in records.items():
        mat_has_element = set()
        for r in R_VALUES:
            pr = rec["per_radius"][r]
            for e in all_elements:
                c = pr["element_counts"].get(e, 0)
                total_atom_counts[e] += c
                if c > 0:
                    n_structures_with[e] += 1
                    sum_atoms_when_present[e] += c
                    n_structures_when_present[e] += 1
                    mat_has_element.add(e)
        for e in mat_has_element:
            n_materials_with[e] += 1

    grand_total_atoms = sum(total_atom_counts.values())
    rows = []
    for e in all_elements:
        rows.append({
            "element": e,
            "atom_fraction": total_atom_counts[e] / grand_total_atoms,
            "n_materials_containing": n_materials_with[e],
            "fraction_materials_containing": n_materials_with[e] / n_materials,
            "n_structures_containing": n_structures_with[e],
            "fraction_structures_containing": n_structures_with[e] / n_structures_total,
            "mean_atoms_per_structure_when_present": (
                sum_atoms_when_present[e] / n_structures_when_present[e] if n_structures_when_present[e] > 0 else 0.0
            ),
        })
    rows.sort(key=lambda d: -d["atom_fraction"])

    avg_material_fraction = {e: 0.0 for e in all_elements}
    for mat in records:
        for e in all_elements:
            avg_material_fraction[e] += per_material_element_fraction[mat].get(e, 0.0)
    for e in all_elements:
        avg_material_fraction[e] /= n_materials

    p = np.array([v for v in avg_material_fraction.values() if v > 0])
    h_material = float(-np.sum(p * np.log(p)))
    eff_material = float(np.exp(h_material))

    atom_weighted_counts = {e: total_atom_counts[e] for e in all_elements}
    h_atom, eff_atom = entropy_from_counts(atom_weighted_counts)

    return {
        "rows": rows,
        "material_level_entropy_nats": h_material,
        "material_level_effective_elements": eff_material,
        "atom_weighted_entropy_nats": h_atom,
        "atom_weighted_effective_elements": eff_atom,
        "n_materials": n_materials,
        "n_structures_total": n_structures_total,
        "n_unique_elements": len(all_elements),
    }


def step3(records: dict) -> dict:
    per_r = {}
    for r in R_VALUES:
        entropies = [records[m]["per_radius"][r]["entropy_nats"] for m in records]
        effs = [records[m]["per_radius"][r]["effective_elements"] for m in records]
        atoms = [records[m]["per_radius"][r]["num_atoms"] for m in records]
        vols = [records[m]["per_radius"][r]["hull_volume"] for m in records]
        sr_ratios = [records[m]["per_radius"][r]["surface_to_bulk_ratio"] for m in records]
        mads = [records[m]["per_radius"][r]["stoichiometry_mad_from_bulk"] for m in records]
        per_r[r] = {
            "mean_entropy_nats": float(np.mean(entropies)),
            "mean_effective_elements": float(np.mean(effs)),
            "mean_num_atoms": float(np.mean(atoms)),
            "mean_hull_volume": float(np.mean(vols)),
            "mean_surface_to_bulk_ratio": float(np.mean(sr_ratios)),
            "mean_stoichiometry_mad_from_bulk": float(np.mean(mads)),
        }

    radii_arr = np.array(R_VALUES)
    entropy_arr = np.array([per_r[r]["mean_entropy_nats"] for r in R_VALUES])
    mad_arr = np.array([per_r[r]["mean_stoichiometry_mad_from_bulk"] for r in R_VALUES])

    entropy_deriv = np.diff(entropy_arr) / np.diff(radii_arr)
    mad_deriv = np.diff(mad_arr) / np.diff(radii_arr)
    rel_entropy_deriv = np.abs(entropy_deriv) / entropy_arr[:-1]
    rel_mad_deriv = np.abs(mad_deriv) / mad_arr[:-1]

    flatten_threshold = 0.02
    entropy_flatten_r = None
    for i, r in enumerate(R_VALUES[:-1]):
        if all(rel_entropy_deriv[i:] < flatten_threshold):
            entropy_flatten_r = R_VALUES[i + 1]
            break
    mad_flatten_r = None
    for i, r in enumerate(R_VALUES[:-1]):
        if all(rel_mad_deriv[i:] < flatten_threshold):
            mad_flatten_r = R_VALUES[i + 1]
            break

    train_atoms = np.array([per_r[r]["mean_num_atoms"] for r in TRAIN_RADII])
    train_vols = np.array([per_r[r]["mean_hull_volume"] for r in TRAIN_RADII])
    train_atom_range = (float(train_atoms.min()), float(train_atoms.max()))
    train_vol_range = (float(train_vols.min()), float(train_vols.max()))

    id_coverage = {}
    for r in sorted(ID_RADII):
        id_coverage[r] = {
            "mean_num_atoms": per_r[r]["mean_num_atoms"],
            "mean_hull_volume": per_r[r]["mean_hull_volume"],
            "atom_frac_of_training_range": (per_r[r]["mean_num_atoms"] - train_atom_range[0]) / (train_atom_range[1] - train_atom_range[0]),
            "vol_frac_of_training_range": (per_r[r]["mean_hull_volume"] - train_vol_range[0]) / (train_vol_range[1] - train_vol_range[0]),
        }
    ood_extrapolation = {}
    for r in sorted(OOD_RADII):
        atoms_r = per_r[r]["mean_num_atoms"]
        vol_r = per_r[r]["mean_hull_volume"]
        if atoms_r < train_atom_range[0]:
            atom_extrap_percent = 100 * (train_atom_range[0] - atoms_r) / train_atom_range[0]
        elif atoms_r > train_atom_range[1]:
            atom_extrap_percent = 100 * (atoms_r - train_atom_range[1]) / train_atom_range[1]
        else:
            atom_extrap_percent = 0.0
        if vol_r < train_vol_range[0]:
            vol_extrap_percent = 100 * (train_vol_range[0] - vol_r) / train_vol_range[0]
        elif vol_r > train_vol_range[1]:
            vol_extrap_percent = 100 * (vol_r - train_vol_range[1]) / train_vol_range[1]
        else:
            vol_extrap_percent = 0.0
        ood_extrapolation[r] = {
            "mean_num_atoms": atoms_r,
            "mean_hull_volume": vol_r,
            "atom_extrapolation_percent_beyond_training_range": atom_extrap_percent,
            "vol_extrapolation_percent_beyond_training_range": vol_extrap_percent,
        }

    n_materials = len(records)
    n_total_structures = n_materials * len(R_VALUES)
    split_fracs = {"train": 0.60, "ID": 0.25, "OOD": 0.15}
    n_train_structs = n_materials * len(TRAIN_RADII)
    n_id_structs = n_materials * len(ID_RADII)
    n_ood_structs = n_materials * len(OOD_RADII)

    return {
        "per_radius": per_r,
        "entropy_relative_derivative_percent_per_A": (100 * rel_entropy_deriv).tolist(),
        "stoichiometry_mad_relative_derivative_percent_per_A": (100 * rel_mad_deriv).tolist(),
        "entropy_flatten_radius": entropy_flatten_r,
        "stoichiometry_mad_flatten_radius": mad_flatten_r,
        "flatten_threshold_percent_per_A": 100 * flatten_threshold,
        "training_atom_count_range": train_atom_range,
        "training_hull_volume_range": train_vol_range,
        "id_coverage_of_training_range": id_coverage,
        "ood_extrapolation_beyond_training_range": ood_extrapolation,
        "split_structure_counts": {
            "n_materials": n_materials,
            "n_total_structures": n_total_structures,
            "n_train_structures": n_train_structs,
            "n_train_structures_actual_fraction": n_train_structs / n_total_structures,
            "n_id_structures": n_id_structs,
            "n_id_structures_actual_fraction": n_id_structs / n_total_structures,
            "n_ood_structures": n_ood_structs,
            "n_ood_structures_actual_fraction": n_ood_structs / n_total_structures,
            "target_split_fractions": split_fracs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    args.partial_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    materials = sorted(p.name for p in args.raw_root.iterdir() if p.is_dir())
    assert len(TRAIN_RADII) == 11, f"expected 11 training radii, got {len(TRAIN_RADII)}: {TRAIN_RADII}"

    records = {}
    for mat in materials:
        ckpt_path = args.partial_dir / f"{mat}.json"
        if ckpt_path.exists():
            records[mat] = json.loads(ckpt_path.read_text())
            continue
        rec = load_material_record(args.raw_root, mat)
        ckpt_path.write_text(json.dumps(rec))
        records[mat] = rec

    s1 = step1(records)
    s2 = step2(records)
    s3 = step3(records)

    result = {
        "n_materials": len(records),
        "step1_per_split_metrics": s1,
        "step2_element_prevalence": {k: v for k, v in s2.items() if k != "rows"},
        "step3_radius_range": s3,
    }
    args.out.write_text(json.dumps(result, indent=2))

    csv_lines = [
        "element,atom_fraction,n_materials_containing,fraction_materials_containing,"
        "n_structures_containing,fraction_structures_containing,mean_atoms_per_structure_when_present"
    ]
    for row in s2["rows"]:
        csv_lines.append(
            f"{row['element']},{row['atom_fraction']:.8f},{row['n_materials_containing']},"
            f"{row['fraction_materials_containing']:.8f},{row['n_structures_containing']},"
            f"{row['fraction_structures_containing']:.8f},{row['mean_atoms_per_structure_when_present']:.6f}"
        )
    args.csv_out.write_text("\n".join(csv_lines) + "\n")

    print(json.dumps({
        "n_materials": len(records),
        "training_radii": TRAIN_RADII,
        "material_level_entropy_nats": s2["material_level_entropy_nats"],
        "material_level_effective_elements": s2["material_level_effective_elements"],
        "atom_weighted_entropy_nats": s2["atom_weighted_entropy_nats"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
