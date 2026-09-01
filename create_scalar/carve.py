import json
import re
import warnings
from pathlib import Path

import numpy as np
import spglib
from ase.data import atomic_numbers

from create_scalar.config import (
    CARVE_R_MIN,
    CARVE_R_MAX,
    CARVE_DELTA_BOX,
    CARVE_SYMPREC,
    CARVE_COORD_DECIMALS,
)

warnings.filterwarnings("ignore")


def _cellpar_to_cell(a, b, c, alpha, beta, gamma):
    al, be, ga = np.radians([alpha, beta, gamma])
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * np.cos(ga), b * np.sin(ga), 0.0
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c ** 2 - cx ** 2 - cy ** 2, 0.0))
    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])


def _parse_symops(text):
    lines = text.splitlines()
    header_seen = False
    in_loop = False
    ops = []
    for l in lines:
        if "_space_group_symop_operation_xyz" in l or "_symmetry_equiv_pos_as_xyz" in l:
            header_seen = True
            continue
        if header_seen and not in_loop:
            in_loop = True
        if in_loop:
            s = l.strip().strip("'").strip('"')
            if s == "" or s.startswith("loop_") or s.startswith("_"):
                break
            ops.append(s)
    return ops


def _apply_symops(fracs, symbols, ops, tol=1e-3):
    if len(ops) <= 1:
        return fracs, symbols
    all_f = []
    all_s = []
    for op in ops:
        parts = op.split(",")
        for f, sym in zip(fracs, symbols):
            local = {"x": f[0], "y": f[1], "z": f[2]}
            newf = [eval(p, {"__builtins__": {}}, local) % 1.0 for p in parts]
            all_f.append(newf)
            all_s.append(sym)
    all_f = np.array(all_f)
    uniq_f = []
    uniq_s = []
    for f, s in zip(all_f, all_s):
        dup = False
        for uf, us in zip(uniq_f, uniq_s):
            if us == s and np.linalg.norm(((f - uf + 0.5) % 1.0) - 0.5) < tol:
                dup = True
                break
        if not dup:
            uniq_f.append(f)
            uniq_s.append(s)
    return np.array(uniq_f), uniq_s


def parse_cif(cif_path):
    text = Path(cif_path).read_text(encoding="utf-8", errors="replace")

    def get_len(tag):
        m = re.search(rf"_cell_length_{tag}\s+([-\d.]+)", text)
        return float(m.group(1))

    def get_angle(tag):
        m = re.search(rf"_cell_angle_{tag}\s+([-\d.]+)", text)
        return float(m.group(1))

    cellpar = (
        get_len("a"), get_len("b"), get_len("c"),
        get_angle("alpha"), get_angle("beta"), get_angle("gamma"),
    )

    lines = text.splitlines()
    header_idx = None
    for i, l in enumerate(lines):
        if l.strip().lower() == "_atom_site_label":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"no _atom_site_label loop found in {cif_path}")

    headers = []
    i = header_idx
    while i < len(lines) and lines[i].strip().startswith("_atom_site"):
        headers.append(lines[i].strip())
        i += 1
    data_start = i

    label_idx = headers.index("_atom_site_label")
    type_idx = headers.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in headers else None
    fx_idx = headers.index("_atom_site_fract_x")
    fy_idx = headers.index("_atom_site_fract_y")
    fz_idx = headers.index("_atom_site_fract_z")

    symbols = []
    fracs = []
    for l in lines[data_start:]:
        l = l.strip()
        if not l or l.startswith("_") or l.startswith("loop_") or l.startswith("#"):
            break
        parts = l.split()
        if len(parts) < len(headers):
            break
        sym = parts[type_idx] if type_idx is not None else re.match(r"[A-Za-z]+", parts[label_idx]).group(0)
        symbols.append(sym)
        fracs.append([float(parts[fx_idx]), float(parts[fy_idx]), float(parts[fz_idx])])
    fracs = np.array(fracs, dtype=np.float64)

    ops = _parse_symops(text)
    fracs, symbols = _apply_symops(fracs, symbols, ops)

    cell = _cellpar_to_cell(*cellpar)
    return cell, fracs, symbols


def read_unit_cell(cif_path):
    return parse_cif(cif_path)


def build_primitive_cell(cell, fracs, symbols, symprec=CARVE_SYMPREC):
    numbers = np.array([atomic_numbers[s] for s in symbols])
    lattice = (cell, fracs, numbers)
    result = spglib.find_primitive(lattice, symprec=symprec)
    if result is None:
        return cell, fracs, symbols
    pcell, ppos, pnum = result
    if len(pnum) >= len(numbers):
        return cell, fracs, symbols
    number_to_symbol = {atomic_numbers[s]: s for s in symbols}
    psym = [number_to_symbol[n] for n in pnum]
    return pcell, ppos, psym


def perpendicular_spacings(cell):
    a1, a2, a3 = cell[0], cell[1], cell[2]
    volume = abs(np.dot(a1, np.cross(a2, a3)))
    d1 = volume / np.linalg.norm(np.cross(a2, a3))
    d2 = volume / np.linalg.norm(np.cross(a3, a1))
    d3 = volume / np.linalg.norm(np.cross(a1, a2))
    return np.array([d1, d2, d3])


def replica_counts(cell, r_max=CARVE_R_MAX, delta_box=CARVE_DELTA_BOX):
    spacings = perpendicular_spacings(cell)
    target = r_max + delta_box
    n = np.ceil(target / spacings).astype(int)
    n = np.maximum(n, 1)
    margins = n * spacings - r_max
    return n, margins, spacings


def build_supercell(cell, frac_pos, symbols, n):
    n1, n2, n3 = n
    i_range = np.arange(-n1, n1 + 1)
    j_range = np.arange(-n2, n2 + 1)
    k_range = np.arange(-n3, n3 + 1)
    ii, jj, kk = np.meshgrid(i_range, j_range, k_range, indexing="ij")
    shifts = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    cart_shifts = shifts @ cell

    n_atoms = len(symbols)
    n_shifts = len(shifts)
    all_pos = np.zeros((n_shifts * n_atoms, 3), dtype=np.float64)
    all_syms = []
    for a_idx in range(n_atoms):
        atom_cart = frac_pos[a_idx] @ cell
        all_pos[a_idx * n_shifts:(a_idx + 1) * n_shifts] = cart_shifts + atom_cart
        all_syms.extend([symbols[a_idx]] * n_shifts)
    return all_syms, all_pos


def carve_sphere(symbols, positions, r, centre=np.zeros(3)):
    d = np.linalg.norm(positions - centre, axis=1)
    mask = d <= r
    kept_syms = [symbols[i] for i in np.nonzero(mask)[0]]
    kept_pos = positions[mask]
    return kept_syms, kept_pos


def save_xyz(symbols, positions, path, material, r_val):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"Carved structure | material={material} | R={r_val} | centre=origin\n")
        for el, (x, y, z) in zip(symbols, positions):
            f.write(
                f"{el:2s} {x:{6+CARVE_COORD_DECIMALS}.{CARVE_COORD_DECIMALS}f} "
                f"{y:{6+CARVE_COORD_DECIMALS}.{CARVE_COORD_DECIMALS}f} "
                f"{z:{6+CARVE_COORD_DECIMALS}.{CARVE_COORD_DECIMALS}f}\n"
            )


def carve_material(cif_path, output_dir, material=None, r_min=CARVE_R_MIN, r_max=CARVE_R_MAX,
                    delta_box=CARVE_DELTA_BOX):
    cif_path = Path(cif_path)
    material = material or cif_path.stem
    cell0, frac0, symbols0 = parse_cif(cif_path)
    cell, frac_pos, symbols = build_primitive_cell(cell0, frac0, symbols0)
    n, margins, spacings = replica_counts(cell, r_max, delta_box)
    all_syms, all_pos = build_supercell(cell, frac_pos, symbols, n)

    written = []
    counts = {}
    for r in range(r_min, r_max + 1):
        kept_syms, kept_pos = carve_sphere(all_syms, all_pos, float(r))
        out_path = Path(output_dir) / f"{material}_R{r}.xyz"
        save_xyz(kept_syms, kept_pos, out_path, material, r)
        written.append(str(out_path))
        counts[str(r)] = len(kept_syms)

    report = {
        "material": material,
        "cif": str(cif_path),
        "replica_counts": [int(x) for x in n],
        "perpendicular_spacings": [float(x) for x in spacings],
        "margins": [float(x) for x in margins],
        "delta_box": float(delta_box),
        "r_min": r_min,
        "r_max": r_max,
        "counts": counts,
        "written": written,
    }
    return report


def carve_all(raw_dir, output_dir, r_min=CARVE_R_MIN, r_max=CARVE_R_MAX,
               delta_box=CARVE_DELTA_BOX):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    reports = {}
    cif_files = sorted(raw_dir.rglob("*.cif"))
    for cif_path in cif_files:
        if cif_path.name.startswith("._"):
            continue
        material = cif_path.stem
        report = carve_material(cif_path, output_dir, material, r_min, r_max, delta_box)
        reports[material] = report
    return reports


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--r-min", type=int, default=CARVE_R_MIN)
    parser.add_argument("--r-max", type=int, default=CARVE_R_MAX)
    parser.add_argument("--delta-box", type=float, default=CARVE_DELTA_BOX)
    args = parser.parse_args()

    report = carve_material(
        args.cif, args.output, r_min=args.r_min, r_max=args.r_max, delta_box=args.delta_box
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
