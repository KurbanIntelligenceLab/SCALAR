"""
Quaternion rotation generation for scalar dataset.

Uses create_scalar.config (utils/generate_quaternions-style settings):
TARGET_TOTAL_FILES, SPLIT_FRACTIONS, MAX_ROTS_PER_FILE, ANGLE_BY_SPLIT, etc.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from create_scalar import config as scalar_config

# Worker globals (set by pool initializer; used by _process_train_id / _process_ood)
_TRAIN_GRID: Optional[Dict[float, np.ndarray]] = None
_EXCLUDE_QS: Optional[np.ndarray] = None
_BUDGETS: Optional[Dict[str, Dict[str, int]]] = None
_OUTPUT_DIR: Optional[Path] = None


def _init_pool(
    train_grid: Dict[float, np.ndarray],
    exclude_qs: np.ndarray,
    budgets: Dict[str, Dict[str, int]],
    output_dir: Path,
):
    """Pool initializer: set worker globals. Must stay module-level for pickling."""
    global _TRAIN_GRID, _EXCLUDE_QS, _BUDGETS, _OUTPUT_DIR
    _TRAIN_GRID = train_grid
    _EXCLUDE_QS = exclude_qs
    _BUDGETS = budgets
    _OUTPUT_DIR = output_dir


def _process_train_id(xyz_path: Path) -> Tuple[Dict[str, int], np.ndarray]:
    """Worker target for pass 1 (train + ID). Must stay module-level for pickling."""
    return ScalarQuaternionGenerator._process_train_id_impl(xyz_path)


def _process_ood(xyz_path: Path) -> Dict[str, int]:
    """Worker target for pass 2 (OOD). Must stay module-level for pickling."""
    return ScalarQuaternionGenerator._process_ood_impl(xyz_path)


class ScalarQuaternionGenerator:
    """
    Quaternion rotation generator for the scalar dataset.

    Uses create_scalar.config (TARGET_TOTAL_FILES, SPLIT_FRACTIONS,
    MAX_ROTS_PER_FILE, ANGLE_BY_SPLIT, etc.) via scalar_config.
    """

    def __init__(
        self,
        xyz_files: List[Path],
        output_root: Path,
        label: str = "scalar",
    ):
        self.xyz_files = xyz_files
        self.output_root = output_root
        self.label = label

    # ─────────────────────────────────────────────────────────────────────────
    # Static helpers (OOP; use scalar_config)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _stable_seed(*parts) -> int:
        s = "|".join(map(str, parts)).encode("utf-8")
        h = 2166136261
        for b in s:
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h)

    @staticmethod
    def _split_for_r(r: int) -> str:
        if r in scalar_config.r_splits["ID"]:
            return "ID"
        if r in scalar_config.r_splits["OOD"]:
            return "OOD"
        return "train"

    @staticmethod
    def _generate_uniform_rotations(
        angle_sep_deg: float, seed: int, n_quats: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cos_cap = np.cos(np.radians(angle_sep_deg) / 2)
        quats = []
        trials = max(200_000, 50_000 * n_quats)

        while len(quats) < n_quats and trials:
            q = R.random(random_state=rng).as_quat()
            if q[3] < 0:
                q = -q
            if not quats:
                quats.append(q)
            else:
                Q = np.vstack(quats)
                if np.max(np.abs(Q @ q)) <= cos_cap:
                    quats.append(q)
            trials -= 1

        if len(quats) < n_quats:
            raise RuntimeError(
                f"Only {len(quats)} of {n_quats} rotations placed for {angle_sep_deg}°."
            )
        return np.asarray(quats, dtype=np.float64)

    @staticmethod
    def _sample_with_exclusion(
        angle_sep_deg: float,
        seed: int,
        n_quats: int,
        exclude_Q: Optional[np.ndarray] = None,
        margin_deg: float = 0.0,
        left_mul: Optional[R] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cos_internal = np.cos(np.radians(angle_sep_deg) / 2)
        cos_excl = (
            np.cos(np.radians(margin_deg) / 2)
            if (exclude_Q is not None and margin_deg > 0)
            else None
        )

        eff_quats = []
        trials = max(300_000, 75_000 * n_quats)

        while len(eff_quats) < n_quats and trials:
            q = R.random(random_state=rng).as_quat()
            if q[3] < 0:
                q = -q

            if left_mul is not None:
                q_eff = (left_mul * R.from_quat(q)).as_quat()
                if q_eff[3] < 0:
                    q_eff = -q_eff
            else:
                q_eff = q

            if eff_quats:
                Qeff = np.vstack(eff_quats)
                if np.max(np.abs(Qeff @ q_eff)) > cos_internal:
                    trials -= 1
                    continue

            if cos_excl is not None:
                if np.max(np.abs(exclude_Q @ q_eff)) > cos_excl:
                    trials -= 1
                    continue

            eff_quats.append(q_eff)
            trials -= 1

        if len(eff_quats) < n_quats:
            raise RuntimeError(
                f"Only {len(eff_quats)} of {n_quats} placed "
                f"(sep {angle_sep_deg}°, margin {margin_deg}°)."
            )
        return np.asarray(eff_quats, dtype=np.float64)

    @staticmethod
    def _unique_rotations(
        coords: np.ndarray, quats: np.ndarray, tol: float = 1e-6
    ) -> np.ndarray:
        seen = set()
        unique = []
        for q in quats:
            Rcoords = R.from_quat(q).apply(coords)
            key = tuple(map(tuple, np.round(Rcoords / tol).astype(int)))
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return np.array(unique, dtype=np.float64)

    @staticmethod
    def _save_xyz_file(
        coords,
        elements,
        num_atoms,
        r_val,
        path,
        rotation_angles=None,
        *,
        split: Optional[str] = None,
        spacing: Optional[float] = None,
        regime: Optional[str] = None,
    ):
        with open(path, "w") as f:
            f.write(f"{num_atoms}\n")
            if rotation_angles is None:
                meta = ["Original structure"]
                if split is not None:
                    meta.append(f"split={split}")
                meta.append(f"R={r_val}")
                f.write(" | ".join(meta) + "\n")
            else:
                x, y, z = rotation_angles
                meta = ["Rotated structure"]
                if split is not None:
                    meta.append(f"split={split}")
                if regime is not None:
                    meta.append(f"regime={regime}")
                if spacing is not None:
                    meta.append(f"spacing={spacing:.1f}")
                meta.append(f"euler_deg=({x:.1f},{y:.1f},{z:.1f})")
                meta.append(f"R={r_val}")
                f.write(" | ".join(meta) + "\n")
            for el, (X, Y, Z) in zip(elements, coords):
                f.write(f"{el:2s} {X:15.8f} {Y:15.8f} {Z:15.8f}\n")

    @staticmethod
    def _load_xyz(xyz_path: Path):
        try:
            with xyz_path.open(encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except UnicodeDecodeError:
            with xyz_path.open(encoding="latin-1", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        num_atoms = int(lines[0])
        atom_lines = lines[2 : 2 + num_atoms]
        elements, coords = zip(
            *[(ln.split()[0], list(map(float, ln.split()[1:4]))) for ln in atom_lines]
        )
        coords = np.array(coords, dtype=np.float64)
        return num_atoms, elements, coords

    @staticmethod
    def _compute_budgets_from_target(
        xyz_files: List[Path], source_label: str
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        base_counts = {"train": 0, "ID": 0, "OOD": 0}
        for p in xyz_files:
            m = re.match(r"(.+)_R(\d+)\.xyz", p.name)
            if not m:
                continue
            r = int(m.group(2))
            if r not in scalar_config.r_values:
                continue
            base_counts[ScalarQuaternionGenerator._split_for_r(r)] += 1

        total_base = sum(base_counts.values())
        if total_base == 0:
            raise RuntimeError(f"No valid base xyz files found in {source_label}")

        desired = {
            k: int(
                round(
                    scalar_config.TARGET_TOTAL_FILES * scalar_config.SPLIT_FRACTIONS[k]
                )
            )
            for k in scalar_config.SPLIT_FRACTIONS
        }
        delta = scalar_config.TARGET_TOTAL_FILES - sum(desired.values())
        if delta != 0:
            desired["train"] += delta

        budgets = {
            "train": {"base": 0},
            "ID": {"base": 0},
            "OOD": {"dense": 0, "sparse": 0},
        }

        for split in ("train", "ID"):
            n_base = base_counts[split]
            if n_base == 0:
                budgets[split]["base"] = 0
                continue
            k = max(0, (desired[split] - n_base) // n_base)
            k = min(k, scalar_config.MAX_ROTS_PER_FILE[split]["base"])
            budgets[split]["base"] = int(k)

        n_base_ood = base_counts["OOD"]
        if n_base_ood == 0:
            budgets["OOD"]["dense"] = 0
            budgets["OOD"]["sparse"] = 0
        else:
            ood_rot_total = max(0, desired["OOD"] - n_base_ood)
            dense_total = int(round(0.60 * ood_rot_total))
            sparse_total = ood_rot_total - dense_total
            k_dense = min(
                dense_total // n_base_ood,
                scalar_config.MAX_ROTS_PER_FILE["OOD"]["dense"],
            )
            k_sparse = min(
                sparse_total // n_base_ood,
                scalar_config.MAX_ROTS_PER_FILE["OOD"]["sparse"],
            )
            budgets["OOD"]["dense"] = int(k_dense)
            budgets["OOD"]["sparse"] = int(k_sparse)

        return budgets, base_counts

    @staticmethod
    def _expected_counts_from_budgets(
        base_counts: Dict[str, int], budgets: Dict[str, Dict[str, int]]
    ) -> Dict[str, int]:
        exp = {"train": 0, "ID": 0, "OOD": 0}
        exp["train"] = base_counts["train"] * (1 + budgets["train"]["base"])
        exp["ID"] = base_counts["ID"] * (1 + budgets["ID"]["base"])
        exp["OOD"] = base_counts["OOD"] * (
            1 + budgets["OOD"]["dense"] + budgets["OOD"]["sparse"]
        )
        return exp

    @staticmethod
    def _process_train_id_impl(xyz_path: Path) -> Tuple[Dict[str, int], np.ndarray]:
        global _TRAIN_GRID, _EXCLUDE_QS, _BUDGETS, _OUTPUT_DIR
        counts = {"train": 0, "ID": 0, "OOD": 0}
        id_quats_out = np.empty((0, 4), dtype=np.float64)

        m = re.match(r"(.+)_R(\d+)\.xyz", xyz_path.name)
        if not m:
            return counts, id_quats_out

        material, r_str = m.group(1), m.group(2)
        r = int(r_str)
        if r not in scalar_config.r_values:
            return counts, id_quats_out

        split = ScalarQuaternionGenerator._split_for_r(r)
        if split == "OOD":
            return counts, id_quats_out

        num_atoms, elements, coords = ScalarQuaternionGenerator._load_xyz(xyz_path)

        out_dir = _OUTPUT_DIR / material / f"R{r}" / "xyz"
        out_dir.mkdir(parents=True, exist_ok=True)

        ScalarQuaternionGenerator._save_xyz_file(
            coords,
            elements,
            num_atoms,
            r,
            out_dir / "rot_0.xyz",
            None,
            split=split,
        )
        counts[split] += 1

        if split == "train":
            k_train = _BUDGETS["train"]["base"]
            if k_train <= 0:
                return counts, id_quats_out

            spacing = scalar_config.ANGLE_BY_SPLIT["train"]["base"]
            grid = _TRAIN_GRID[spacing]
            if grid.shape[0] < k_train:
                raise RuntimeError(
                    "Train grid smaller than requested budget. "
                    "Increase MAX_ROTS_PER_FILE/train grid size."
                )
            rng = np.random.default_rng(
                ScalarQuaternionGenerator._stable_seed(material, r, "train", spacing)
            )
            idx = rng.choice(grid.shape[0], size=k_train, replace=False)
            all_quats = grid[idx]

        else:
            k_id = _BUDGETS["ID"]["base"]
            if k_id <= 0:
                return counts, id_quats_out

            spacing = scalar_config.ANGLE_BY_SPLIT["ID"]["base"]
            seed = ScalarQuaternionGenerator._stable_seed(material, r, "ID", spacing)
            all_quats = ScalarQuaternionGenerator._sample_with_exclusion(
                spacing,
                seed,
                k_id,
                exclude_Q=_EXCLUDE_QS,
                margin_deg=scalar_config.MARGINS["ID"],
                left_mul=scalar_config.ID_OFFSET,
            )

        unique_qs = ScalarQuaternionGenerator._unique_rotations(
            coords, all_quats, tol=scalar_config.coord_tolerance
        )

        for idx, q in enumerate(unique_qs, start=1):
            rot_coords = R.from_quat(q).apply(coords)
            angles = R.from_quat(q).as_euler("xyz", degrees=True)
            sp = (
                scalar_config.ANGLE_BY_SPLIT["train"]["base"]
                if split == "train"
                else scalar_config.ANGLE_BY_SPLIT["ID"]["base"]
            )
            ScalarQuaternionGenerator._save_xyz_file(
                rot_coords,
                elements,
                num_atoms,
                r,
                out_dir / f"rot_{idx}.xyz",
                angles,
                split=split,
                spacing=sp,
                regime=None,
            )
            counts[split] += 1

        if split == "ID":
            id_quats_out = unique_qs

        return counts, id_quats_out

    @staticmethod
    def _process_ood_impl(xyz_path: Path) -> Dict[str, int]:
        global _EXCLUDE_QS, _BUDGETS, _OUTPUT_DIR
        counts = {"train": 0, "ID": 0, "OOD": 0}

        m = re.match(r"(.+)_R(\d+)\.xyz", xyz_path.name)
        if not m:
            return counts

        material, r_str = m.group(1), m.group(2)
        r = int(r_str)
        if r not in scalar_config.r_values:
            return counts

        split = ScalarQuaternionGenerator._split_for_r(r)
        if split != "OOD":
            return counts

        num_atoms, elements, coords = ScalarQuaternionGenerator._load_xyz(xyz_path)

        out_dir = _OUTPUT_DIR / material / f"R{r}" / "xyz"
        out_dir.mkdir(parents=True, exist_ok=True)

        ScalarQuaternionGenerator._save_xyz_file(
            coords,
            elements,
            num_atoms,
            r,
            out_dir / "rot_0.xyz",
            None,
            split="OOD",
        )
        counts["OOD"] += 1

        k_dense = _BUDGETS["OOD"]["dense"]
        k_sparse = _BUDGETS["OOD"]["sparse"]
        if (k_dense + k_sparse) <= 0:
            return counts

        outputs: List[Tuple[str, float, np.ndarray]] = []

        if k_dense > 0:
            spacing = scalar_config.ANGLE_BY_SPLIT["OOD"]["dense"]
            seed = ScalarQuaternionGenerator._stable_seed(
                material, r, "OOD-dense", spacing
            )
            qs = ScalarQuaternionGenerator._sample_with_exclusion(
                spacing,
                seed,
                k_dense,
                exclude_Q=_EXCLUDE_QS,
                margin_deg=scalar_config.MARGINS["OOD"],
                left_mul=scalar_config.OOD_OFFSET,
            )
            outputs.append(("dense", spacing, qs))

        if k_sparse > 0:
            spacing = scalar_config.ANGLE_BY_SPLIT["OOD"]["sparse"]
            seed = ScalarQuaternionGenerator._stable_seed(
                material, r, "OOD-sparse", spacing
            )
            qs = ScalarQuaternionGenerator._sample_with_exclusion(
                spacing,
                seed,
                k_sparse,
                exclude_Q=_EXCLUDE_QS,
                margin_deg=scalar_config.MARGINS["OOD"],
                left_mul=scalar_config.OOD_OFFSET,
            )
            outputs.append(("sparse", spacing, qs))

        file_idx = 1
        for regime, spacing, quats in outputs:
            unique_qs = ScalarQuaternionGenerator._unique_rotations(
                coords, quats, tol=scalar_config.coord_tolerance
            )
            for q in unique_qs:
                rot_coords = R.from_quat(q).apply(coords)
                angles = R.from_quat(q).as_euler("xyz", degrees=True)
                ScalarQuaternionGenerator._save_xyz_file(
                    rot_coords,
                    elements,
                    num_atoms,
                    r,
                    out_dir / f"rot_{file_idx}.xyz",
                    angles,
                    split="OOD",
                    regime=regime,
                    spacing=spacing,
                )
                counts["OOD"] += 1
                file_idx += 1

        return counts

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self) -> Dict[str, int]:
        """Run two-pass quaternion generation. Returns actual counts per split."""
        budgets, base_counts = self._compute_budgets_from_target(
            self.xyz_files, self.label
        )
        expected = self._expected_counts_from_budgets(base_counts, budgets)
        total_expected = sum(expected.values())

        print(f"\n=== DATASET: {self.label} ===")
        print("=== BASE COUNTS (input xyz) ===")
        print(f"  Train : {base_counts['train']}")
        print(f"  ID    : {base_counts['ID']}")
        print(f"  OOD   : {base_counts['OOD']}")
        print(f"  Total : {sum(base_counts.values())}\n")

        print("=== ROTATION BUDGETS (per file) ===")
        print(
            f"  Train : {budgets['train']['base']} @ "
            f"{scalar_config.ANGLE_BY_SPLIT['train']['base']}°"
        )
        print(
            f"  ID    : {budgets['ID']['base']} @ "
            f"{scalar_config.ANGLE_BY_SPLIT['ID']['base']}°"
        )
        print(
            f"  OOD   : {budgets['OOD']['dense']} dense @ "
            f"{scalar_config.ANGLE_BY_SPLIT['OOD']['dense']}°"
            f" + {budgets['OOD']['sparse']} sparse @ "
            f"{scalar_config.ANGLE_BY_SPLIT['OOD']['sparse']}°"
        )
        print()

        print("=== EXPECTED OUTPUT COUNTS (incl. originals, pre-dedup) ===")
        print(f"  Train : {expected['train']}")
        print(f"  ID    : {expected['ID']}")
        print(f"  OOD   : {expected['OOD']}")
        print(
            f"  Total : {total_expected} (target {scalar_config.TARGET_TOTAL_FILES})\n"
        )

        train_spacing = scalar_config.ANGLE_BY_SPLIT["train"]["base"]
        train_grid_size = scalar_config.MAX_ROTS_PER_FILE["train"]["base"]
        train_grid = {
            train_spacing: self._generate_uniform_rotations(
                train_spacing,
                self._stable_seed(
                    scalar_config.GLOBAL_SEED, "train-grid", train_spacing
                ),
                train_grid_size,
            )
        }
        train_exclude_qs = np.vstack(list(train_grid.values()))

        print("=== PASS 1: generating Train + ID (ID excluded from Train) ===")
        actual = {"train": 0, "ID": 0, "OOD": 0}
        all_id_quats: List[np.ndarray] = []

        with ProcessPoolExecutor(
            max_workers=scalar_config.max_workers,
            initializer=_init_pool,
            initargs=(
                train_grid,
                train_exclude_qs,
                budgets,
                self.output_root,
            ),
        ) as ex:
            for counts, id_qs in ex.map(_process_train_id, self.xyz_files):
                for k in actual:
                    actual[k] += counts[k]
                if id_qs.size:
                    all_id_quats.append(id_qs)

        id_exclude_qs = (
            np.vstack(all_id_quats)
            if all_id_quats
            else np.empty((0, 4), dtype=np.float64)
        )
        print(
            f"PASS 1 complete. Collected {id_exclude_qs.shape[0]} ID quaternions "
            "for exclusion.\n"
        )

        print("=== PASS 2: generating OOD (excluded from Train ∪ ID) ===")
        train_plus_id_exclude = (
            np.vstack([train_exclude_qs, id_exclude_qs])
            if id_exclude_qs.size
            else train_exclude_qs
        )

        with ProcessPoolExecutor(
            max_workers=scalar_config.max_workers,
            initializer=_init_pool,
            initargs=(
                train_grid,
                train_plus_id_exclude,
                budgets,
                self.output_root,
            ),
        ) as ex:
            for result in ex.map(_process_ood, self.xyz_files):
                for k in actual:
                    actual[k] += result[k]

        total_actual = sum(actual.values())
        print("\n=== ACTUAL OUTPUT COUNTS (post-dedup) ===")
        print(f"  Train : {actual['train']}")
        print(f"  ID    : {actual['ID']}")
        print(f"  OOD   : {actual['OOD']}")
        print(f"  Total : {total_actual} (target {scalar_config.TARGET_TOTAL_FILES})")

        if (
            abs(total_actual - scalar_config.TARGET_TOTAL_FILES)
            > 0.10 * scalar_config.TARGET_TOTAL_FILES
        ):
            print("\n[NOTE] Total differs from target by >10%. Common reasons:")
            print("  - Dedup collisions (high symmetry can collapse rotations).")
            print(
                "  - Uneven base_counts or low per-file budgets from "
                "the target split fractions."
            )
            print(
                "Try tweaking TARGET_TOTAL_FILES, SPLIT_FRACTIONS, "
                "MAX_ROTS_PER_FILE, or angles."
            )

        return actual


def run_scalar_quaternions(
    xyz_files: List[Path], output_root: Path, label: str = "scalar"
) -> Dict[str, int]:
    """
    Run scalar quaternion generation (TARGET_TOTAL_FILES, MAX_ROTS_PER_FILE, etc.).

    Uses create_scalar.config via scalar_config. Returns actual counts per split.
    """
    gen = ScalarQuaternionGenerator(xyz_files, output_root, label=label)
    return gen.generate()
