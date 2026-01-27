"""
Central configuration for scalar dataset generation.

Uses settings from utils/generate_quaternions.py (r_splits, angles, margins,
TARGET_TOTAL_FILES, SPLIT_FRACTIONS, MAX_ROTS_PER_FILE). Paths are scalar-specific.
"""

from pathlib import Path

from scipy.spatial.transform import Rotation as R

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS (scalar-specific)
# ═══════════════════════════════════════════════════════════════════════════════
RAW_DATA_DIR = Path("scalar_raw")
OUTPUT_DIR = Path("scalar")

QUATERNIONS_SUBDIR = "quaternions"
UNIT_CELLS_SUBDIR = "unit_cells"

# ═══════════════════════════════════════════════════════════════════════════════
# FROM utils/generate_quaternions.py
# ═══════════════════════════════════════════════════════════════════════════════
r_values = list(range(10, 31))  # R10..R30

r_splits = {
    "ID": [13, 15, 17, 20, 24, 27],
    "OOD": [10, 11, 29, 30],
}

# Target total generated files (including originals)
TARGET_TOTAL_FILES = 100_000

# Split budget fractions (must sum to 1.0)
SPLIT_FRACTIONS = {
    "train": 0.60,
    "ID": 0.25,
    "OOD": 0.15,
}

# Angle regimes (split-specific). OOD has both dense and sparse.
ANGLE_BY_SPLIT = {
    "train": {"base": 22.0},
    "ID": {"base": 18.0},
    "OOD": {"dense": 16.0, "sparse": 28.0},
}

# Caps: hard per-file limits so counts don't blow up
MAX_ROTS_PER_FILE = {
    "train": {"base": 60},
    "ID": {"base": 50},
    "OOD": {"dense": 35, "sparse": 25},
}

# Enforced angular margins (geodesic angle on SO(3)) from reference sets
MARGINS = {
    "ID": 8.0,
    "OOD": 8.0,
    "train": 0.0,
}

# Fixed offsets (Euler xyz degrees)
id_offset_euler = [12, 16, 24]
ood_offset_euler = [30, 50, 70]
ID_OFFSET = R.from_euler("xyz", id_offset_euler, degrees=True)
OOD_OFFSET = R.from_euler("xyz", ood_offset_euler, degrees=True)

GLOBAL_SEED = 1337
max_workers = 4
coord_tolerance = 1e-6
