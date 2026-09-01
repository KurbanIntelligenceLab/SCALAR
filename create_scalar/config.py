from pathlib import Path

from scipy.spatial.transform import Rotation as R

RAW_DATA_DIR = Path("scalar_raw")
OUTPUT_DIR = Path("scalar")

QUATERNIONS_SUBDIR = "quaternions"
UNIT_CELLS_SUBDIR = "unit_cells"

r_values = list(range(10, 31))

r_splits = {
    "ID": [13, 15, 17, 20, 24, 27],
    "OOD": [10, 11, 29, 30],
}

TARGET_TOTAL_FILES = 100_000

SPLIT_FRACTIONS = {
    "train": 0.60,
    "ID": 0.25,
    "OOD": 0.15,
}

ANGLE_BY_SPLIT = {
    "train": {"base": 22.0},
    "ID": {"base": 18.0},
    "OOD": {"dense": 16.0, "sparse": 28.0},
}

MAX_ROTS_PER_FILE = {
    "train": {"base": 60},
    "ID": {"base": 50},
    "OOD": {"dense": 35, "sparse": 25},
}

MARGINS = {
    "ID": 8.0,
    "OOD": 8.0,
    "train": 0.0,
}

id_offset_euler = [12, 16, 24]
ood_offset_euler = [30, 50, 70]
ID_OFFSET = R.from_euler("xyz", id_offset_euler, degrees=True)
OOD_OFFSET = R.from_euler("xyz", ood_offset_euler, degrees=True)

GLOBAL_SEED = 1337
max_workers = 4
coord_tolerance = 1e-6

CARVE_R_MIN = 10
CARVE_R_MAX = 30
CARVE_DELTA_BOX = 10.0
CARVE_SYMPREC = 1e-3
CARVE_COORD_DECIMALS = 8
