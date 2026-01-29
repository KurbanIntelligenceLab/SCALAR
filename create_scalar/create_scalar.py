#!/usr/bin/env python3
"""
Create scalar dataset from raw data.

Takes scalar_raw (directory or zip file) as input and outputs scalar/ with:
- unit_cells/  – CIF files
- quaternions/ – rotated XYZ structures (Material/Rx/xyz/rot_*.xyz)

Uses create_scalar.config and create_scalar.generate_quaternions (TARGET_TOTAL_FILES,
SPLIT_FRACTIONS, MAX_ROTS_PER_FILE, etc.; no create_c2np).

Usage:
    python create_scalar.py --raw-data scalar_raw --output scalar
    python create_scalar.py --raw-data scalar_raw.zip --output scalar

Final output structure:
    scalar/
    ├── quaternions/
    │   └── {Material}/
    │       └── R{x}/
    │           └── xyz/
    │               └── rot_*.xyz
    └── unit_cells/
        └── {Material}.cif
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from create_scalar.config import QUATERNIONS_SUBDIR, UNIT_CELLS_SUBDIR
from create_scalar.generate_quaternions import run_scalar_quaternions


def _is_macos_metadata(path: Path) -> bool:
    """Check if path is a macOS metadata file (AppleDouble or __MACOSX)."""
    return path.name.startswith("._") or "__MACOSX" in path.parts


def extract_cifs(raw_data_dir: Path, output_dir: Path) -> int:
    """Extract all CIF files from raw_data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    cifs_subdir = raw_data_dir / "cifs"
    search_dir = cifs_subdir if cifs_subdir.exists() else raw_data_dir

    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".cif") and not f.startswith("._"):
                full_path = Path(root) / f
                if "__MACOSX" not in full_path.parts:
                    shutil.copy2(os.path.join(root, f), output_dir / f)
                    count += 1
    return count


def extract_xyz_files(raw_data_dir: Path, output_dir: Path) -> int:
    """Extract XYZ files from raw_data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    materials_subdir = raw_data_dir / "materials"
    search_dir = materials_subdir if materials_subdir.exists() else raw_data_dir

    if not search_dir.exists():
        return 0

    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(".xyz") and not file.startswith("._"):
                xyz_path = Path(root) / file
                if "__MACOSX" in xyz_path.parts:
                    continue
                dest_path = output_dir / file
                if dest_path.exists() and xyz_path != dest_path:
                    rel_path = xyz_path.relative_to(search_dir)
                    dest_path = output_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(xyz_path, dest_path)
                count += 1
    return count


def extract_zip_if_needed(raw_data_path: Path) -> tuple[Path, bool]:
    """
    Extract zip file if needed, return path to data directory and cleanup flag.

    Returns:
        Tuple of (data_path, needs_cleanup)
    """
    if raw_data_path.suffix.lower() in (".zip", ".zipx"):
        print(f"  Detected zip file: {raw_data_path.name}")
        print("  Extracting to temporary directory...")
        temp_extract = tempfile.mkdtemp(prefix="scalar_raw_")
        with zipfile.ZipFile(raw_data_path, "r") as zf:
            zf.extractall(temp_extract)
        return Path(temp_extract), True
    return raw_data_path, False


def create_scalar(raw_data_dir: str = "scalar_raw", output_dir: str = "scalar") -> bool:
    """
    Create scalar dataset from raw data (directory or zip file).

    Args:
        raw_data_dir: Path to raw data directory or zip file (scalar_raw or scalar_raw.zip)
        output_dir: Path to output scalar directory
    """
    print("\n" + "=" * 60)
    print("Creating scalar dataset from raw data")
    print("=" * 60)
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Raw data: {raw_data_dir}")
    print(f"  Output: {output_dir}")

    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    unit_cells_dir = output_path / UNIT_CELLS_SUBDIR
    quaternions_dir = output_path / QUATERNIONS_SUBDIR

    if not raw_path.exists():
        zip_path = Path(f"{raw_data_dir}.zip")
        if zip_path.exists():
            print(f"  Path '{raw_data_dir}' not found, using '{zip_path.name}' instead")
            raw_path = zip_path
        else:
            print(f"\n  [ERROR] Raw data path not found: {raw_data_dir}")
            print(f"  Also checked: {zip_path}")
            return False

    data_path, needs_cleanup = extract_zip_if_needed(raw_path)

    try:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 60)
        print("Step 1: Extracting CIF files")
        print("-" * 60)
        cif_count = extract_cifs(data_path, unit_cells_dir)
        print(f"  Extracted {cif_count} CIF files to {unit_cells_dir}")

        print("\n" + "-" * 60)
        print("Step 2: Extracting XYZ files")
        print("-" * 60)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_materials = Path(temp_dir) / "materials"
            xyz_count = extract_xyz_files(data_path, temp_materials)
            print(f"  Extracted {xyz_count} XYZ files")

            print("\n" + "-" * 60)
            print("Step 3: Generating quaternion rotations")
            print("-" * 60)

            if xyz_count == 0:
                print("  [WARN] No XYZ files found, skipping quaternion generation")
            else:
                xyz_files = sorted(
                    f for f in temp_materials.rglob("*.xyz")
                    if not _is_macos_metadata(f)
                )
                run_scalar_quaternions(
                    xyz_files,
                    quaternions_dir,
                    label="scalar",
                )

        mat_count = (
            len([d for d in quaternions_dir.iterdir() if d.is_dir()])
            if quaternions_dir.exists()
            else 0
        )

        print("\n" + "=" * 60)
        print("scalar dataset created successfully!")
        print("=" * 60)
        print(f"\n  Output: {output_path.resolve()}")
        print(f"\n  {output_path}/")
        print(f"  ├── {QUATERNIONS_SUBDIR}/  ({mat_count} materials)")
        print(f"  └── {UNIT_CELLS_SUBDIR}/   ({cif_count} CIF files)")

        return True
    finally:
        if needs_cleanup and data_path.exists():
            print("\n  Cleaning up temporary extraction directory...")
            shutil.rmtree(data_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create scalar dataset from raw data (scalar_raw dir or zip)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m create_scalar.create_scalar
    python -m create_scalar.create_scalar --raw-data scalar_raw --output scalar
    python -m create_scalar.create_scalar --raw-data scalar_raw.zip --output scalar
        """,
    )
    parser.add_argument(
        "--raw-data",
        "-r",
        default="scalar_raw",
        help="Path to raw data directory or zip file (default: scalar_raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scalar",
        help="Path to output directory (default: scalar)",
    )

    args = parser.parse_args()
    success = create_scalar(
        raw_data_dir=args.raw_data,
        output_dir=args.output,
    )
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
