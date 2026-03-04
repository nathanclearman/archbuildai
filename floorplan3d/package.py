"""
Package the FloorPlan3D Blender add-on into a .zip for installation.

Usage:
    python package.py                  # Creates floorplan3d_addon.zip
    python package.py --output my.zip  # Custom output filename
"""

import argparse
import os
import zipfile
from pathlib import Path


ADDON_DIR = Path(__file__).parent / "blender_addon"
EXCLUDE_PATTERNS = {"__pycache__", ".pyc", ".pyo", ".DS_Store"}


def should_include(path):
    """Check if a file should be included in the package."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in str(path):
            return False
    return True


def package_addon(output_path):
    """Create a zip package of the Blender add-on."""
    output_path = Path(output_path)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ADDON_DIR):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for filename in files:
                filepath = Path(root) / filename
                if not should_include(filepath):
                    continue

                # Archive path: floorplan3d/relative_path
                arcname = "floorplan3d" / filepath.relative_to(ADDON_DIR)
                zf.write(filepath, arcname)

    print(f"Add-on packaged: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Install in Blender:")
    print(f"  1. Open Blender > Edit > Preferences > Add-ons")
    print(f"  2. Click 'Install...' and select {output_path.name}")
    print(f"  3. Enable 'FloorPlan3D' in the add-ons list")
    print(f"  4. Open the N-panel (press N) > FloorPlan3D tab")


def main():
    parser = argparse.ArgumentParser(description="Package FloorPlan3D Blender add-on")
    parser.add_argument(
        "--output", "-o",
        default="floorplan3d_addon.zip",
        help="Output zip file path",
    )
    args = parser.parse_args()

    package_addon(args.output)


if __name__ == "__main__":
    main()
