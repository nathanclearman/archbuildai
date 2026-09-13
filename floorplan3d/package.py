"""
Package the ArchbuildAI add-on as a Blender extension zip (Blender 4.2+).

Usage:
    python package.py                  # archbuildai_addon.zip, no YOLO weights
    python package.py --with-weights   # also bundle blender_addon/weights/*.pt (~100 MB)
    python package.py --output my.zip

Layout inside the zip (extension layout: manifest at the root):
    blender_manifest.toml, __init__.py, operators.py, ..., api/, weights/ (opt),
    vlm/   <- the VLM runtime copied from ../model/ (inference.py, prompts.py,
              schema.py, cv_walls.py, claude_refiner.py) so the shipped add-on
              needs no repo checkout. The fine-tuned adapter is never bundled:
              Hybrid uses the base model only; the Qwen-only backend takes an
              adapter folder from the add-on preferences.

Install: Blender > Edit > Preferences > Get Extensions > (v) > Install from Disk.
"""

import argparse
import os
import zipfile
from pathlib import Path


ADDON_DIR = Path(__file__).parent / "blender_addon"
MODEL_DIR = Path(__file__).parent / "model"
VLM_RUNTIME_FILES = ["inference.py", "prompts.py", "schema.py", "cv_walls.py", "claude_refiner.py"]
EXCLUDE_DIRS = {"__pycache__", "vlm"}          # vlm/ is regenerated from model/ below
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAMES = {".DS_Store"}


def should_include(path, with_weights=False):
    """Check if a file should be included in the package."""
    path = Path(path)
    if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.suffix == ".pt" and not with_weights:
        return False
    return True


def package_addon(output_path, with_weights=False):
    """Create the extension zip."""
    output_path = Path(output_path)
    manifest = ADDON_DIR / "blender_manifest.toml"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}")
    missing = [f for f in VLM_RUNTIME_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        raise SystemExit(f"VLM runtime files missing from {MODEL_DIR}: {missing}")

    n_files = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ADDON_DIR):
            skip = set(EXCLUDE_DIRS) | (set() if with_weights else {"weights"})
            dirs[:] = [d for d in dirs if d not in skip]
            for filename in files:
                filepath = Path(root) / filename
                if not should_include(filepath, with_weights):
                    continue
                zf.write(filepath, str(filepath.relative_to(ADDON_DIR)))
                n_files += 1
        for f in VLM_RUNTIME_FILES:
            zf.write(MODEL_DIR / f, f"vlm/{f}")
            n_files += 1

    print(f"Extension packaged: {output_path}  ({n_files} files, "
          f"{output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("Install in Blender 4.2+: Edit > Preferences > Get Extensions > v > Install from Disk")
    print("Then: Preferences > Add-ons > ArchbuildAI > install packages / download base model")


def main():
    parser = argparse.ArgumentParser(description="Package the ArchbuildAI Blender extension")
    parser.add_argument("--output", "-o", default="archbuildai_addon.zip", help="Output zip path")
    parser.add_argument("--with-weights", action="store_true",
                        help="Bundle blender_addon/weights/*.pt (YOLO geometry backend) into the zip")
    args = parser.parse_args()
    package_addon(args.output, with_weights=args.with_weights)


if __name__ == "__main__":
    main()
