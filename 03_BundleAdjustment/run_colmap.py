"""
COLMAP 3D Reconstruction Pipeline
Runs the full COLMAP pipeline: Feature Extraction → Matching → Sparse → Dense.

Usage:
    python run_colmap.py [--skip-dense] [--colmap-bin COLMAP_PATH]

Requirements:
    - COLMAP installed and accessible in PATH (or specify --colmap-bin)
    - CUDA-capable GPU for dense reconstruction (optional, use --skip-dense to skip)
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def find_colmap():
    """Try to find the COLMAP executable."""
    colmap = shutil.which("colmap")
    if colmap is not None:
        return colmap
    # Common Windows install locations
    for p in [
        "C:\\Program Files\\COLMAP\\bin\\colmap",
        "C:\\COLMAP\\bin\\colmap",
        os.path.expandvars("%LOCALAPPDATA%\\COLMAP\\bin\\colmap"),
    ]:
        exe = f"{p}.exe" if not p.endswith(".exe") else p
        if os.path.exists(exe):
            return exe
    return None


def run_cmd(cmd, desc=""):
    """Run a command and print its output in real-time."""
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="COLMAP 3D Reconstruction Pipeline")
    parser.add_argument("--dataset", default="data", help="Dataset directory (default: data)")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense reconstruction")
    parser.add_argument("--colmap-bin", default=None, help="Path to COLMAP executable")
    parser.add_argument("--camera-model", default="PINHOLE", help="Camera model (default: PINHOLE)")
    parser.add_argument("--single-camera", type=int, default=1, help="Use single camera for all images (default: 1)")
    args = parser.parse_args()

    # Resolve COLMAP binary
    colmap_bin = args.colmap_bin or find_colmap()
    if colmap_bin is None:
        print("ERROR: COLMAP not found. Please install COLMAP or specify --colmap-bin.")
        print("Download from: https://github.com/colmap/colmap/releases")
        print("Extract COLMAP-dev-windows-cuda.zip and add bin/ to PATH, or use --colmap-bin <path>")
        sys.exit(1)
    print(f"Using COLMAP: {colmap_bin}")

    # Setup paths
    dataset = Path(args.dataset)
    image_path = dataset / "images"
    colmap_path = dataset / "colmap"
    sparse_path = colmap_path / "sparse"
    dense_path = colmap_path / "dense"
    database_path = colmap_path / "database.db"

    if not image_path.exists():
        print(f"ERROR: Image path not found: {image_path}")
        sys.exit(1)

    os.makedirs(sparse_path, exist_ok=True)
    if not args.skip_dense:
        os.makedirs(dense_path, exist_ok=True)

    # Step 1: Feature Extraction
    run_cmd(
        [
            colmap_bin, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", args.camera_model,
            "--ImageReader.single_camera", str(args.single_camera),
        ],
        desc="Step 1: Feature Extraction"
    )

    # Step 2: Feature Matching (Exhaustive)
    run_cmd(
        [
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(database_path),
        ],
        desc="Step 2: Exhaustive Feature Matching"
    )

    # Step 3: Sparse Reconstruction (Mapper / Bundle Adjustment)
    run_cmd(
        [
            colmap_bin, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(sparse_path),
        ],
        desc="Step 3: Sparse Reconstruction (Bundle Adjustment)"
    )

    if args.skip_dense:
        print("\n=== Skipping dense reconstruction (--skip-dense) ===")
        print("Results:")
        print(f"  Database: {database_path}")
        print(f"  Sparse:   {sparse_path}")
        print("\nAll COLMAP steps completed!")
        return

    # Check sparse output exists
    sparse_recon = sparse_path / "0"
    if not sparse_recon.exists():
        print(f"ERROR: Sparse reconstruction not found at {sparse_recon}")
        print("Available entries:", list(sparse_path.iterdir()))
        sys.exit(1)

    # Step 4: Image Undistortion
    run_cmd(
        [
            colmap_bin, "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(sparse_recon),
            "--output_path", str(dense_path),
        ],
        desc="Step 4: Image Undistortion"
    )

    # Step 5: Patch Match Stereo
    run_cmd(
        [
            colmap_bin, "patch_match_stereo",
            "--workspace_path", str(dense_path),
        ],
        desc="Step 5: Dense Reconstruction (Patch Match Stereo)"
    )

    # Step 6: Stereo Fusion
    run_cmd(
        [
            colmap_bin, "stereo_fusion",
            "--workspace_path", str(dense_path),
            "--output_path", str(dense_path / "fused.ply"),
        ],
        desc="Step 6: Stereo Fusion → fused.ply"
    )

    print("\n" + "=" * 60)
    print("All COLMAP steps completed!")
    print("=" * 60)
    print("Results:")
    print(f"  Database:       {database_path}")
    print(f"  Sparse (cams):  {sparse_recon}")
    print(f"  Dense (PLY):    {dense_path / 'fused.ply'}")
    print(f"\n  You can view the dense PLY file with MeshLab: https://www.meshlab.net/")


if __name__ == "__main__":
    main()
