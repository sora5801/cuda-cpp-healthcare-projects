#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic organ-mesh pointers (Linux/macOS)
# Project 10.02 : Real-Time Soft-Tissue Deformation. Nothing to download.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 10.02 -- Real-Time Soft-Tissue Deformation (PBD)"
echo
echo "There is no file to download: the mesh is built from the parameters in"
echo "data/sample/cloth_params.txt."
echo
echo "For REAL organ meshes + GPU PBD frameworks:"
echo "  SOFA  : https://github.com/sofa-framework/sofa  (physics + haptics)"
echo "  iMSTK : https://github.com/Kitware/iMSTK        (CUDA deformation)"
echo "  FleX  : https://github.com/NVIDIAGameWorks/FleX (GPU PBD particles)"
echo "  Patient meshes: segment CT/MRI (e.g., 3D Slicer) into tetra/surface meshes."
echo
echo "Bigger mesh (no download):"
echo "  python scripts/make_synthetic.py --R 128 --C 128 --steps 600"
echo
echo "Target data dir: $PROJECT_ROOT/data"
