#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL 4D-CT data (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.19 : Motion-Compensated 4D-CT Reconstruction. Downloads nothing.
#
# CONTRACT (CLAUDE.md section 8): documented, and NEVER bypasses credentials.
# The committed sample is synthetic, so this only prints where the real datasets
# live and how production tools consume them.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.19 -- Motion-Compensated 4D-CT Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Options for real 4D-CT data (a phase-binned sinogram / projection set):"
echo "  * DIR-Lab 4D-CT lung (https://www.dir-lab.com/) -- 10 cases with expert"
echo "    landmark pairs; the standard deformable-registration benchmark."
echo "  * TCIA 4D-CT lung radiotherapy (https://www.cancerimagingarchive.net) --"
echo "    real DICOM; registration may be required. This script does NOT bypass it."
echo "  * POPI model (https://www.creatis.insa-lyon.fr/rio/popi-model) -- a"
echo "    point-validated breathing 4D-CT dataset."
echo "  * Toolkits with 4D reconstruction + sample data: RTK (ROOSTER), ASTRA,"
echo "    TIGRE, Plastimatch. Forward-project into this sinogram format."
echo
echo "Offline stand-in (no download, reproducible, SYNTHETIC):"
echo "  python scripts/make_synthetic.py --phases 10 --ang-phase 12 --det 257 --img 128"
echo
echo "When wiring a real dataset, keep it idempotent:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for credentialed sets, print registration instructions ONLY"
