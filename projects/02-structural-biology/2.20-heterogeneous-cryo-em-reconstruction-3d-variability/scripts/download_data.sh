#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. This REDUCED-SCOPE teaching
# project runs on a committed SYNTHETIC sample, so this script only prints
# pointers to the real datasets and defers to scripts/make_synthetic.py for a
# larger offline stand-in. It downloads nothing.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a SYNTHETIC sample (data/sample/volumes.txt);"
echo "no download is required to run the demo. The real heterogeneous cryo-EM"
echo "datasets the catalog points at are large and need preprocessing:"
echo
echo "  EMPIAR-10180 (spliceosome), EMPIAR-10076 (80S ribosome),"
echo "  EMPIAR-10028 (TRPV1)              -> https://www.ebi.ac.uk/empiar/"
echo "  cryoDRGN benchmark sets + tooling -> https://github.com/ml-struct-bio/cryodrgn"
echo
echo "  EMPIAR entries are openly downloadable but tens-to-hundreds of GB; turning"
echo "  a particle stack into per-particle volumes (CTF, poses, back-projection)"
echo "  is upstream of this project. Respect each dataset's license."
echo
echo "  For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --n 64 --g 8"
