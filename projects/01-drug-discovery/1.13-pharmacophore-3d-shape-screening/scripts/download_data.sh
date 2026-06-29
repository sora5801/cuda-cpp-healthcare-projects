#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point to the FULL dataset (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.13 : Pharmacophore & 3D Shape Screening
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real 3D conformer libraries are large
# and/or licensed and come as SDF/MOL2 (not this project's simple text format),
# so this script PRINTS instructions + links rather than blindly downloading
# gigabytes; for an offline run, the committed synthetic sample suffices.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.13 -- Pharmacophore & 3D Shape Screening"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a tiny SYNTHETIC sample in data/sample/ that is enough"
echo "to build and run the demo offline. No download is required."
echo
echo "To screen REAL molecules, obtain 3D conformers from a public library:"
echo "  * ZINC20 conformers : https://zinc20.docking.org   (free for research)"
echo "  * DUD-E             : https://dude.docking.org      (actives + decoys, 3D)"
echo "  * Enamine REAL      : https://enamine.net           (make-on-demand library)"
echo
echo "Those come as SDF/MOL2. Convert to this project's 'x y z radius' text format"
echo "with a short RDKit script (read 3D coordinates, map each element to its van"
echo "der Waals radius), then run:"
echo "    ./build/cmake/pharmacophore-3d-shape-screening <your_file.txt>"
echo
echo "For a larger SYNTHETIC problem with no downloads, regenerate the sample:"
echo "    python scripts/make_synthetic.py"
echo
echo "Respect each source's license; none are redistributed in this repo."
