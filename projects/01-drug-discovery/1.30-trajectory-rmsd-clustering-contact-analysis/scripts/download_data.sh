#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URLs +
# how to fetch, and NEVER bypasses credentials/registration. The demo runs fully
# offline on the committed synthetic sample, so this script only prints guidance
# and pointers to the real molecular-dynamics trajectory archives.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.30 -- Trajectory RMSD, Clustering & Contact Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/trajectory_sample.txt) is SYNTHETIC and is"
echo "all the demo needs -- no download is required to build, run, or verify."
echo
echo "Real molecular-dynamics trajectories to analyze with the same pipeline"
echo "(you will need to adapt the loader to the real file format + atom count):"
echo "  * MDCATH  (curated all-atom trajectories):"
echo "        https://huggingface.co/datasets/compsciencelab/mdcath"
echo "  * GPCRmd  (GPCR molecular-dynamics database):  https://gpcrmd.org"
echo "  * MDDB    (molecular-dynamics database):       https://www.mddbr.eu"
echo "  * PDB trajectory depositions (RCSB / PDB-Dev): https://www.rcsb.org"
echo
echo "These archives may require account registration and carry their own licenses."
echo "Respect every license; this script does NOT attempt to bypass any login."
echo
echo "For a larger SYNTHETIC trajectory (no download, fully offline), run:"
echo "    python scripts/make_synthetic.py --frames 100000"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print the source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
