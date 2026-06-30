#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This teaching project needs NO download --
# it builds its own tiny synthetic coarse-grained membrane patch
# (scripts/make_synthetic.py + the in-code build_system()). This script only
# points at the real membrane databases the catalog names, for further study.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.19 -- Membrane Protein Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs entirely on a SYNTHETIC sample -- no download needed."
echo "  Regenerate / resize the committed sample with:"
echo "    python scripts/make_synthetic.py --n-lipids 32 --n-prot 7 --steps 400"
echo
echo "Real-world membrane-protein resources (for further study; not auto-fetched"
echo "because they need force-field setup tools like CHARMM-GUI):"
echo "  * MemProtMD  -- 3133 membrane proteins in bilayers : https://memprotmd.bioch.ox.ac.uk"
echo "  * GPCRdb     -- GPCR structures and MD data         : https://gpcrdb.org"
echo "  * OPM        -- orientations of proteins in membranes: https://opm.phar.umich.edu"
echo "  * CGMD Platform benchmark systems                   : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/"
echo
echo "  These are atomistic/structural sets. Building a runnable MD system from"
echo "  them requires CHARMM-GUI Membrane Builder or packmol-memgen (see README)."
