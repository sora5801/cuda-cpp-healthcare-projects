#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL structures for SASA (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
#
# This project's "real data" is a molecular structure (a PDB file) converted to
# the simple "<element> x y z" format the loader reads. This script prints the
# recipe; it requires no credentials and downloads nothing on its own beyond an
# OPTIONAL public PDB fetch you can uncomment. It defers to make_synthetic.py for
# the fully-offline stand-in (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.31 -- Solvent-Accessible Surface Area (SASA) on GPU"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real SASA runs on an actual structure. To prepare one:"
echo "  1) Pick a PDB id, e.g. 1CRN (crambin, a 46-residue test protein)."
echo "  2) Download the structure (public, no login) from RCSB:"
echo "       https://files.rcsb.org/download/1CRN.pdb"
echo "     (Optional one-liner you can run yourself:)"
echo "       curl -L https://files.rcsb.org/download/1CRN.pdb -o \"$DATA_DIR/1CRN.pdb\""
echo "  3) Convert ATOM/HETATM records to '<element> x y z' (Angstrom). The"
echo "     element is PDB columns 77-78; coords are columns 31-54. Tools that"
echo "     do this cleanly: Biopython (Bio.PDB) or MDTraj."
echo "  4) Validate your SASA against FreeSASA (https://github.com/mittinatten/freesasa)."
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py    # writes data/sample/molecule_sample.xyz"
echo
echo "Idempotency note: when wiring a real fetch, skip the download if the file"
echo "already exists with the expected SHA256, and NEVER bypass any registration."
