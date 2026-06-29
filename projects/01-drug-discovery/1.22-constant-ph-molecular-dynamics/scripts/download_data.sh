#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project ships a committed SYNTHETIC
# sample (data/sample/cph_system.txt) that is enough for the demo, so there is
# nothing to auto-download; this script only points at the real benchmarks.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.22 -- Constant-pH Molecular Dynamics"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs entirely on the committed SYNTHETIC sample:"
echo "    data/sample/cph_system.txt   (regenerate: python scripts/make_synthetic.py)"
echo
echo "There is NO automatic download. To extend the project with real"
echo "experimental pKa values, obtain them yourself from:"
echo "  * PKAD  -- experimental protein-residue pKa database:"
echo "      https://compbio.clemson.edu/pkad/"
echo "  * PHMD / benchmark pKa sets for Asp/Glu/His/Cys/Lys residues"
echo "      (see the constant-pH MD literature, e.g. AMBER CpHMD papers)."
echo "  * DrugBank -- ionizable-group compounds (registration required):"
echo "      https://go.drugbank.com"
echo
echo "Respect each source's license. For credentialed sets, register on the"
echo "site -- this script will NOT bypass authentication. Map a real residue into"
echo "the toy by editing pKa_intrinsic / charges / positions in the sample file"
echo "(field meanings are in data/README.md)."
