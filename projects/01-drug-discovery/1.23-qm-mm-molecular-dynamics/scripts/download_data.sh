#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real QM/MM data + tool pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. THERE IS NOTHING TO DOWNLOAD for this demo:
# the ensemble is generated from data/sample/ensemble_params.txt by the program.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 1.23 -- QM/MM Molecular Dynamics"
echo
echo "There is NO file to download: the program derives every trajectory's"
echo "(field, x0) from the sweep in data/sample/ensemble_params.txt, and the"
echo "model potential-energy surface is built analytically in src/qmmm.h."
echo
echo "For REAL QM/MM (enzyme reactions, covalent inhibitors, proton wires):"
echo "  Enzyme-drug complexes (PDB) : https://www.rcsb.org"
echo "  Enzyme reaction database    : https://www.brenda-enzymes.org"
echo "  SAMPL blind-challenge sets  : https://github.com/samplchallenges"
echo
echo "Production GPU QM/MM engines to graduate to:"
echo "  AMBER + QUICK (GPU DFT) : https://github.com/merzlab/QUICK"
echo "  TeraChem (GPU DFT)      : https://www.petachem.com"
echo "  OpenMM + PySCF QM/MM    : https://github.com/openmm/openmm"
echo "  CP2K (periodic QM/MM)   : https://github.com/cp2k/cp2k"
echo
echo "Bigger SYNTHETIC ensemble (no download):"
echo "  python scripts/make_synthetic.py --nf 64 --nx 64"
echo
echo "Target data dir: $PROJECT_ROOT/data"
