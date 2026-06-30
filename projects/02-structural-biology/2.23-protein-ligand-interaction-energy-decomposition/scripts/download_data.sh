#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.23 : Protein-Ligand Interaction Energy Decomposition
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.23 -- Protein-Ligand Interaction Energy Decomposition"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# This teaching project runs entirely on the committed SYNTHETIC sample (no real
# structure/force-field parser is shipped). Real per-residue MM-GBSA needs a full
# MD+parameter stack (AMBER prmtop or GROMACS top + a trajectory), so this script
# prints where to obtain real complexes and how to feed them through the proper
# tools, rather than pretending to download a ready-to-run file.
echo "This project ships a SYNTHETIC sample only (data/sample/complex_sample.txt)."
echo "The committed tiny sample is enough to run the demo offline."
echo
echo "For a larger SYNTHETIC system, regenerate with more residues/frames:"
echo "    python scripts/make_synthetic.py --residues 200 --frames 500"
echo
echo "To work with REAL protein-ligand complexes, obtain structures from:"
echo "    PDBbind : http://www.pdbbind.org.cn   (curated complexes + affinities)"
echo "    KLIFS   : https://klifs.net           (kinase-ligand structures)"
echo "    ChEMBL  : https://www.ebi.ac.uk/chembl/ (activity data for target families)"
echo "    ClinVar : https://www.ncbi.nlm.nih.gov/clinvar/ (resistance mutations)"
echo
echo "Then produce per-residue MM-GBSA inputs with a proper toolchain (study these):"
echo "    AMBER MMPBSA.py decomp : https://ambermd.org/AmberTools.php"
echo "    gmx_MMPBSA            : https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA"
echo "Respect each source's license; do not redistribute restricted structures."
