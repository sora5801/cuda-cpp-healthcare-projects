#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URL,
# and NEVER bypasses credentials/registration. This project's demo runs on a
# committed SYNTHETIC sample (data/sample/complex_sample.txt), so there is no
# mandatory download -- this script explains how to obtain REAL complexes from
# the PDB and how the offline sample is (re)generated.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.21 -- Protein-Nucleic Acid Docking & Co-Folding"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project needs NO download to run: data/sample/complex_sample.txt"
echo "is a committed SYNTHETIC complex with a known native pose."
echo
echo "To regenerate (or resize) the synthetic sample:"
echo "    python scripts/make_synthetic.py --spacing 3500"
echo
echo "To work with REAL protein-nucleic-acid complexes:"
echo "  * Protein Data Bank (PDB): https://www.rcsb.org"
echo "      Download a structure, e.g. 1FNT, as mmCIF/PDB:"
echo "      https://files.rcsb.org/download/1FNT.cif"
echo "  * RNA-Puzzles benchmarks:  https://github.com/RNA-Puzzles"
echo "  * Rfam RNA families:       https://rfam.org"
echo
echo "  You must convert a downloaded structure into this loader's integer"
echo "  format (extract atoms, assign charge signs in {-1,0,+1}, scale"
echo "  coordinates to milli-Angstrom). See data/README.md for the format."
echo "  No registration or credentials are required for the public PDB; if a"
echo "  benchmark needs an account, follow its site's instructions -- this"
echo "  script will not bypass them."
