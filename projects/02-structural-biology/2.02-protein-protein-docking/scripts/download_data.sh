#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real protein-complex sources (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.2 : Protein-Protein Docking. Downloads nothing automatically.
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs,
# and NEVER bypasses credentials/registration. The committed synthetic sample is
# enough to run the demo offline; this script points to the real benchmarks.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 2.2 -- Protein-Protein Docking"
echo
echo "Real rigid-body docking benchmarks (free for research; please cite):"
echo "  Docking Benchmark 5.5 : https://zlab.umassmed.edu/benchmark/"
echo "                          230 non-redundant complexes (bound + unbound)."
echo "  SAbDab (antibodies)   : https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/"
echo "  PDB (any complex)     : https://www.rcsb.org   (split chains -> receptor/ligand)"
echo
echo "To dock a real complex with this code:"
echo "  1) download a PDB/mmCIF complex and split it into two chains."
echo "  2) write each chain's atoms as 'x y z' lines (Angstrom)."
echo "  3) prepend a header 'n_recv n_lig N spacing' (OMIT the known-answer"
echo "     fields -- a real complex has no pre-known rigid translation)."
echo "  See data/README.md for the exact file format."
echo
echo "No-download synthetic option (works offline):"
echo "  python scripts/make_synthetic.py --N 48 --spacing 1.5"
echo
echo "Target data dir: $PROJECT_ROOT/data"
