#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point at the real datasets (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 2.30 : Protein Solubility & Phase Separation Simulation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The LLPS resources below are sequence/
# annotation DATABASES, not ready-to-run particle configurations -- so there is
# no single binary to download for this simulation. This script prints where the
# real data lives and defers to make_synthetic.py for the offline, runnable
# coarse-grained system the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.30 -- Protein Solubility & Phase Separation Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "  Real-world LLPS / IDP resources (databases, not simulation inputs):"
echo "    * PhaSePro   -- proteins undergoing LLPS    https://phasepro.elte.hu"
echo "    * PhaSepDB   -- phase-separation database    http://db.phasep.pro"
echo "    * DisProt    -- intrinsically disordered     https://disprot.org"
echo "    * FuzDB      -- fuzzy protein complexes       https://fuzdb.org"
echo "    * CALVADOS   -- residue-level IDP force field https://github.com/KULL-Centre/CALVADOS"
echo
echo "  These give SEQUENCES and per-residue stickiness scales (Kapcha-Rossky,"
echo "  HPS, CALVADOS). To build a runnable system from a sequence you map each"
echo "  residue to its lambda, place beads on a chain, and feed the loader format"
echo "  in data/README.md -- exactly what make_synthetic.py does with synthetic"
echo "  stickiness values."
echo
echo "  The committed tiny sample in data/sample/system.txt runs the demo offline."
echo "  For a larger SYNTHETIC system, run:"
echo "    python scripts/make_synthetic.py --chains 16 --len 12 --box 12.0"
