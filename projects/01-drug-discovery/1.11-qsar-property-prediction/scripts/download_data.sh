#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Point at the FULL QSAR datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.11 : QSAR / Property Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real QSAR benchmarks ship as
# SMILES + labels and must be featurized into atom/bond graphs with RDKit (a
# Python step outside this C++ demo), so this script does NOT auto-download:
# it prints where to get each dataset and how to convert it, and defers to
# scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.11 -- QSAR / Property Prediction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/molecules_sample.txt) is a tiny SYNTHETIC"
echo "molecule batch and is all the demo needs. To study REAL QSAR data:"
echo
echo "  MoleculeNet  (curated ML benchmarks; ESOL, FreeSolv, Lipophilicity, BBBP, ...)"
echo "    https://moleculenet.org   (CSV of SMILES + labels; no login)"
echo "  ChEMBL  (measured bioactivities for ~2.4M compounds)"
echo "    https://www.ebi.ac.uk/chembl/   (bulk download; no login)"
echo "  Therapeutics Data Commons (TDC)  (66 ready-made drug-discovery ML tasks)"
echo "    https://tdcommons.ai   (pip install PyTDC; programmatic access)"
echo "  PCBA  (128 PubChem BioAssays over ~440k compounds)"
echo "    https://moleculenet.org"
echo
echo "  These ship as SMILES + labels. To turn them into the CSR graph format"
echo "  this project reads (see data/README.md), featurize with RDKit, e.g.:"
echo "    pip install rdkit pandas"
echo "    # for each SMILES: atoms -> 6-dim feature rows, bonds -> edge list,"
echo "    # then emit 'num_mols num_nodes num_edges' + features + counts + edges."
echo
echo "  For a larger SYNTHETIC batch without any download, run:"
echo "    python scripts/make_synthetic.py"
echo
echo "  When wiring a real fetch, keep it idempotent: skip if the file exists"
echo "  with the right SHA256; print URL + size + checksum; never store secrets."
