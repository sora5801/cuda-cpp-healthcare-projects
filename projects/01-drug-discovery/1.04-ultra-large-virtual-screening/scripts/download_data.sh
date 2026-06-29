#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.4 : Ultra-Large Virtual Screening
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + the
# recipe, and NEVER bypasses credentials/registration. Real screening libraries
# are huge and license-bound, so this script does NOT auto-download a multi-
# billion-compound set; it prints the RDKit recipe to turn a SMILES list into
# this project's descriptor format, and defers to make_synthetic.py for an
# offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.4 -- Ultra-Large Virtual Screening"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real virtual-screening libraries (descriptors + features):"
echo "  Enamine REAL  >6B make-on-demand : https://enamine.net/compound-collections/real-compounds"
echo "  ZINC20        ~2B purchasable     : https://zinc20.docking.org"
echo "  ChEMBL        bioactivity ref     : https://www.ebi.ac.uk/chembl/"
echo "  ExCAPE-DB     chemogenomics       : https://solr.ideaconsult.net/search/excape/"
echo
echo "These sets are huge and license-bound -- download a SMILES subset from the"
echo "links above (respect each license), then compute this project's columns with"
echo "RDKit (mw logp_x100 hbd hba rotb psa feat_hex):"
echo
echo "  from rdkit import Chem"
echo "  from rdkit.Chem import Descriptors, Lipinski"
echo "  m = Chem.MolFromSmiles(smiles)"
echo "  mw   = round(Descriptors.MolWt(m))"
echo "  logp = round(Descriptors.MolLogP(m) * 100)"
echo "  hbd  = Lipinski.NumHDonors(m);  hba = Lipinski.NumHAcceptors(m)"
echo "  rotb = Descriptors.NumRotatableBonds(m)"
echo "  psa  = round(Descriptors.TPSA(m))"
echo "  # feat = a 32-bit pharmacophore/Morgan bitmask folded to 32 bits"
echo
echo "The committed tiny sample in data/sample/ is enough to run the demo."
echo "For a larger SYNTHETIC problem (no download, fully offline), run:"
echo "    python scripts/make_synthetic.py --n 1000000"
