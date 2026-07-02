#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.10 : Systems-Biology ODE/SDE Network Solver
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real "data" here is the space of curated
# systems-biology MODELS (SBML files), not a single tabular download; this script
# points you to the model repositories and defers to scripts/make_synthetic.py
# for the offline teaching stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.10 -- Systems-Biology ODE/SDE Network Solver"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project integrates a small GRN (the repressilator) as an ENSEMBLE."
echo "The committed synthetic sample (data/sample/ensemble_params.txt) already"
echo "runs the demo offline. Real curated models live in these open repositories:"
echo
echo "  * BioModels (EMBL-EBI): 1000+ curated SBML models"
echo "      https://www.ebi.ac.uk/biomodels   (the repressilator is BIOMD0000000012)"
echo "  * Reactome pathways:    https://reactome.org"
echo "  * BioGRID network:      https://thebiogrid.org"
echo "  * VCell curated models: https://vcell.org"
echo
echo "These are SBML/XML files. Turning an arbitrary SBML model into ODE RHS code"
echo "is a parsing + code-generation task (see libRoadRunner / Tellurium in the"
echo "README 'Prior art'); it is intentionally OUT OF SCOPE for this teaching demo,"
echo "which hard-codes the repressilator RHS in src/grn.h so the focus stays on the"
echo "GPU batch-ODE pattern. No credentials are required for any link above; respect"
echo "each site's license before redistributing anything."
echo
echo "For a bigger SYNTHETIC sweep (more ensemble members), run:"
echo "  python scripts/make_synthetic.py --na 64 --nn 64"
