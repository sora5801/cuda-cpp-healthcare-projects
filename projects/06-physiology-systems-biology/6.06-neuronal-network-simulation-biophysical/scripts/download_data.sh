#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.6 : Neuronal Network Simulation (Biophysical)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs,
# and NEVER bypasses credentials/registration. The demo runs entirely on the
# committed synthetic sample; the real datasets below are OPTIONAL enrichment.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.6 -- Neuronal Network Simulation (Biophysical)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs on a TINY SYNTHETIC network committed under data/sample/"
echo "(network.txt). No download is required to build or run the demo."
echo
echo "OPTIONAL real-world data sources (study these; none is auto-downloaded):"
echo "  * NeuroMorpho.Org  -- 200,000+ 3D neuronal reconstructions (SWC morphology)."
echo "      https://neuromorpho.org   (free; cite the original reconstruction authors)"
echo "  * ModelDB / modeldb.science -- curated NEURON/GENESIS model files."
echo "      https://modeldb.science"
echo "  * Allen Brain Cell Atlas -- patch-seq morpho-electric data."
echo "      https://portal.brain-map.org"
echo "  * DANDI Archive -- neurophysiology datasets (NWB format)."
echo "      https://dandiarchive.org"
echo
echo "Turning an SWC morphology into this model's compartment chain is left as an"
echo "exercise (see README 'Exercises'): parse the SWC tree, collapse each branch into"
echo "compartments, and order them for the Hines solver."
echo
echo "For a larger SYNTHETIC ring (offline), run:"
echo "    python scripts/make_synthetic.py --ncell 256 --steps 8000"
