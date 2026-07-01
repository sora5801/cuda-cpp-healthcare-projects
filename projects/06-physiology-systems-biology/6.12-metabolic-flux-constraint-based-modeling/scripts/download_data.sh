#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to REAL metabolic models (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 6.12 : Metabolic Flux / Constraint-Based Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials. The committed sample is a tiny SYNTHETIC toy model
# (data/sample/toy_core_model.txt) that runs the demo offline; the genome-scale
# models below are public but big and in SBML/JSON, which our simple text loader
# does not parse -- so this script only prints where to get them and how a real
# workflow (COBRApy) would consume them.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.12 -- Metabolic Flux / Constraint-Based Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC toy model (data/sample/toy_core_model.txt)."
echo "It runs the demo with zero downloads. To regenerate or resize it:"
echo "    python scripts/make_synthetic.py"
echo
echo "Real genome-scale metabolic models (public, but SBML/JSON -- not our text"
echo "format; use COBRApy to read them):"
echo "  * BiGG Models (curated GEMs):     http://bigg.ucsd.edu/models"
echo "      e.g. E. coli core (95 rxns):  http://bigg.ucsd.edu/models/e_coli_core"
echo "  * Recon3D (human, ~10600 rxns):   https://github.com/SBRG/Recon3D"
echo "  * Virtual Metabolic Human portal: https://vmh.life"
echo
echo "Typical real workflow (outside this teaching repo):"
echo "    pip install cobra"
echo "    python -c \"import cobra; m=cobra.io.load_model('e_coli_core'); print(m.optimize())\""
echo
echo "See THEORY.md 'Where this sits in the real world' for how production FBA"
echo "differs (sparse interior-point / revised simplex over 1000s of reactions)."
