#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.21 : Polarizable / AMOEBA Force Field MD
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The committed tiny SYNTHETIC sample is enough
# to run the demo offline; this script only points you at the real AMOEBA
# parameter sets and reference data.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.21 -- Polarizable / AMOEBA Force Field MD"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a tiny SYNTHETIC ensemble (data/sample/amoeba_ensemble.txt)"
echo "that fully exercises the induced-dipole CG solver offline. No download is"
echo "required to build, run, or verify the demo."
echo
echo "To study REAL AMOEBA force-field data and validation targets:"
echo "  * AMOEBA / AMOEBA+ parameter files (Tinker .prm/.key):"
echo "      https://github.com/TinkerTools/tinker        (params/ directory)"
echo "      https://github.com/TinkerTools/poltype2      (AMOEBA+ parameterization)"
echo "  * NIST thermophysical properties (water dielectric / dipole benchmarks):"
echo "      https://webbook.nist.gov"
echo "  * BindingDB experimental affinities (for FEP validation):"
echo "      https://www.bindingdb.org"
echo
echo "These are large and/or license-restricted, so we do NOT redistribute them."
echo "Respect each source's license. For a larger SYNTHETIC ensemble instead, run:"
echo "    python scripts/make_synthetic.py --members 256"
