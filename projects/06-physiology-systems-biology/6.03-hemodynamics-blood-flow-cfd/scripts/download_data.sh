#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real-dataset pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. This project GENERATES its own flow from the
# parameters in data/sample/channel_params.txt, so there is nothing to download
# for the demo. The real, research-grade inputs are credential-gated; this
# script prints instructions and links ONLY and defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.3 -- Hemodynamics / Blood-Flow CFD"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Nothing to download: the solver makes its own channel flow from the"
echo "synthetic parameters in data/sample/channel_params.txt."
echo
echo "For REAL patient-specific hemodynamics (image -> mesh -> CFD), the catalog"
echo "datasets are credential-gated or license-restricted; obtain them yourself"
echo "under their terms (this script will not bypass logins):"
echo "  Vascular Model Repository (geometries) : http://www.vascularmodel.com"
echo "  PhysioNet MIMIC-III waveforms          : https://physionet.org/content/mimiciii/1.4/"
echo "  Zenodo Cardiac Mechanics Emulation     : https://zenodo.org/records/7075055"
echo "  UK Biobank aortic 4D-flow MRI          : https://www.ukbiobank.ac.uk"
echo
echo "Full image-to-simulation pipeline (out of scope for this teaching project):"
echo "  SimVascular / svFSI : https://github.com/SimVascular/svFSI"
echo "  OpenFOAM            : https://github.com/OpenFOAM/OpenFOAM-dev"
echo
echo "Bigger / non-Newtonian SYNTHETIC problem:"
echo "  python scripts/make_synthetic.py --nx 128 --ny 65 --steps 20000"
echo "  python scripts/make_synthetic.py --nu-inf 0.03   # blood shear thinning"
