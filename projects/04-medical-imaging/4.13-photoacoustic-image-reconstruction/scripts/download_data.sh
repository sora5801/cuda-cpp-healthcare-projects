#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.13 : Photoacoustic Image Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# NOTE: this project needs NO download to run -- the committed synthetic sample
# (data/sample/pa_sample.txt) is generated locally by make_synthetic.py. The
# pointers below are for learners who want real photoacoustic data.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.13 -- Photoacoustic Image Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo needs NO download: data/sample/pa_sample.txt is synthetic and"
echo "regenerable with:  python scripts/make_synthetic.py"
echo "For a bigger synthetic problem:  python scripts/make_synthetic.py --sensors 256 --samples 1024 --img 256"
echo
echo "To study REAL photoacoustic data, see (verify URLs; respect each license):"
echo "  * k-Wave toolbox + example datasets .......... http://www.k-wave.org/"
echo "  * k-Wave CUDA fluid solver ................... https://github.com/klepo/k-Wave-Fluid-CUDA"
echo "  * PyTomography (GPU tomography incl. PA) ..... https://github.com/lukepolson/pytomography"
echo "  * In-vivo PA datasets in open-access Nature Communications papers"
echo "  * PASCAA / IPASC challenge data .............. photoacoustics.org (verify URL)"
echo
echo "Credentialed/registration-gated sets: this script will NOT bypass a login."
echo "Register at the source, then place files under data/ yourself."
