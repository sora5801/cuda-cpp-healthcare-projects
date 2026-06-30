#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real SAXS curves and PDB models are fetched
# by the user from the public banks below; this script prints guidance and defers
# to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.24 -- SAXS / SANS Data-Driven Structure Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny SYNTHETIC sample (data/sample/saxs_sample.txt) already"
echo "lets the demo run offline. No real data is required to study this project."
echo
echo "To work with REAL small-angle scattering data, fetch it yourself from:"
echo "  * SASBDB   -- curated SAXS/SANS curves + models : https://www.sasbdb.org"
echo "               (a .dat file has columns 'q  I  sigma', the same layout as"
echo "                our sample's curve section)"
echo "  * RCSB PDB -- atomic models to forward-model     : https://www.rcsb.org"
echo "  * BIOISIS  -- SAXS benchmark database (verify the current URL)"
echo
echo "Converting a real .pdb to our text format: read ATOM records, map each"
echo "element to an electron count (or a proper q-dependent form factor; see"
echo "THEORY.md), then append a SASBDB .dat curve as the 'q I_exp sigma' block."
echo
echo "For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --atoms 500 --nq 60 --out data/sample/big.txt"
echo
echo "Idempotent real-fetch pattern (when you wire one up):"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
