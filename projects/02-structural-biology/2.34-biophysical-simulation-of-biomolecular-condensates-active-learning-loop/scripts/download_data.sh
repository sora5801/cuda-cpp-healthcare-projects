#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch real condensate references (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 2.34 : Biophysical Simulation of Biomolecular Condensates
#                (Active Learning Loop)  --  reduced-scope teaching version
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs,
# and NEVER bypasses credentials/registration. This project's DEMO needs no
# download at all -- its input is a synthetic configuration file written by
# scripts/make_synthetic.py. This script instead points at the public datasets a
# learner would use to go from this teaching toy toward a real active-learning
# loop, and explains why none of them is auto-fetched here.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.34 : Biomolecular Condensates (Active Learning Loop)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This demo needs NO download: the input is a synthetic ensemble config"
echo "(data/sample/condensate_ensemble.txt), regenerated any time with:"
echo "    python scripts/make_synthetic.py"
echo "    python scripts/make_synthetic.py --n-members 200   # a larger sweep"
echo
echo "Public references for a REAL condensate active-learning loop (study these;"
echo "they are not auto-downloaded -- formats vary and some require agreement to"
echo "terms of use, which this script will not bypass):"
echo "  * PhaSePro  -- curated LLPS proteins/regions : https://phasepro.elte.hu"
echo "  * DisProt   -- intrinsically disordered regions : https://disprot.org"
echo "  * RCSB PDB  -- structures of FUS/TDP-43/hnRNPA1 LC domains : https://www.rcsb.org"
echo "  * CALVADOS  -- residue-level IDP CG model + params : https://github.com/KULL-Centre/CALVADOS"
echo
echo "Idempotent pattern to follow when wiring any real fetch:"
echo "  1) skip the download if the file already exists with the right SHA256"
echo "  2) print source URL + expected size + checksum before downloading"
echo "  3) for credentialed/registered sources, print instructions ONLY"
