#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Guidance for fetching REAL backbones (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 2.10 : Protein Design / Inverse Folding Inference
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The committed SYNTHETIC sample in
# data/sample/ is enough to run and verify the demo; this script explains where
# REAL protein backbones come from and how you would wire one in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.10 -- Protein Design / Inverse Folding Inference"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC backbone (data/sample/backbone_sample.txt)."
echo "No download is required to run the demo. To experiment with REAL structures:"
echo
echo "  PDB  (https://www.rcsb.org)   -- experimental protein structures."
echo "       Fetch one entry's coordinates, e.g. small protein 1UBQ (ubiquitin):"
echo "         curl -L -o \"\$DATA_DIR/1UBQ.pdb\" https://files.rcsb.org/download/1UBQ.pdb"
echo "       Then parse its CA atom records into '<x> <y> <z> <native_letter>'"
echo "       lines (one residue per line) -- see data/README.md. PDB data is"
echo "       freely redistributable for most entries (verify per entry)."
echo
echo "  CATH (https://www.cathdb.info)              -- 500k+ classified domains."
echo "  ProteinGym (https://github.com/OATML-Markslab/ProteinGym) -- fitness sets."
echo "  CAMEO (https://www.cameo3d.org)             -- fresh validation backbones."
echo
echo "  For a larger SYNTHETIC backbone instead, run:"
echo "    python scripts/make_synthetic.py --shells 40 --per 12"
echo
echo "  Idempotent-download pattern when wiring a real fetch:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
