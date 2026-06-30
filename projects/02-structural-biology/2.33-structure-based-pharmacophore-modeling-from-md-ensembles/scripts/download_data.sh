#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real sources below need registration
# and/or a pharmacophore-typing pipeline to turn structures and MD frames into
# feature points; that is out of scope for this teaching version, so this script
# only PRINTS guidance and defers to scripts/make_synthetic.py for an offline,
# fully-synthetic stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.33 -- Structure-Based Pharmacophore Modeling from MD Ensembles"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a fully SYNTHETIC sample (data/sample/pharmacophore_sample.txt)."
echo "No real dataset is downloaded. Real sources (require registration and a"
echo "pharmacophore-typing pipeline -- see THEORY.md 'Where this sits in the real world'):"
echo
echo "  GPCRmd trajectory archive : https://gpcrmd.org        (GPCR MD ensembles)"
echo "  DUD-E actives/decoys      : https://dude.docking.org  (screening validation)"
echo "  RCSB PDB                  : https://www.rcsb.org      (target-class structures)"
echo "  ZINC drug-like library    : https://zinc20.docking.org (screening library)"
echo
echo "  Respect each source's license; none of that data is redistributed here."
echo "  The committed tiny sample is enough to run the demo offline."
echo "  For a larger SYNTHETIC screen, run:"
echo "    python scripts/make_synthetic.py --N 1000000"
