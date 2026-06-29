#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.19 -- Network / Polypharmacology Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real polypharmacology knowledge
# graphs (STRING, DrugBank, STITCH, DrugComb) are large and several require
# account registration or non-redistributable licenses, so this script does NOT
# auto-download them -- it prints exactly where to get each one. The committed
# SYNTHETIC sample (data/sample/) is sufficient to run the demo offline, and
# scripts/make_synthetic.py generates larger synthetic problems on demand.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.19 -- Network / Polypharmacology Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This demo runs on a SYNTHETIC knowledge graph (data/sample/), so no"
echo "download is required. To experiment with real polypharmacology data,"
echo "obtain the sources below yourself and respect each license:"
echo
echo "  STRING PPI network   https://string-db.org/cgi/download        (CC BY 4.0; protein-protein edges + confidence scores)"
echo "  DrugBank             https://go.drugbank.com/releases/latest   (requires free academic registration; drugs + targets)"
echo "  STITCH               http://stitch.embl.de/cgi/download.pl     (drug-protein interactions; check per-use terms)"
echo "  DrugComb             https://drugcomb.fimm.fi/                  (drug-combination synergy; cite the publication)"
echo
echo "Workflow to turn a real edge list into TransE embeddings (see THEORY.md):"
echo "  1) parse edges into (head, relation, tail) triples with integer entity IDs"
echo "  2) train TransE/RotatE embeddings with PyTorch Geometric or DGL on a GPU"
echo "  3) export the query head + relation + all tail embeddings into this"
echo "     project's text layout (data/README.md) and pass it as argv[1]"
echo
echo "For a larger SYNTHETIC problem right now, run:"
echo "  python scripts/make_synthetic.py --n 100000 --dim 64"
