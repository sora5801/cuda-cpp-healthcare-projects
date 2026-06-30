#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
#               REDUCED-SCOPE TEACHING VERSION.
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This teaching project runs on a tiny
# SYNTHETIC sample (data/sample/attention_sample.txt); no real dataset is needed
# to build or demo it. This script only prints where the real data lives and
# defers to make_synthetic.py for a bigger offline problem.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.1 -- Protein Structure Prediction Inference (AlphaFold-class)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project needs NO download: the committed synthetic sample in"
echo "  data/sample/attention_sample.txt"
echo "is sufficient to build, run, and verify the demo offline."
echo
echo "Where the real-world data lives (for further study, NOT required here):"
echo "  * AlphaFold Protein Structure DB (200M+ predicted structures):"
echo "      https://alphafold.ebi.ac.uk/"
echo "  * RCSB PDB (227k+ experimental structures):  https://www.rcsb.org"
echo "  * UniProt / UniRef90 (MSA sequence databases): https://www.uniprot.org"
echo "  * CAMEO / CASP15 prediction benchmarks:        https://www.cameo3d.org"
echo
echo "Note: a real AlphaFold/ESMFold run also needs multi-gigabyte trained model"
echo "WEIGHTS and (for AF2) MSA databases -- see those projects' repos. This"
echo "teaching version uses random synthetic Q/K/V instead (no weights)."
echo
echo "For a larger SYNTHETIC attention problem (more residues), run:"
echo "    python scripts/make_synthetic.py --L 64"
