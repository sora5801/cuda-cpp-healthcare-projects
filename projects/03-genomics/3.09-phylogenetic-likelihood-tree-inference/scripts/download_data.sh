#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.9 -- Phylogenetic Likelihood / Tree Inference
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, points at the source URLs,
# and NEVER bypasses credentials/registration. The committed tiny SYNTHETIC
# sample already runs the demo offline; this script only points at the real
# curated databases and defers to make_synthetic.py for a larger offline set.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.9 -- Phylogenetic Likelihood / Tree Inference"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a tiny SYNTHETIC sample (data/sample/phylo_sample.txt)"
echo "that is sufficient to build, run, and verify the demo offline."
echo
echo "Real curated phylogenetic alignments / trees (study these next):"
echo "  * TreeBASE       https://www.treebase.org/        (alignments + trees)"
echo "  * SILVA rRNA     https://www.arb-silva.de/        (large rRNA alignment)"
echo "  * NCBI CDD       https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml"
echo "  * Open Tree      https://opentreeoflife.github.io/ (aggregated phylogenies)"
echo
echo "These arrive as FASTA/PHYLIP/NEXUS alignments with Newick trees. Converting"
echo "one to this project's compact text format (encode bases A/C/G/T->0..3, write"
echo "a POST-ORDER node list; see data/README.md) is left as a README exercise."
echo "Respect each source's license; none is redistributed here."
echo
echo "For a larger OFFLINE synthetic problem instead, run:"
echo "    python scripts/make_synthetic.py --n-sites 50000 --seed 7"
