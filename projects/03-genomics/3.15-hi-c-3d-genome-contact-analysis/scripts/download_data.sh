#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch a FULL Hi-C dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.15 : Hi-C / 3D Genome Contact Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# format notes, and NEVER bypasses credentials/registration. Public Hi-C maps
# are large binary .hic / .mcool files that need a converter (cooler / hic2cool),
# so this script PRINTS the exact, vetted steps rather than blindly downloading
# multi-GB files. The committed synthetic sample already runs the demo offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.15 -- Hi-C / 3D Genome Contact Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny sample in data/sample/hic_sample.txt is enough to run the"
echo "demo offline. To work with REAL Hi-C contact maps, use a public source:"
echo
echo "  * 4DN Data Portal   https://data.4dnucleome.org/   (.mcool, many cell types)"
echo "  * ENCODE            https://www.encodeproject.org/  (cell-line Hi-C)"
echo "  * GEO GSE63525      https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525"
echo "                      (Rao et al. 2014, classic high-resolution maps)"
echo
echo "Real maps ship as binary .hic (Juicer) or .mcool (cooler). Convert one bin"
echo "resolution to the plain 'i j count' text this project reads, e.g. with cooler:"
echo
echo "    pip install cooler"
echo "    # dump one chromosome of a .mcool at 100 kb as COO upper triangle:"
echo "    cooler dump -t pixels --join -r chr1 my_map.mcool::/resolutions/100000 \\"
echo "      | awk '{print \$2/100000, \$5/100000, \$7}' > data/full/chr1_100kb.txt"
echo
echo "  (Add an 'n nnz' header line matching this project's loader format; see"
echo "   data/README.md. Bin indices must be 0-based and upper-triangular i<=j.)"
echo
echo "For a larger SYNTHETIC problem instead (no download), run:"
echo "    python scripts/make_synthetic.py --out data/full/synthetic_big.txt"
echo
echo "[download_data] Nothing downloaded (by design). Follow the steps above for real data."
