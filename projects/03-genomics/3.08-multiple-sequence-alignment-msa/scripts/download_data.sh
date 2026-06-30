#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.8 : Multiple Sequence Alignment (MSA)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. The committed
# tiny SYNTHETIC sample (data/sample/) already runs the demo offline; this
# script points at real MSA benchmarks for going further.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.8 -- Multiple Sequence Alignment (MSA)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/sequences_sample.fasta) is SYNTHETIC"
echo "and sufficient to run the demo. No download is required for that."
echo
echo "Real MSA benchmarks you can study (verify each license before use):"
echo "  * BAliBASE  -- curated reference alignments"
echo "      https://www.lbgi.fr/balibase/"
echo "  * HomFam    -- large homologous-family benchmark (used by Clustal Omega)"
echo "      (search 'HomFam benchmark'; distributed with Clustal Omega papers)"
echo "  * Pfam seed alignments -- protein family seed MSAs"
echo "      https://www.ebi.ac.uk/interpro/download/  (Pfam section)"
echo
echo "These provide multi-FASTA inputs the loader reads directly (DNA mode here"
echo "expects A/C/G/T only -- protein sets need the substitution-matrix upgrade"
echo "described in THEORY.md before they will load)."
echo
echo "For a larger SYNTHETIC family (no download), run e.g.:"
echo "    python scripts/make_synthetic.py --n 32 --sub 0.12 --indel 0.08"
echo
echo "Idempotent-download pattern to follow when wiring a real set:"
echo "    1) skip the fetch if the file already exists with the right SHA256"
echo "    2) print source URL + expected size + SHA256 before downloading"
echo "    3) for credentialed sets, print registration instructions ONLY"
