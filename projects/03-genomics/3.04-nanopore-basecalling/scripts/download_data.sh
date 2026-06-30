#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This project decodes posterior
# matrices (a neural network's OUTPUT). Real ONT data is RAW SIGNAL (.pod5 /
# .fast5), not posteriors -- turning signal into posteriors requires running a
# basecaller's network, which is the out-of-scope stage. So this script does not
# auto-download anything; it explains the sources and points to make_synthetic.py
# for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.4 -- Nanopore Basecalling (CTC greedy decode)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project consumes POSTERIOR MATRICES (a network's output),"
echo "not raw signal. The committed synthetic sample in data/sample/ is enough"
echo "to run the demo offline; no download is required."
echo
echo "To experiment with a LARGER synthetic batch (still offline):"
echo "    python scripts/make_synthetic.py --reads 4096"
echo
echo "To work with REAL nanopore data you need two things this script will not"
echo "do for you (they require external tools / accounts):"
echo "  1) Raw signal (.pod5 / .fast5). Public sources (respect each license):"
echo "       - ONT Open Dataset (PromethION human WGS) via SRA/ENA:"
echo "           https://www.ncbi.nlm.nih.gov/sra      https://www.ebi.ac.uk/ena"
echo "       - R9.4.1 / R10.4.1 benchmarks (awesome-nanopore index):"
echo "           https://github.com/GoekeLab/awesome-nanopore"
echo "       - GIAB ONT truth sets (NA12878 / HG002):"
echo "           https://www.nist.gov/programs-projects/genome-bottle"
echo "       - ENA Project PRJNA594038 (multi-species ONT):"
echo "           https://www.ebi.ac.uk/ena"
echo "  2) A basecaller to turn that signal into posteriors (the out-of-scope"
echo "     network stage): ONT Dorado -> https://github.com/nanoporetech/dorado"
echo "     Dorado can emit per-step probabilities; export those in this"
echo "     project's text format (see data/README.md) to feed this decoder."
echo
echo "[download_data] Nothing downloaded (by design). The demo runs on the"
echo "[download_data] committed synthetic sample."
