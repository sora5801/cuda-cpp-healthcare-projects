#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.13 : MSA Generation Acceleration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + sizes,
# and NEVER bypasses credentials/registration. The real MSA databases are huge
# (UniRef90 ~210 GB) and are not redistributed here; this script prints where to
# get them and defers to scripts/make_synthetic.py for an offline stand-in. The
# committed tiny sample already lets the demo run with zero downloads.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.13 -- MSA Generation Acceleration"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/profile_db_sample.txt) is SYNTHETIC and"
echo "is all the demo needs. The real MSA databases are not redistributed here:"
echo
echo "  UniRef90    ~210 GB   https://www.uniprot.org/help/uniref"
echo "  UniClust30            https://uniclust.mmseqs.com"
echo "  MGnify                https://www.ebi.ac.uk/metagenomics/"
echo "  BFD                   https://bfd.mmseqs.com"
echo
echo "To build a real profile HMM and search, use HMMER/HHblits or MMseqs2:"
echo "  https://github.com/soedinglab/MMseqs2   (GPU-capable search/clustering)"
echo "  https://github.com/sokrypton/ColabFold  (GPU MSA server for AlphaFold2)"
echo
echo "For a larger SYNTHETIC problem that runs with this project as-is:"
echo "    python scripts/make_synthetic.py --n 4096 --seed 7"
echo
echo "[download_data] Nothing to download; exiting cleanly."
