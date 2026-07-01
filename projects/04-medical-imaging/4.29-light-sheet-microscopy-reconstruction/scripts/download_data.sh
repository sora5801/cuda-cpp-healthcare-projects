#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.29 : Light-Sheet Microscopy Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.29 -- Light-Sheet Microscopy Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# This teaching project runs on a SYNTHETIC sample (see scripts/make_synthetic.py
# and data/README.md). Real light-sheet datasets are terabyte-scale volumes in
# formats (TIFF/HDF5/N5/Zarr) that this tiny text-based loader does not read, and
# several require a data-use agreement. So this script prints the public sources
# and defers to the synthetic generator rather than downloading TBs.
echo "  This project ships a SYNTHETIC sample; no download is required to run the demo."
echo "  Regenerate or enlarge it with:"
echo "    python scripts/make_synthetic.py                 # default 32x32 plane"
echo "    python scripts/make_synthetic.py --h 64 --w 64   # a larger synthetic plane"
echo
echo "  Real, publicly-documented LSFM data sources (study these; formats differ"
echo "  from this loader and some need registration -- respect every license):"
echo "    - OpenOrganelle (Janelia):        https://openorganelle.janelia.org/"
echo "    - EMBL LSFM public datasets:      https://www.embl.org/"
echo "    - BioImage Archive (EBI) LSFM:    https://www.ebi.ac.uk/biostudies/bioimages"
echo "    - Zebrafish SPIM atlas data:      from the Nature Methods SPIM papers"
echo
echo "  For a credentialed set, register at the source FIRST; this script never"
echo "  bypasses authentication (CLAUDE.md section 8)."
