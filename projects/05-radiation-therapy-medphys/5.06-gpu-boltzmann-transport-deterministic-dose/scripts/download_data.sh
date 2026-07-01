#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point to the FULL datasets (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real deterministic-transport benchmarks
# below are either license-restricted or need registration, so this script only
# PRINTS where to get them; the committed synthetic slab (data/sample/) plus
# scripts/make_synthetic.py let the demo run fully offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC slab (data/sample/slab_problem.txt), which is"
echo "all the demo needs. The real-world references are listed below for study;"
echo "each must be obtained under its own terms -- we do not redistribute them."
echo
echo "  1) AAPM TG-105 report (deterministic/MC dose calc guidance)"
echo "       https://www.aapm.org/pubs/reports/  (search 'TG-105')"
echo "  2) IROC Houston heterogeneous phantom program (credentialing)"
echo "       https://www.mdanderson.org/  (search 'IROC Houston phantom')"
echo "  3) IAEA photon cross-section / nuclear data services"
echo "       https://www-nds.iaea.org/"
echo "  4) Acuros XB validation: Varian/Eclipse white papers (vendor-published)"
echo
echo "To make a LARGER synthetic problem (more cells / higher S_N order), run:"
echo "    python scripts/make_synthetic.py --ncell 400 --nord 16"
echo
echo "When wiring a real cross-section set, follow the idempotent pattern:"
echo "    1) skip the download if the file already exists with the right SHA256"
echo "    2) print source URL + expected size + checksum before fetching"
echo "    3) for credentialed sets, print registration instructions ONLY"
