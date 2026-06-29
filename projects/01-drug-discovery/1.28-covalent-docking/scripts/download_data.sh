#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.28 : Covalent Docking
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URL +
# expected size, and NEVER bypasses credentials/registration. This project's demo
# runs entirely on the committed SYNTHETIC sample (data/sample/), so no download
# is required to build, run, or learn. This script only points at the REAL
# covalent-docking resources a learner could study next.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.28 -- Covalent Docking"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC sample (data/sample/covalent_sample.txt)"
echo "that is sufficient for the demo. No external download is needed."
echo
echo "Real covalent-docking resources to study (manual, license-respecting):"
echo "  * PDB covalent complexes ...... https://www.rcsb.org   (search 'covalent ligand')"
echo "      e.g. KRAS G12C + sotorasib (6OIM), BTK + ibrutinib (5P9J)."
echo "  * ChEMBL covalent inhibitors .. https://www.ebi.ac.uk/chembl/"
echo "  * BindingDB covalent entries .. https://www.bindingdb.org"
echo "  * CovDocker benchmark (2025) .. arXiv:2506.21085 (verify the released URL)"
echo
echo "To regenerate the synthetic sample (deterministic):"
echo "    python scripts/make_synthetic.py"
echo
echo "NOTE: PDB/ChEMBL/BindingDB each carry their own license terms -- read and"
echo "respect them. Several covalent benchmarks require registration; this"
echo "script prints links ONLY and never bypasses any access control."
