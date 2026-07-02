#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.19 : Defibrillation & High-Voltage Shock Simulation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL(s), and
# NEVER bypasses credentials/registration. This project is a REDUCED-SCOPE
# teaching model (a 1-D FitzHugh-Nagumo cable) that runs entirely on the tiny
# SYNTHETIC sample in data/sample/, so there is nothing to download for the demo.
# This script points at the real research data + tools and can regenerate a
# synthetic problem.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.19 -- Defibrillation & High-Voltage Shock Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project needs NO download: the committed synthetic sample"
echo "  data/sample/defib_sweep.txt  is sufficient to build and run the demo."
echo
echo "Real defibrillation research data + solvers (respect each license/registration):"
echo "  - PhysioNet fibrillation/defibrillation recordings : https://physionet.org"
echo "  - openCARP defibrillation tutorial cases           : https://opencarp.org"
echo "  - Cardioid (LLNL) bidomain shock examples          : https://github.com/llnl/cardioid"
echo "  - Chaste (bidomain + electrode BCs)                : https://github.com/Chaste/Chaste"
echo "  - MonoAlg3D_C (GPU bidomain-capable)               : https://github.com/rsachetto/MonoAlg3D_C"
echo
echo "  PhysioNet requires accepting a data-use agreement; patient-specific ICD"
echo "  datasets require institutional/IRB access. This script does NOT bypass"
echo "  either -- register at the source and download manually."
echo
echo "To (re)generate the synthetic sample or a variant, run:"
echo "    python scripts/make_synthetic.py"
echo "    python scripts/make_synthetic.py --biphasic 1 --shock-len 20"
