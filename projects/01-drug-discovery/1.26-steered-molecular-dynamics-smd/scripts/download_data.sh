#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  SMD data pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.26 : Steered Molecular Dynamics (SMD)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. For THIS project there is nothing to fetch -- the
# reduced 1-D model is fully specified by the 14 numbers in
# data/sample/smd_config.txt. This script prints where to find REAL full-atom SMD
# material and defers to scripts/make_synthetic.py for larger offline ensembles.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.26 -- Steered Molecular Dynamics (SMD)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "There is NO file to download: the 1-D teaching model is fully defined by"
echo "data/sample/smd_config.txt (14 numbers; see data/README.md)."
echo
echo "For REAL full-atom SMD (pull a ligand out of a pocket in a true MD field):"
echo "  NAMD SMD tutorials : https://www.ks.uiuc.edu/Training/Tutorials/"
echo "                       (constant-velocity / constant-force SMD walkthroughs)"
echo "  GROMACS pull code  : https://github.com/gromacs/gromacs   (GPU pull-coord)"
echo "  OpenMM             : https://github.com/openmm/openmm     (CustomExternalForce)"
echo "  alchemlyb          : https://github.com/alchemistry/alchemlyb (Jarzynski post-proc)"
echo "  BindingDB          : https://www.bindingdb.org           (residence-time data)"
echo "  PDB                : https://www.rcsb.org                 (force-probe structures)"
echo
echo "Bigger SYNTHETIC ensemble for this project (no download, tighter Jarzynski):"
echo "  python scripts/make_synthetic.py --n-traj 65536"
