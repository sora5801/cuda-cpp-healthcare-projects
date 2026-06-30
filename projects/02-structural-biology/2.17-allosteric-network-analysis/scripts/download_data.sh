#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch a REAL trajectory dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.17 -- Allosteric Network Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real allosteric-trajectory archives are
# large and account-gated, so this script PRINTS INSTRUCTIONS only and defers to
# scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.17 -- Allosteric Network Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny sample (data/sample/trajectory.txt) is SYNTHETIC and is"
echo "all the demo needs. To study a REAL allosteric trajectory, obtain one of:"
echo
echo "  * GPCRmd allosteric trajectory archive  https://gpcrmd.org"
echo "      Browse to a GPCR system, download the trajectory (.xtc/.dcd) +"
echo "      topology (.pdb/.psf). Free, but registration is required -- this"
echo "      script does NOT log in for you."
echo "  * MDAnalysis test trajectories          https://github.com/MDAnalysis/mdanalysis"
echo "  * ProDy benchmark structures/ensembles  https://github.com/prody/ProDy"
echo "  * Allosteric Database (ASD)             http://mdl.shsmu.edu.cn/ASD/"
echo
echo "CONVERT a real trajectory into this project's plain-text format (the loader"
echo "expects '# SITE_ALLO i', '# SITE_ACTIVE j', then 'N T', then T*N lines of"
echo "'x y z' Calpha coordinates, frame-major) using MDAnalysis, e.g.:"
echo
echo "    import MDAnalysis as mda"
echo "    u = mda.Universe('topology.pdb', 'traj.xtc')"
echo "    ca = u.select_atoms('name CA')"
echo "    # write N T header, then loop frames writing ca.positions ..."
echo
echo "For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --residues 200 --frames 1000"
