#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real hydration-free-energy data pointers (POSIX)
# ---------------------------------------------------------------------------
# Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs and
# NEVER bypasses credentials/registration. This project's demo needs NOTHING to
# download -- the committed sample fully specifies a reproducible calculation, and
# the model bath is generated deterministically. This script only points you at
# the REAL experimental benchmarks you would validate a production engine against.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 1.32 -- Alchemical Hydration Free Energy (dG_solv)"
echo
echo "Nothing to fetch: data/sample/alchemy_config.txt fully specifies the"
echo "calculation, and the solvent bath is built deterministically in code."
echo
echo "REAL experimental hydration-free-energy benchmarks (for validating a"
echo "production FEP engine -- NOT consumed by this teaching demo):"
echo "  FreeSolv      : https://github.com/MobleyLab/FreeSolv    (643 dG_hyd; permissive)"
echo "  MNSol         : https://comp.chem.umn.edu/mnsol/         (license acceptance required)"
echo "  SAMPL         : https://github.com/samplchallenges/SAMPL (blind challenges)"
echo "  NIST ThermoML : https://trc.nist.gov                     (curated thermochemistry)"
echo
echo "MNSol requires accepting a license on its website; this script does NOT"
echo "bypass that -- download it manually if you accept the terms."
echo
echo "Bigger / finer SYNTHETIC problem (no download):"
echo "  python scripts/make_synthetic.py --n-windows 21 --n-walkers 256"
echo
echo "Target data dir: $PROJECT_ROOT/data"
