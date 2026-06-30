#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs +
# licensing, and NEVER bypasses credentials/registration. The committed tiny
# SYNTHETIC sample (data/sample/complex_sample.txt) already runs the demo
# offline, so this script only points at the real co-folding benchmarks for
# learners who want to go further. It downloads nothing automatically because
# those benchmarks carry their own licenses and are large.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.14 -- Protein-Ligand Co-Folding"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/complex_sample.txt) is SYNTHETIC and is"
echo "all the demo needs. The real co-folding benchmarks below are for further"
echo "study. This project's loader expects its own tiny token format (see"
echo "data/README.md); turning a real PDB complex into that format is left as an"
echo "exercise -- the point here is the diffusion+attention loop, not a parser."
echo
echo "Real protein-ligand complex benchmarks (study these):"
echo "  * PoseBusters  : 428 recent PDB complexes for pose validation"
echo "                   https://github.com/maabuu/posebusters  (MIT; check per-entry PDB terms)"
echo "  * PDBbind v2020: protein-ligand complexes + binding affinities"
echo "                   http://www.pdbbind.org.cn  (registration required; academic license)"
echo "  * Astex Diverse: 85 drug-like ligand complexes"
echo "                   https://www.ccdc.cam.ac.uk (verify current URL / terms)"
echo "  * CASF         : cross-docking scoring benchmarks"
echo "                   http://www.pdbbind.org.cn/casf.php"
echo
echo "For a larger SYNTHETIC complex (more tokens / steps), run:"
echo "  python scripts/make_synthetic.py --n-protein 24 --n-ligand 9 --steps 240"
echo
echo "NOTE: PDBbind/Astex require accepting a license or registering. This script"
echo "will NOT bypass that -- follow each site's instructions yourself."
