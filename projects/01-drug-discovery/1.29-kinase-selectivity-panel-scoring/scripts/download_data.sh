#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.29 : Kinase Selectivity Panel Scoring
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. The real
# selectivity datasets require registration or limit redistribution, so this
# script only PRINTS instructions + links and defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.29 -- Kinase Selectivity Panel Scoring"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC committed sample (data/sample/kinase_panel_sample.txt),"
echo "which is all the demo needs. The real selectivity datasets are external:"
echo
echo "  * KLIFS (kinase-ligand interaction fingerprints) -> https://klifs.net"
echo "      Use the web API or the 'kissim' package to build per-kinase IFPs."
echo "      Free for academic use; check site terms before any redistribution."
echo "  * KINOMEscan / Kd selectivity (Karaman et al. 2008, DiscoverX/Eurofins)"
echo "      Published supplements or commercial panels; provider-specific license."
echo "  * ChEMBL kinase bioactivity -> https://www.ebi.ac.uk/chembl/  (CC BY-SA 3.0)"
echo "  * Drug-Target Commons (DTC) -> https://drugtargetcommons.fimm.fi  (CC BY 4.0)"
echo
echo "These require registration and/or forbid wholesale redistribution, so they"
echo "are NOT downloaded automatically and NOT committed (CLAUDE.md section 8)."
echo
echo "For a larger SYNTHETIC panel, edit the PANEL list in scripts/make_synthetic.py"
echo "and re-run:  python scripts/make_synthetic.py"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for credentialed sets, print registration instructions ONLY"
