#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs and
# NEVER bypasses credentials/registration. Real airway geometries come from
# patient CT archives that require an account and a data-use agreement, so this
# script only PRINTS how to obtain them and points at make_synthetic.py for an
# offline stand-in. The committed tiny sample already runs the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.5 -- Respiratory / Lung Airflow & Particle Deposition"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a self-contained SYNTHETIC parameter file"
echo "(data/sample/lung_params.txt). No download is required to run the demo."
echo
echo "To drive the model from a REAL patient airway geometry, obtain a lung CT"
echo "volume, segment the airway tree, and fit per-generation radii/lengths."
echo "Public sources (each needs registration / a data-use agreement -- respect it):"
echo "  * LIDC-IDRI lung CT (1010 cases), TCIA:"
echo "      https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
echo "  * COPDGene lung CT (10000 subjects): https://www.copdgene.org"
echo "  * SPIROMICS bronchial CT:            https://www.spiromics.org"
echo "  * PhysioNet respiratory waveforms:   https://physionet.org"
echo "  Airway segmentation tooling: 3D Slicer + SlicerMorph"
echo "      https://github.com/SlicerMorph/SlicerMorph"
echo
echo "For a larger SYNTHETIC experiment (no download), regenerate the sample:"
echo "    python scripts/make_synthetic.py --d_p 1.0 --n 1000000"
echo
echo "[download_data] Done (informational only; nothing was downloaded)."
