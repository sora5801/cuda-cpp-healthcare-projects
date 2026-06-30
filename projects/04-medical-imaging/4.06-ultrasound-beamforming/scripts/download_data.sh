#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL ultrasound RF data (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Defers to scripts/make_synthetic.py for an
# offline, reproducible stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.6 -- Ultrasound Beamforming"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Options for real / standard RF beamforming data:"
echo "  * PICMUS (Plane-Wave Imaging Challenge in Medical Ultrasound) --"
echo "    canonical RF datasets (point targets, cysts, in-vivo) for beamformer"
echo "    evaluation: https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/"
echo "    (registration may be required; this script does NOT bypass it)."
echo "  * Field II (https://field-ii.dk/) -- CPU simulator that GENERATES"
echo "    realistic RF data for arbitrary phantoms; export to the data/README"
echo "    format, then beamform with this project's GPU kernel."
echo "  * k-Wave / k-Wave-Fluid-CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA)"
echo "    -- full-wave acoustic propagation (more physical than our point model)."
echo "  * MUST (https://www.biomecardio.com/MUST/) -- reference DAS + sample RF."
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py"
echo "  python scripts/make_synthetic.py --elements 128 --samples 512 --nx 192 --nz 192 --extra"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for credentialed sets, print registration instructions ONLY"
