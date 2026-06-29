#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the REAL training datasets (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This TEACHING project does not train a
# network -- its NNP weights are fixed surrogates in src/nnpmm.h -- so there is no
# dataset to fetch for the demo. This script only POINTS at the real QM/DFT
# reference datasets you would use to train a genuine NNP (MACE/NequIP), and
# defers to make_synthetic.py for the offline run config.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.35 -- QMMM/ML Potential Hybrid MD"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This reduced-scope teaching demo needs NO download: the NNP weights are"
echo "fixed synthetic surrogates (src/nnpmm.h) and the only input is the tiny"
echo "committed ensemble config (data/sample/ensemble_params.txt)."
echo
echo "To train a REAL ML potential for hybrid NNP/MM MD, use these QM reference"
echo "datasets (each has its own license + access terms -- respect them):"
echo "  * Transition1x : ~10M DFT calculations along reaction paths"
echo "      https://zenodo.org/record/5781475"
echo "  * SPICE        : drug-like + biomolecular DFT energies/forces"
echo "      https://github.com/openmm/spice-dataset"
echo "  * ANI-1ccx     : CCSD(T)*-quality energies (reactive extensions: verify URL)"
echo
echo "For a larger SYNTHETIC ensemble to stress the GPU path, run:"
echo "  python scripts/make_synthetic.py --M 65536"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for credentialed sets, print registration instructions ONLY"
