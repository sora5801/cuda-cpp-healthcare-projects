#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real DTI-dataset pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 7.2 : Drug-Target Interaction Prediction (GNN)
#
# Nothing to auto-download: the demo runs fully offline on the tiny synthetic
# sample in data/sample/. This script prints where the REAL datasets live and how
# to convert them into this project's loader format (data/README.md). It NEVER
# bypasses registration/credentials (CLAUDE.md sec 8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 7.2 -- Drug-Target Interaction Prediction (GNN)"
echo
echo "The demo runs on a SYNTHETIC sample (data/sample/dti_sample.txt); nothing"
echo "to fetch. Real DTI benchmarks (featurize molecular graphs + protein vectors,"
echo "then write the format in data/README.md):"
echo
echo "  BindingDB : https://www.bindingdb.org/     (~2.9M measured Kd/Ki affinities)"
echo "  ChEMBL    : https://www.ebi.ac.uk/chembl/   (>20M bioactivity records)"
echo "  Davis     : kinase inhibitor affinities, 442 kinases x 68 drugs"
echo "  KIBA      : integrated kinase-inhibitor bioactivity benchmark"
echo
echo "Toolkits that featurize + train the full model:"
echo "  DeepPurpose : https://github.com/kexinhuang12345/DeepPurpose"
echo "  TorchDrug   : https://github.com/DeepGraphLearning/torchdrug"
echo "  DGL-LifeSci : https://github.com/awslabs/dgl-lifesci"
echo
echo "Bigger SYNTHETIC batch (no download):"
echo "  python scripts/make_synthetic.py --drugs 64 --proteins 16"
echo
echo "Target data dir: $PROJECT_ROOT/data"
