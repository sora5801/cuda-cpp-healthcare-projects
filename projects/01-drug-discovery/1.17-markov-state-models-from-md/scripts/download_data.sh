#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real MD-trajectory pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.17 : Markov State Models from MD
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, and NEVER bypasses
# credentials/registration. There is nothing to auto-download here: building an
# MSM needs a featurized MD trajectory, which you produce from raw MD with a
# tool like PyEMMA/deeptime. This script prints the pointers and the expected
# input layout; the committed synthetic sample is enough to run the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.17 -- Markov State Models from MD"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "An MSM is built from a FEATURIZED trajectory. Featurize + tICA-reduce a"
echo "raw MD run, scale a few leading components to [0,1], and write them into"
echo "the format in data/README.md ('N D K lag' then N rows of D floats, in time order)."
echo
echo "  mdCATH         : https://huggingface.co/datasets/compsciencelab/mdcath  (5 us MD, 272 proteins)"
echo "  Fast-folders   : chignolin / Trp-cage / Villin (Piana/Shaw, publicly shared)"
echo "  GPCRmd         : https://gpcrmd.org                 (curated GPCR MD)"
echo "  D. E. Shaw     : millisecond trajectories via RCSB deposition"
echo "  PyEMMA         : https://github.com/markovmodel/PyEMMA   (featurize/tICA/cluster)"
echo "  deeptime       : https://github.com/deeptime-ml/deeptime (modern MSM/VAMP tools)"
echo
echo "Bigger synthetic trajectory (no download):"
echo "  python scripts/make_synthetic.py --frames 50000"
