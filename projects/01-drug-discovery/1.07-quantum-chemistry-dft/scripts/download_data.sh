#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.7 : Quantum Chemistry / DFT  (reduced-scope RHF/SCF)
#
# This project's DEMO needs NO download: it runs on the tiny committed molecule
# in data/sample/ (a hand-written H2 geometry). This script does NOT fetch
# anything automatically -- the catalog's reference datasets are large research
# corpora of precomputed quantum-chemistry results, used to BENCHMARK or TRAIN
# models, not to drive this teaching SCF. It prints where to get them and how they
# relate to this project. Respect each dataset's license (CLAUDE.md section 8).
# ===========================================================================
set -euo pipefail
cat <<'EOF'

Project 1.7 - Quantum Chemistry / DFT : full reference datasets
-----------------------------------------------------------------
The demo runs offline on data/sample/h2.txt. The datasets below are large
corpora of precomputed DFT/CCSD(T) results (for ML and benchmarking), NOT inputs
to this SCF. Listed for further study only:

  QM9        134k organic molecules with DFT-computed properties
             https://doi.org/10.6084/m9.figshare.978904
  ANI-1ccx   CCSD(T)-level energies for diverse organic molecules
             https://github.com/isayev/ANI1ccx_dataset
  PubChemQC  DFT calculations for ~3M PubChem molecules
             http://pubchemqc.riken.jp
  CSD        Cambridge Structural Database (crystal structures; licensed)
             https://www.ccdc.cam.ac.uk

To make more inputs for THIS project (H/He molecules), use:
  python scripts/make_synthetic.py --mol heh+

No files were downloaded (by design).
EOF
