# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL datasets (Windows)
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
Write-Host ""
Write-Host "Project 1.7 - Quantum Chemistry / DFT : full reference datasets" -ForegroundColor Cyan
Write-Host "-----------------------------------------------------------------"
Write-Host "The demo runs offline on data/sample/h2.txt. The datasets below are"
Write-Host "large corpora of precomputed DFT/CCSD(T) results (for ML and benchmarking),"
Write-Host "NOT inputs to this SCF. Listed for further study only:"
Write-Host ""
Write-Host "  QM9        134k organic molecules with DFT-computed properties"
Write-Host "             https://doi.org/10.6084/m9.figshare.978904"
Write-Host "  ANI-1ccx   CCSD(T)-level energies for diverse organic molecules"
Write-Host "             https://github.com/isayev/ANI1ccx_dataset"
Write-Host "  PubChemQC  DFT calculations for ~3M PubChem molecules"
Write-Host "             http://pubchemqc.riken.jp"
Write-Host "  CSD        Cambridge Structural Database (crystal structures; licensed)"
Write-Host "             https://www.ccdc.cam.ac.uk"
Write-Host ""
Write-Host "To make more inputs for THIS project (H/He molecules), use:"
Write-Host "  python scripts/make_synthetic.py --mol heh+"
Write-Host ""
Write-Host "No files were downloaded (by design)." -ForegroundColor Yellow
