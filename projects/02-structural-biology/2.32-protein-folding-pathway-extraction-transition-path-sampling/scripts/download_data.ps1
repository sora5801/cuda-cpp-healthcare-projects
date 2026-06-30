# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. A research-grade TPS run needs
# all-atom MD trajectories (Anton/Shaw, GPCRmd) and protein structures (PDB);
# this project ships a SYNTHETIC 1-D teaching model instead, so there is no bulk
# download to perform -- this script only prints where the real inputs live and
# defers to scripts/make_synthetic.py for the offline parameter file.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.32 -- Protein Folding Pathway Extraction (TPS)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project is a SYNTHETIC 1-D teaching model: its only input is the"
Write-Host "parameter file data/sample/tps_params.txt, already committed. There is no"
Write-Host "bulk dataset to download. The demo runs fully offline."
Write-Host ""
Write-Host "To (re)generate or rescale the synthetic parameter file:"
Write-Host "    python scripts/make_synthetic.py --shooters 16384 --barrier 6.0"
Write-Host ""
Write-Host "Real, research-grade TPS inputs (require accounts / external requests --"
Write-Host "this script does NOT fetch or bypass them):"
Write-Host "    * Anton / D. E. Shaw millisecond folding trajectories (request access)"
Write-Host "    * GPCRmd MD trajectories & pathways:  https://gpcrmd.org"
Write-Host "    * Protein structures (Trp-cage 1L2Y, chignolin 5AWL):  https://www.rcsb.org"
Write-Host "    * SAMPL host-guest kinetics challenges (search 'SAMPL challenge')"
Write-Host ""
Write-Host "Production TPS engines to study (do not copy wholesale):"
Write-Host "    OpenPathSampling, WESTPA, HTMD -- see README 'Prior art & further reading'."
