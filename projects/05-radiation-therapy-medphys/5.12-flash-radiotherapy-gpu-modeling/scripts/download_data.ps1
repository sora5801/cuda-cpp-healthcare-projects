# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.12 : FLASH Radiotherapy GPU Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This project's "input" is a tiny
# ensemble-configuration file (a parameter sweep); the physics lives in the
# code, so there is NO large binary dataset to download. Real FLASH-RT
# validation data (dosimetry, tumour oxygenation, radiolysis yields) is
# credentialed or not redistributable -- we point to it and generate a
# synthetic stand-in instead.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.12 -- FLASH Radiotherapy GPU Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC ensemble-config sample; no bulk download is"
Write-Host "needed to run the demo. Real-world reference data (for those extending the"
Write-Host "model) is credentialed / not redistributable:"
Write-Host ""
Write-Host "  * FLASH-RT experimental dosimetry -- CERN/CLEAR, UCLouvain, Stanford FLASH"
Write-Host "    programs (institutional access; verify each program's data-sharing policy)."
Write-Host "  * AAPM FLASH-RT working-group benchmark datasets (verify current URL)."
Write-Host "  * Published tumour oxygen-tension (pO2) measurements -- see the literature"
Write-Host "    (e.g. Eppendorf-electrode and EPR-oximetry studies)."
Write-Host "  * Geant4-DNA radiolysis validation datasets -- https://geant4-dna.org"
Write-Host ""
Write-Host "To (re)generate the committed synthetic sample offline, run:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "For a finer oxygen sweep (more ensemble members), run e.g.:"
Write-Host "    python scripts/make_synthetic.py --n-po2 32"
Write-Host ""
Write-Host "[download_data] Nothing to download; done."
