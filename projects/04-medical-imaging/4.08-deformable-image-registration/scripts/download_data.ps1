# ===========================================================================
# scripts/download_data.ps1  --  Real DIR dataset pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, and it NEVER bypasses
# credentials or registration. This project ships its own SYNTHETIC image pair,
# so there is nothing to auto-download; the real registration datasets below all
# require agreeing to a data-use / challenge license, which you must do yourself.
# We print the links + instructions and defer to make_synthetic.py for offline
# runs -- a bigger synthetic pair is one command away.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.8 -- Deformable Image Registration"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "There is NOTHING to download for the demo: this project generates its"
Write-Host "own synthetic fixed/moving image pair (data/sample/dir_pair.txt)."
Write-Host ""
Write-Host "REAL registration benchmarks (each needs you to register/accept a license):"
Write-Host "  Learn2Reg challenge : https://learn2reg.grand-challenge.org/"
Write-Host "      lung / brain / abdominal CT + MR pairs with evaluation."
Write-Host "  OASIS brain MRI     : https://www.oasis-brains.org/"
Write-Host "      the brain set used by the Learn2Reg inter-subject task."
Write-Host "  DIR-Lab lung CT     : https://dir-lab.com/"
Write-Host "      4D-CT respiratory pairs with expert landmarks (gold-standard TRE)."
Write-Host ""
Write-Host "Do NOT commit any of the above into this repo (license + patient data)."
Write-Host "Convert a real image slice into this project's text format yourself, or"
Write-Host "make a bigger SYNTHETIC pair (no download, fully offline):"
Write-Host "  python scripts/make_synthetic.py --nx 128 --ny 128 --shift 8.0"
