# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL/real data (Windows)
# ---------------------------------------------------------------------------
# Project 5.8 -- Linac QA & Machine Performance Assessment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. There is no single downloadable file
# this demo consumes directly -- real linac-QA data is machine/vendor-specific
# and often site-restricted -- so this script prints authoritative pointers and
# defers to scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.8 -- Linac QA & Machine Performance Assessment"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This demo runs on a SYNTHETIC sample (data/sample/qa_planes_sample.txt)."
Write-Host "No real dataset is fetched. Authoritative reference material:"
Write-Host ""
Write-Host "  AAPM TG-119  IMRT QA test plans   : https://www.aapm.org/pubs/reports/RPT_82.pdf"
Write-Host "  AAPM TG-218  tolerance limits     : https://doi.org/10.1002/mp.12810"
Write-Host "  OpenMedPhys / awesome-medphys     : https://github.com/jrkerns/awesome-medphys"
Write-Host "  Pylinac (example EPID/log data)   : https://github.com/jrkerns/pylinac"
Write-Host ""
Write-Host "Respect every dataset license; credentialed sets require registration"
Write-Host "(this script does NOT bypass it). Regenerate the offline sample with:"
Write-Host "    python scripts/make_synthetic.py            # 24x24 planes"
Write-Host "    python scripts/make_synthetic.py --nx 128 --ny 128   # larger synthetic"
