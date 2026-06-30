# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.4 : Cryo-ET Subtomogram Averaging
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs and
# NEVER bypasses credentials/registration. Real cryo-ET subtomograms are large
# and have their own usage/citation policies, so this script prints links and
# instructions ONLY and defers to scripts/make_synthetic.py for the offline
# stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.4 -- Cryo-ET Subtomogram Averaging"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC committed sample (data/sample/), so the"
Write-Host "demo runs offline with no download. Real cryo-ET data lives here:"
Write-Host ""
Write-Host "  EMDB STA maps       : https://www.ebi.ac.uk/emdb/"
Write-Host "  EMPIAR raw data     : https://www.ebi.ac.uk/empiar/  (e.g. EMPIAR-10064)"
Write-Host "  SHREC cryo-ET bench : search 'SHREC subtomogram challenge' (URL moves yearly)"
Write-Host "  CryoDRGN-ET         : https://github.com/ml-struct-bio/cryodrgn"
Write-Host ""
Write-Host "These sets are large (GBs-TBs) and have their own citation/usage terms;"
Write-Host "this repo does NOT redistribute them. Respect each source's license."
Write-Host ""
Write-Host "To (re)generate the synthetic sample the demo uses:"
Write-Host "    python scripts/make_synthetic.py            # default: 6 cubes, 16^3, 12 angles"
Write-Host "    python scripts/make_synthetic.py --d 32     # a bigger synthetic problem"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY (never bypass)"
Write-Host "    4) extract d^3 cubes around picked particles into the loader's text layout"
Write-Host "       (header 'n_sub d n_angles', then the reference cube, then candidate cubes)"
