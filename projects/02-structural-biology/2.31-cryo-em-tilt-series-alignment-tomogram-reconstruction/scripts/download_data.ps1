# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.31 -- Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. This project runs
# fully on the committed SYNTHETIC sample; the "real" cryo-ET datasets below are
# large research archives, so this script only prints where to get them.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.31 -- Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "  This project ships a TINY SYNTHETIC sample (data/sample/tilt_series_sample.txt)"
Write-Host "  that is enough to build and run the demo OFFLINE. No download is required."
Write-Host ""
Write-Host "  Real cryo-ET tilt series are large (multi-GB) research archives:"
Write-Host "    * EMPIAR tilt-series archives    https://www.ebi.ac.uk/empiar/"
Write-Host "        e.g. EMPIAR-10045 (in-situ ribosome tilt series)"
Write-Host "    * EMDB subtomogram averages      https://www.ebi.ac.uk/emdb/"
Write-Host "    * SHREC cryo-ET benchmark        (verify the current URL on the SHREC site)"
Write-Host "  Respect each entry's license before redistributing. These are NOT fetched"
Write-Host "  here (size + per-entry terms); see data/README.md for how to adapt them to"
Write-Host "  this project's simple text layout."
Write-Host ""
Write-Host "  For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --maxtilt 70 --step 4 --det 257 --img 192"
