# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real dose-engine benchmark sets
# below all require registration or forbid redistribution, so this script only
# prints instructions + links and defers to scripts/make_synthetic.py for the
# offline stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.4 -- Collapsed-Cone / Superposition-Convolution Dose"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a TINY SYNTHETIC phantom (data/sample/phantom.txt) that is"
Write-Host "enough to run the demo offline. The reference benchmark datasets below are for"
Write-Host "going further; each requires registration or has redistribution limits, so we"
Write-Host "do NOT download them automatically -- follow the links and accept each license."
Write-Host ""
Write-Host "  * AAPM TG-105 report + heterogeneous-media dose test cases:"
Write-Host "      https://www.aapm.org/pubs/reports/  (search 'TG-105')"
Write-Host "  * IROC Houston phantom credentialing (lung phantom CT + dosimetry):"
Write-Host "      https://www.mdanderson.org/  (IROC Houston Quality Assurance Center)"
Write-Host "  * TCIA clinical photon planning datasets (CT + RTDOSE/RTPLAN DICOM):"
Write-Host "      https://www.cancerimagingarchive.net/  (register, then browse RT collections)"
Write-Host "  * CIRS IMRT verification phantom data: https://www.cirsinc.com/"
Write-Host ""
Write-Host "To regenerate the committed SYNTHETIC phantom instead:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "When wiring a real dataset later, keep this idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right SHA256"
Write-Host "    2) print source URL + expected size + checksum before fetching"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
