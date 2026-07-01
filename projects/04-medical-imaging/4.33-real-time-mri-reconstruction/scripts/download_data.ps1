# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.33 : Real-Time MRI Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Every real cardiac / dynamic MRI raw-
# k-space dataset for this project sits behind a challenge registration or data-use
# agreement, so this script only PRINTS the registration instructions and links, and
# points at make_synthetic.py for an offline stand-in. The committed data/sample/
# already lets the demo run offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.33 -- Real-Time MRI Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "All real dynamic/cardiac MRI raw-k-space datasets require a challenge"
Write-Host "registration or a data-use agreement and CANNOT be auto-downloaded. Register"
Write-Host "with the provider, then export a radial (or re-gridded) k-space trajectory into"
Write-Host "the text layout documented in data/README.md."
Write-Host ""
Write-Host "  CMRxRecon 2023 -- cardiac MRI reconstruction challenge (multi-coil k-space):"
Write-Host "     https://cmrxrecon.github.io/"
Write-Host ""
Write-Host "  ACDC -- Automated Cardiac Diagnosis Challenge (cine cardiac MRI):"
Write-Host "     https://www.creatis.insa-lyon.fr/Challenge/acdc/"
Write-Host ""
Write-Host "  OCMR -- open cardiovascular MRI raw data (incl. real-time free-breathing):"
Write-Host "     https://ocmr.info/"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample in data/sample/ is enough to run the demo."
Write-Host "For a larger synthetic problem (more spokes / frames / a bigger grid), run:"
Write-Host "    python scripts/make_synthetic.py --n 64 --spokes 128 --win 34 --frames 8"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY (never bypass)"
