# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.21 : Microcirculation & Oxygen Transport
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. If a
# dataset needs an account, this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.21 -- Microcirculation & Oxygen Transport"
Write-Host "[download_data] Target data dir: $DataDir"

Write-Host ""
Write-Host "This project ships a SYNTHETIC sample (data/sample/microvessel_network.txt)"
Write-Host "and does not require any download to run the demo. Real microvascular data:"
Write-Host "  - Vascular Model Repository            : http://www.vascularmodel.com"
Write-Host "  - Allen Institute two-photon microscopy: https://portal.brain-map.org"
Write-Host "  - PhysioNet O2 saturation waveforms    : https://physionet.org (credentialed)"
Write-Host "  - Secomb-group microvascular networks  : https://secomb.org (verify terms)"
Write-Host "Respect each dataset's license/registration; this script never bypasses it."
Write-Host ""
Write-Host "  For a larger SYNTHETIC problem, run:"
Write-Host "    python scripts/make_synthetic.py --nx 24 --ny 24 --nz 16"
Write-Host ""
Write-Host "  When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
