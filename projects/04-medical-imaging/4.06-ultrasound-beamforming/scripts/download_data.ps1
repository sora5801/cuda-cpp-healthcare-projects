# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL ultrasound RF data (Windows)
# ---------------------------------------------------------------------------
# Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Real RF datasets need registration
# or a simulator, so this script prints instructions + links only and defers to
# scripts/make_synthetic.py for an offline, reproducible stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.6 -- Ultrasound Beamforming"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Options for real / standard RF beamforming data:"
Write-Host "  * PICMUS (Plane-Wave Imaging Challenge in Medical Ultrasound) --"
Write-Host "    canonical RF datasets (point targets, cysts, in-vivo) for beamformer"
Write-Host "    evaluation: https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/"
Write-Host "    (registration may be required; this script does NOT bypass it)."
Write-Host "  * Field II (https://field-ii.dk/) -- CPU simulator that GENERATES"
Write-Host "    realistic RF data for arbitrary phantoms; export to the data/README"
Write-Host "    format, then beamform with this project's GPU kernel."
Write-Host "  * k-Wave / k-Wave-Fluid-CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA)"
Write-Host "    -- full-wave acoustic propagation (more physical than our point model)."
Write-Host "  * MUST (https://www.biomecardio.com/MUST/) -- reference DAS + sample RF."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py"
Write-Host "  python scripts/make_synthetic.py --elements 128 --samples 512 --nx 192 --nz 192 --extra"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for credentialed sets, print registration instructions ONLY"
