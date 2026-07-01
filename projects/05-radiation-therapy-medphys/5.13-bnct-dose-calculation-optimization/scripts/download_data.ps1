# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This project's demo runs entirely on
# the committed SYNTHETIC sample (data/sample/bnct_params.txt); the real BNCT
# reference data below is optional and only relevant if you extend the model
# toward a validated code. We therefore print guidance rather than downloading
# license-restricted or registration-gated material.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.13 -- BNCT Dose Calculation & Optimization"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC parameter sample that fully drives the demo:"
Write-Host "    $DataDir\sample\bnct_params.txt"
Write-Host "Regenerate or scale it with:"
Write-Host "    python scripts/make_synthetic.py --histories 1000000 --seed 7"
Write-Host ""
Write-Host "OPTIONAL real BNCT references (for extending toward a validated code):"
Write-Host "  * OpenMC (open-source, GPU-capable neutron Monte Carlo) + validation tests:"
Write-Host "      https://github.com/openmc-dev/openmc/tree/develop/tests"
Write-Host "  * GATE 10 neutron transport for BNCT: https://github.com/OpenGATE/opengate"
Write-Host "  * ENDF/B-VIII.0 evaluated neutron cross sections (used by real codes):"
Write-Host "      https://www.nndc.bnl.gov/endf/   (verify current URL; large, evaluated)"
Write-Host "  * IAEA BNCT benchmark cases: search https://www.iaea.org (registration/"
Write-Host "    request may be required -- this script will NOT bypass that)."
Write-Host ""
Write-Host "None of the above is required to run the demo. If you download ENDF/B or"
Write-Host "IAEA data yourself, respect each source's license; do not redistribute"
Write-Host "gated data through this repository (CLAUDE.md section 8)."
Write-Host ""
Write-Host "Idempotent pattern to follow when wiring a real fetch:"
Write-Host "    1) skip the download if the file already exists with the right SHA256"
Write-Host "    2) print source URL + expected size + checksum before downloading"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
