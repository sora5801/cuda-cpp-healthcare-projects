# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point to the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real deterministic-transport benchmarks
# below are either license-restricted or need registration, so this script only
# PRINTS where to get them; the committed synthetic slab (data/sample/) plus
# scripts/make_synthetic.py let the demo run fully offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC slab (data/sample/slab_problem.txt), which is"
Write-Host "all the demo needs. The real-world references are listed below for study;"
Write-Host "each must be obtained under its own terms -- we do not redistribute them."
Write-Host ""
Write-Host "  1) AAPM TG-105 report (deterministic/MC dose calc guidance)"
Write-Host "       https://www.aapm.org/pubs/reports/  (search 'TG-105')"
Write-Host "  2) IROC Houston heterogeneous phantom program (credentialing)"
Write-Host "       https://www.mdanderson.org/  (search 'IROC Houston phantom')"
Write-Host "  3) IAEA photon cross-section / nuclear data services"
Write-Host "       https://www-nds.iaea.org/"
Write-Host "  4) Acuros XB validation: Varian/Eclipse white papers (vendor-published)"
Write-Host ""
Write-Host "To make a LARGER synthetic problem (more cells / higher S_N order), run:"
Write-Host "    python scripts/make_synthetic.py --ncell 400 --nord 16"
Write-Host ""
Write-Host "When wiring a real cross-section set, follow the idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right SHA256"
Write-Host "    2) print source URL + expected size + checksum before fetching"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
