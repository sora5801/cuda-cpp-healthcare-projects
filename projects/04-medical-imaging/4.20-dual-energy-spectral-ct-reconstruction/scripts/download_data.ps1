# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL spectral-CT data (Windows)
# ---------------------------------------------------------------------------
# Project 4.20 : Dual-Energy / Spectral CT Reconstruction
#
# This project's committed sample is synthetic. Real dual-energy / photon-
# counting CT datasets require registration and are large, so this script only
# PRINTS pointers and instructions -- it downloads nothing and never bypasses any
# credential or license gate (CLAUDE.md section 8). Use make_synthetic.py for an
# offline, reproducible stand-in.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 4.20 -- Dual-Energy / Spectral CT Reconstruction"
Write-Host ""
Write-Host "Real spectral-CT data (register / accept the license on each site):"
Write-Host "  * AAPM Spectral CT challenge data -- verify URL at https://www.aapm.org/"
Write-Host "  * MARS photon-counting CT datasets -- https://www.marsbioimaging.com/"
Write-Host "  * TCIA DECT collections           -- https://www.cancerimagingarchive.net/"
Write-Host "  * XCAT phantom simulated DECT     -- license from Duke"
Write-Host ""
Write-Host "Realistic physics inputs (to replace the analytic curves in the code):"
Write-Host "  * NIST XCOM attenuation cross-sections -- https://physics.nist.gov/PhysRefData/Xcom/"
Write-Host "  * SpekPy tube spectra                  -- https://bitbucket.org/spekpy/"
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --n 100000   # many synthetic bins"
Write-Host ""
Write-Host "Note: convert real sinograms to the simple text format in data/README.md,"
Write-Host "or extend src/reference_cpu.cpp::load_sinogram to read your format."
Write-Host "Target data dir: $ProjectRoot\data"
