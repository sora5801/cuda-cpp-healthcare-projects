# ===========================================================================
# scripts/download_data.ps1  --  Realistic MC physics pointers (Windows)
# ---------------------------------------------------------------------------
# Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
# This project generates its own data from parameters; there is nothing to
# download. This prints where real cross-section data / engines live. See sec 8.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 5.01 -- Monte Carlo Dose Calculation (simplified slab)"
Write-Host ""
Write-Host "There is no file to download: the simulation makes its own data from"
Write-Host "the parameters in data/sample/mc_params.txt."
Write-Host ""
Write-Host "For REAL physics (cross sections, electron transport, CT geometry):"
Write-Host "  EGSnrc : https://github.com/nrc-cnrc/EGSnrc   (reference MC + PEGS data)"
Write-Host "  GATE   : https://github.com/OpenGATE/opengate (Geant4 clinical MC)"
Write-Host "  MC-GPU : https://github.com/DIDSR/MCGPU        (open CUDA photon MC)"
Write-Host ""
Write-Host "More histories (smoother statistics):"
Write-Host "  python scripts/make_synthetic.py --photons 4000000"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
