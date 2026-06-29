# ===========================================================================
# scripts/download_data.ps1  --  Molecular-resolution RD pointers (Windows)
# ---------------------------------------------------------------------------
# Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion. Nothing to download.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 14.02 -- Spatial / Whole-Cell Reaction-Diffusion"
Write-Host ""
Write-Host "There is no file to download: the grid is built from the parameters in"
Write-Host "data/sample/grayscott_params.txt."
Write-Host ""
Write-Host "This flagship is the continuum (grid stencil) TEACHING version. The full"
Write-Host "project is PARTICLE-based reaction-diffusion at molecular resolution:"
Write-Host "  ReaDDy  : https://github.com/readdy/readdy   (GPU particle RD)"
Write-Host "  Smoldyn : https://github.com/ssandrews/Smoldyn"
Write-Host "  MCell   : https://mcell.org/"
Write-Host "  STEPS   : https://github.com/CNS-OIST/STEPS"
Write-Host ""
Write-Host "Bigger grid (no download):"
Write-Host "  python scripts/make_synthetic.py --nx 256 --ny 256 --steps 12000"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
