# ===========================================================================
# scripts/download_data.ps1  --  Real proteomics-data pointers (Windows)
# ---------------------------------------------------------------------------
# Project 12.01 : Mass-Spectrometry Proteomics Search. Nothing to download.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 12.01 -- Mass-Spectrometry Proteomics Search"
Write-Host ""
Write-Host "Real data: observed MS/MS spectra (mzML) searched against a peptide DB."
Write-Host "Bin observed peaks + theoretical fragments to a common grid, then write the"
Write-Host "format in data/README.md."
Write-Host ""
Write-Host "  ProteomeXchange / PRIDE : https://www.proteomexchange.org  (raw/mzML)"
Write-Host "  MSFragger               : https://github.com/Nesvilab/MSFragger"
Write-Host "  GiCOPS (GPU search)     : https://github.com/pcdslab/gicops"
Write-Host "  OpenMS                  : https://github.com/OpenMS/OpenMS  (mzML I/O)"
Write-Host ""
Write-Host "Bigger synthetic set (no download):"
Write-Host "  python scripts/make_synthetic.py --N 8192"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
