# ===========================================================================
# scripts/download_data.ps1  --  Real cytometry-data pointers (Windows)
# ---------------------------------------------------------------------------
# Project 11.09 : Flow Cytometry & High-Content Screening Analysis. Nothing to fetch.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 11.09 -- Flow Cytometry & High-Content Screening Analysis"
Write-Host ""
Write-Host "Real data is in FCS files; export a few markers per event into the"
Write-Host "format in data/README.md ('N D K' then N rows of D floats in [0,1])."
Write-Host ""
Write-Host "  FlowRepository : http://flowrepository.org      (public FCS datasets)"
Write-Host "  FlowKit        : https://github.com/whitews/FlowKit   (read/transform FCS)"
Write-Host "  RAPIDS cuML    : https://github.com/rapidsai/cuml     (GPU clustering)"
Write-Host ""
Write-Host "Bigger synthetic set (no download):"
Write-Host "  python scripts/make_synthetic.py --scale 50"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
