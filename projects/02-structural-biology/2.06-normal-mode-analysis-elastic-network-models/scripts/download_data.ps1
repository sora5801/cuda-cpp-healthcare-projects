# ===========================================================================
# scripts/download_data.ps1  --  Real protein-structure pointers (Windows)
# ---------------------------------------------------------------------------
# Project 2.06 : Normal Mode Analysis / Elastic Network Models. Nothing to fetch.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 2.06 -- Normal Mode Analysis / Elastic Network Models"
Write-Host ""
Write-Host "Use a real structure: download a PDB/CIF, extract the CA atoms' x/y/z,"
Write-Host "and prepend 'N cutoff' to make data/sample/protein_ca.txt."
Write-Host ""
Write-Host "  RCSB PDB     : https://www.rcsb.org           (experimental structures)"
Write-Host "  AlphaFold DB : https://alphafold.ebi.ac.uk    (predicted structures)"
Write-Host "  ProDy        : https://github.com/prody/ProDy  (ANM/GNM; parses PDB)"
Write-Host ""
Write-Host "Tip (with ProDy): prody.parsePDB('1abc').select('name CA').getCoords()"
Write-Host ""
Write-Host "Bigger synthetic structure (no download):"
Write-Host "  python scripts/make_synthetic.py --N 120"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
