# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md SS8): idempotent, documented, prints the source URLs and
# NEVER bypasses credentials/registration. Real scRNA-seq matrices ship as
# .h5ad / .mtx / 10x HDF5 and need Scanpy/AnnData to export a dense slice into
# THIS project's plain-text format, so this script prints the path rather than
# silently converting. The committed synthetic sample already runs the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.12 -- Single-Cell RNA-seq Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample (data/sample/scrna_sample.txt) is"
Write-Host "enough to build and run the demo offline. No download is required."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem (still offline, deterministic):"
Write-Host "    python scripts/make_synthetic.py --cells 300 --genes 48 --k 10"
Write-Host ""
Write-Host "REAL scRNA-seq datasets (from the catalog):"
Write-Host "  * Human Cell Atlas  : https://www.humancellatlas.org/"
Write-Host "  * 10x Genomics sets : https://www.10xgenomics.com/resources/datasets"
Write-Host "  * CellxGene Census  : https://cellxgene.cziscience.com/   (50M+ cells)"
Write-Host "  * NCBI GEO          : https://www.ncbi.nlm.nih.gov/geo/"
Write-Host ""
Write-Host "To use a real matrix here, export a dense slice with Scanpy in Python:"
Write-Host "    import scanpy as sc"
Write-Host "    a = sc.read_10x_h5('filtered_feature_bc_matrix.h5')   # or sc.read_h5ad(...)"
Write-Host "    a = a[:300, :48]                                      # tiny teaching slice"
Write-Host "    # then write 'N G k target_sum' + rows of '<label> count0..' as in data/README.md"
Write-Host ""
Write-Host "Some assets require (free) registration. This script will NOT bypass any"
Write-Host "login or license; it only points you to the source pages above."
