# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.13 : Gene Regulatory Network Inference (ARACNE: MI + DPI)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. This
# project's "real data" is single-cell RNA-seq for which there is no committable
# ground-truth network, so the demo runs on a labeled-synthetic sample; this
# script prints how to obtain real data and defers to make_synthetic.py offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.13 -- Gene Regulatory Network Inference"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a labeled-SYNTHETIC sample with a KNOWN ground-truth"
Write-Host "network (TF->A->B, TF->C, D->E; F,G,H,I noise) so you can watch ARACNE"
Write-Host "recover it. Real scRNA-seq has no such ground truth to redistribute."
Write-Host ""
Write-Host "To try REAL data, obtain an expression matrix (genes x cells) from:"
Write-Host "  * Gene Expression Omnibus (GEO)        https://www.ncbi.nlm.nih.gov/geo/"
Write-Host "  * BEELINE benchmark GRN datasets       https://github.com/Murali-group/BEELINE"
Write-Host "  * Human Cell Atlas scRNA-seq           https://www.humancellatlas.org"
Write-Host "  * ENCODE TF binding ChIP-seq (truth)   https://www.encodeproject.org"
Write-Host "then reshape it to this loader's text format (see data/README.md):"
Write-Host "  line 1: '<n_genes> <n_samples>'; then one row per gene: '<name> v0 v1 ...'"
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --samples 400"
Write-Host ""
Write-Host "Note: BEELINE/GEO/HCA are large and license-bound; respect each license."
Write-Host "This script downloads nothing by itself and never bypasses credentials."
