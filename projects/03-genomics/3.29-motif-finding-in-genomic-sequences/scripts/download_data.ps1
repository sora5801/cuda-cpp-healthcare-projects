# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL ChIP-seq motif input (Windows)
# ---------------------------------------------------------------------------
# Project 3.29 : Motif Finding in Genomic Sequences
#
# This project's "real data" is a multi-FASTA of ChIP-seq peak sequences, which
# you assemble from a peak BED + a reference genome (no single file to grab).
# This script prints the recipe + tool links and defers to make_synthetic.py for
# an offline stand-in (CLAUDE.md sec 8). It downloads nothing and needs no
# credentials.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.29 -- Motif Finding in Genomic Sequences"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real input is a FASTA of transcription-factor ChIP-seq peak sequences:"
Write-Host "  1) Pick a TF ChIP-seq experiment and download its peak BED, e.g."
Write-Host "       ENCODE  : https://www.encodeproject.org/   (thousands of TF experiments)"
Write-Host "       ReMap   : https://remap.univ-amu.fr/        (~5k experiments)"
Write-Host "       GEO     : https://www.ncbi.nlm.nih.gov/geo/"
Write-Host "  2) Extract peak summit +/- ~100 bp from a reference genome FASTA:"
Write-Host "       bedtools getfasta -fi genome.fa -bed peaks.bed -fo peaks.fasta"
Write-Host "  3) Feed peaks.fasta to the program (or to MEME / HOMER for comparison)."
Write-Host "  4) Validate the recovered motif against a known PWM in JASPAR 2024:"
Write-Host "       https://jaspar.elixir.no/"
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --n 5000 --len 200"
Write-Host ""
Write-Host "Note: licenses vary by dataset -- respect each source's terms; do not"
Write-Host "redistribute genome-derived sequence unless the license permits it."
