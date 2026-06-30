# ===========================================================================
# scripts/download_data.ps1  --  How to get a REAL reference genome (Windows)
# ---------------------------------------------------------------------------
# Project 3.17 : CRISPR Guide Design & Off-Target Scoring
#
# This project's "real data" is a reference GENOME (FASTA) to scan a guide
# against, plus optional benchmark guide sets. None of it requires credentials,
# but the files are large (a human chromosome is hundreds of MB), so this script
# PRINTS THE RECIPE and defers to scripts/make_synthetic.py for an offline
# stand-in (CLAUDE.md §8). It downloads nothing by itself.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.17 -- CRISPR Guide Design & Off-Target Scoring"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real off-target scanning runs a guide against a reference genome:"
Write-Host "  1) Download a reference genome (FASTA) from UCSC, e.g. one human"
Write-Host "     chromosome (smaller than the whole genome):"
Write-Host "       https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz"
Write-Host "     (whole genome: https://genome.ucsc.edu/  | mouse: mm10)"
Write-Host "  2) gunzip it, strip the '>' header and newlines to get one ACGT string."
Write-Host "  3) Write the loader format documented in data/README.md:"
Write-Host "       guide  myGuide  <20-letter ACGT spacer>"
Write-Host "       genome <the ACGT string>"
Write-Host "  4) Validated guide efficiencies/off-targets for benchmarking:"
Write-Host "       CRISPOR  https://crispor.gi.ucsc.edu/"
Write-Host "       GeCKO v2 https://www.addgene.org/pooled-library/"
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py --filler 100000   # a larger synthetic genome"
Write-Host ""
Write-Host "Note: the guide must be exactly 20 ACGT bases; the genome must be at"
Write-Host "least 23 bases (one window). See data/README.md for the exact format."
