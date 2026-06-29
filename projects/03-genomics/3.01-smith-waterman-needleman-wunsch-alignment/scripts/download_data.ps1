# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL sequences (Windows)
# ---------------------------------------------------------------------------
# Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
#
# This project aligns two sequences. "Real data" is just two FASTA records from
# a public database. This script prints how to obtain them and defers to
# make_synthetic.py for an offline stand-in (CLAUDE.md section 8). It downloads
# nothing and needs no credentials.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 3.01 -- Smith-Waterman / Needleman-Wunsch Alignment"
Write-Host ""
Write-Host "Pick two sequences from a public FASTA database, e.g.:"
Write-Host "  UniProt/Swiss-Prot : https://www.uniprot.org/downloads"
Write-Host "  NCBI RefSeq        : https://ftp.ncbi.nlm.nih.gov/refseq/"
Write-Host ""
Write-Host "Then write them as two lines (this project uses a DNA A/C/G/T alphabet):"
Write-Host "  line 1 = query sequence"
Write-Host "  line 2 = target sequence"
Write-Host "Save as data/sample/sequences_sample.txt (strip FASTA '>' headers/newlines)."
Write-Host ""
Write-Host "Offline stand-in (no download, reproducible):"
Write-Host "  python scripts/make_synthetic.py --motif 400 --mut 0.2"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
