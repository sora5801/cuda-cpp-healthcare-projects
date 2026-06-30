# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL reads (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.16 : Sequence Error Correction
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints source URLs and
# NEVER bypasses credentials/registration. This project's demo runs on a
# committed SYNTHETIC sample; real benchmark reads are large public archives, so
# this script prints the recipe and defers to make_synthetic.py for an offline
# stand-in. It downloads nothing by itself.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.16 -- Sequence Error Correction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo runs on the committed SYNTHETIC sample (data/sample/reads_sample.txt)."
Write-Host "To experiment on REAL reads with known error profiles, use a public archive:"
Write-Host ""
Write-Host "  GAGE benchmark short reads (known reference -> measurable error rates):"
Write-Host "    http://gage.cbcb.umd.edu/"
Write-Host "  GIAB / Genome in a Bottle (NIST truth sets to score corrected reads):"
Write-Host "    https://www.nist.gov/programs-projects/genome-bottle"
Write-Host "  SRA (ONT / PacBio CLR high-error long reads -- a different regime):"
Write-Host "    https://www.ncbi.nlm.nih.gov/sra"
Write-Host ""
Write-Host "Convert a FASTA/FASTQ slice into this project's simple text format"
Write-Host "(see data/README.md): one '<n> <has_truth>' header line, then the reads."
Write-Host "Set has_truth=0 when you have no error-free reference."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py                 # the committed 120-read sample"
Write-Host "  python scripts/make_synthetic.py --reads 200000  # a larger, GPU-friendlier set"
Write-Host ""
Write-Host "Note: real corrected-read evaluation respects each archive's terms; this"
Write-Host "script never bypasses registration or logins."
