# ===========================================================================
# scripts/download_data.ps1  --  Real BQSR-input pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.25 : Base Quality Score Recalibration (BQSR)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real BQSR needs (a) an aligned BAM and (b)
# known-variant VCFs; the committed synthetic sample stands in so the demo runs
# offline. This script only prints where to get the real inputs.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.25 -- Base Quality Score Recalibration (BQSR)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real BQSR consumes an aligned BAM plus KNOWN-VARIANT VCFs for masking."
Write-Host "This teaching project uses a small SYNTHETIC text alignment instead; to"
Write-Host "experiment with real inputs, fetch these (most are open, no account):"
Write-Host ""
Write-Host "  dbSNP build 155 (known SNPs) : https://www.ncbi.nlm.nih.gov/snp/"
Write-Host "  Mills & 1000G indels (GATK)  : https://storage.googleapis.com/genomics-public-data/"
Write-Host "  GIAB known-variant VCFs      : https://www.nist.gov/programs-projects/genome-bottle"
Write-Host "  1000 Genomes high-cov WGS    : https://www.internationalgenome.org/data"
Write-Host ""
Write-Host "Then convert a region to this project's text format (data/README.md):"
Write-Host "  REF/KNOWN/READS lines; one read per line as 'pos bases q0..q(L-1)'."
Write-Host ""
Write-Host "No download needed for the demo. Bigger SYNTHETIC set:"
Write-Host "  python scripts/make_synthetic.py --reads 50000"
Write-Host ""
Write-Host "Idempotent pattern when wiring a real fetch:"
Write-Host "  1) skip if the file already exists with the expected SHA256"
Write-Host "  2) print source URL + expected size + checksum"
Write-Host "  3) for credentialed sets, print registration instructions ONLY"
