# ===========================================================================
# scripts/download_data.ps1  --  Pointers to REAL RNA-seq data (Windows / PS)
# ---------------------------------------------------------------------------
# Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The DEMO needs no download -- the
# committed synthetic sample (data/sample/reads_sample.txt) is enough. This
# script just guides you to real data if you want to go further.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.23 -- Splice-Aware RNA Alignment"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project runs fully on the committed SYNTHETIC sample:"
Write-Host "    data/sample/reads_sample.txt   (a 3-exon gene + 6 reads)"
Write-Host "No download is required for the demo."
Write-Host ""
Write-Host "To experiment with REAL RNA-seq, fetch from these sources yourself:"
Write-Host ""
Write-Host "  * ENCODE RNA-seq FASTQs + GENCODE annotation (open access):"
Write-Host "      https://www.encodeproject.org/    (reads)"
Write-Host "      https://www.gencodegenes.org/     (GTF gene models = true exons/introns)"
Write-Host "  * SRA RNA-seq benchmarks (SEQC/MAQC):"
Write-Host "      https://www.ncbi.nlm.nih.gov/sra"
Write-Host "      (use the SRA Toolkit 'prefetch' + 'fasterq-dump' to get FASTQ)"
Write-Host "  * GTEx tissue RNA-seq (CONTROLLED ACCESS -- individual-level via dbGaP):"
Write-Host "      https://gtexportal.org/"
Write-Host "      Register at dbGaP; this script does NOT and CANNOT bypass that."
Write-Host ""
Write-Host "Recommended idempotent pattern when you wire a fetch in:"
Write-Host "    1) skip the download if the file already exists with the right SHA256"
Write-Host "    2) print the source URL + expected size + SHA256 before downloading"
Write-Host "    3) for controlled-access sets, print registration instructions ONLY"
Write-Host ""
Write-Host "For a larger SYNTHETIC batch instead, just re-seed the generator:"
Write-Host "    python scripts/make_synthetic.py --seed 7"
