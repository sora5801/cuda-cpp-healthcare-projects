# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.6 -- k-mer Counting & Minimiser Sketching
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The real read sets
# this project targets (NA12878 WGS, GenomeTrakr, GAGE) live in NCBI's Sequence
# Read Archive (SRA) and are fetched with the SRA Toolkit -- a separate install,
# not something we silently shell out to. This script therefore PRINTS the exact,
# reproducible commands and defers to scripts/make_synthetic.py for the offline
# demo. The committed tiny sample already runs the demo with zero downloads.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
$FullDir = Join-Path $DataDir "full"

Write-Host "[download_data] Project 3.6 -- k-mer Counting & Minimiser Sketching"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""

# Idempotence: if a FASTQ already sits in data/full/, do nothing.
if ((Test-Path $FullDir) -and (Get-ChildItem $FullDir -Filter *.fastq* -ErrorAction SilentlyContinue)) {
    Write-Host "[download_data] data/full already contains FASTQ files -- nothing to do."
    exit 0
}

Write-Host "The real read sets are public but large; fetch them with the NCBI SRA Toolkit:"
Write-Host ""
Write-Host "  1) Install the SRA Toolkit:  https://github.com/ncbi/sra-tools  (provides prefetch + fasterq-dump)"
Write-Host "  2) NA12878 Illumina WGS (human reference; accession SRR622457, tens of GB):"
Write-Host "       prefetch SRR622457"
Write-Host "       fasterq-dump --split-files -O `"$FullDir`" SRR622457"
Write-Host "  3) GenomeTrakr pathogen WGS (bacterial surveillance):  BioProject PRJNA183844"
Write-Host "       https://www.ncbi.nlm.nih.gov/bioproject/PRJNA183844"
Write-Host "  4) GAGE assembly benchmark (multi-species short reads):"
Write-Host "       http://gage.cbcb.umd.edu/"
Write-Host "  5) SRA front door (find more accessions):  https://www.ncbi.nlm.nih.gov/sra"
Write-Host ""
Write-Host "These are open-access; no credentials are required, but the SRA Toolkit"
Write-Host "is a separate install -- we never bundle or bypass it. After download, run"
Write-Host "the demo against your FASTQ once a small loader/extractor is wired up"
Write-Host "(the committed sample format is the tiny two-set 'k w s / >A / >B' file)."
Write-Host ""
Write-Host "For an offline OR larger SYNTHETIC problem, use:"
Write-Host "    python scripts/make_synthetic.py --genome 200000 --reads 5000 --readlen 100"
