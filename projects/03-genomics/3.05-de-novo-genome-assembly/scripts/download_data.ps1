# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.5 : De Novo Genome Assembly  (all-vs-all read-overlap stage)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# guidance, and NEVER bypasses credentials/registration. The real assembly
# benchmark datasets are gigabytes, so this script GUIDES you to them rather
# than auto-downloading; the committed tiny synthetic sample is enough to run
# the demo, and scripts/make_synthetic.py scales it up offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.5 -- De Novo Genome Assembly"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project ships a tiny SYNTHETIC sample (data/sample/reads_sample.fasta)"
Write-Host "that is sufficient for the demo. Real de-novo assembly benchmark data is large;"
Write-Host "fetch it from the sources below (none require credentials, but all are GBs):"
Write-Host ""
Write-Host "  * CHM13 (T2T) human reference   : https://github.com/marbl/CHM13"
Write-Host "  * GenomeArk (vertebrate genomes): https://genomeark.github.io/"
Write-Host "  * Human Pangenome (HPRC)        : https://humanpangenome.org/"
Write-Host "  * SRA PacBio HiFi / ONT reads   : https://www.ncbi.nlm.nih.gov/sra"
Write-Host "      (use the SRA Toolkit: 'prefetch <ACC>' then 'fasterq-dump <ACC>')"
Write-Host ""
Write-Host "  Convert downloaded reads to the FASTA this demo expects (>header / sequence),"
Write-Host "  or generate a larger SYNTHETIC set offline:"
Write-Host "    python scripts/make_synthetic.py --genome-len 50000 --read-len 1000 --step 200 --error-rate 0.02"
Write-Host ""
Write-Host "  Idempotent pattern to follow when wiring a real fetch:"
Write-Host "    1) skip the download if the file already exists with the right SHA256"
Write-Host "    2) print source URL + expected size + checksum before downloading"
Write-Host "    3) for any credentialed mirror, print registration instructions ONLY"
