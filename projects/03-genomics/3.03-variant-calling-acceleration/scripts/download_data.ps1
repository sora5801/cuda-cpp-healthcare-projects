# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.3 : Variant Calling Acceleration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The
# benchmark resources below are large and/or access-controlled, so this script
# prints instructions + links ONLY and defers to scripts/make_synthetic.py for
# the offline stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.3 -- Variant Calling Acceleration"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a tiny SYNTHETIC sample (data/sample/) so the demo"
Write-Host "runs offline. No real dataset is downloaded automatically -- the real"
Write-Host "benchmark resources are large and some require registration."
Write-Host ""
Write-Host "REAL BENCHMARK RESOURCES (open in a browser, follow each site's terms):"
Write-Host "  * GiaB truth sets HG001-HG007 (gold-standard germline calls):"
Write-Host "      https://www.nist.gov/programs-projects/genome-bottle"
Write-Host "  * ClinVar (clinically interpreted variants):"
Write-Host "      https://www.ncbi.nlm.nih.gov/clinvar/"
Write-Host "  * gnomAD v4 (population allele frequencies):"
Write-Host "      https://gnomad.broadinstitute.org/"
Write-Host "  * 1000 Genomes high-coverage WGS:"
Write-Host "      https://www.internationalgenome.org/data"
Write-Host ""
Write-Host "TO USE REAL DATA with this teaching kernel:"
Write-Host "  1) Pick one locus; extract candidate haplotypes (local assembly of the"
Write-Host "     active region) and the overlapping reads from a BAM."
Write-Host "  2) Convert to the text format in data/README.md (haplotypes + reads +"
Write-Host "     Phred qualities) and pass the file path as argv[1] to the exe."
Write-Host ""
Write-Host "FOR A LARGER SYNTHETIC PROBLEM (no download needed):"
Write-Host "    python scripts/make_synthetic.py --reads 4096 --read-len 100 --hap-len 120"
Write-Host ""
Write-Host "[download_data] Nothing downloaded (by design). The demo uses the synthetic sample."
