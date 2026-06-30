# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.11 : GWAS at Scale
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# access terms, and NEVER bypasses credentials/registration. Real GWAS cohorts
# are controlled-access and cannot be redistributed, so this script does NOT
# download genotypes -- it prints exactly how to obtain them legally and defers
# to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.11 -- GWAS at Scale"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real GWAS cohorts are CONTROLLED-ACCESS and may NOT be redistributed."
Write-Host "This script will not (and cannot) bypass that. To use real data:"
Write-Host ""
Write-Host "  UK Biobank  (~500k individuals, ~800k variants)"
Write-Host "    Apply for access: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access"
Write-Host "    Genotypes ship as PLINK .bed/.bim/.fam or BGEN; convert with PLINK2."
Write-Host ""
Write-Host "  dbGaP  (controlled-access GWAS datasets)"
Write-Host "    Data Access Request: https://www.ncbi.nlm.nih.gov/gap/"
Write-Host ""
Write-Host "  GWAS Catalog  (OPEN published summary statistics -- no genotypes)"
Write-Host "    Browse / download: https://www.ebi.ac.uk/gwas/  (useful to cross-check hits)"
Write-Host ""
Write-Host "  gnomAD  (OPEN allele-frequency / LD reference panels)"
Write-Host "    https://gnomad.broadinstitute.org/"
Write-Host ""
Write-Host "A real loader would read PLINK2 .bed/.pgen or BGEN, not this demo's text format."
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample (data/sample/gwas_sample.txt) already lets"
Write-Host "the demo run offline. For a larger synthetic cohort, run:"
Write-Host "    python scripts/make_synthetic.py --n 2000 --m 5000 --out data/sample/gwas_big.txt"
