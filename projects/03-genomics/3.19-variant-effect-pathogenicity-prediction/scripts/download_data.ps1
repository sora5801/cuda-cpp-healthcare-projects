# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.19 -- Variant Effect / Pathogenicity Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# guidance, and NEVER bypasses credentials/registration. The committed tiny
# SYNTHETIC sample (data/sample/variants_sample.txt) already lets the demo run
# offline, so this script only points you at the real, labelled resources.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.19 -- Variant Effect / Pathogenicity Prediction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This educational project ships a SYNTHETIC sample and a fixed, UNTRAINED toy"
Write-Host "model, so no download is required to run the demo. To experiment with REAL"
Write-Host "variant-effect resources, fetch them yourself from the sources below and"
Write-Host "respect every license (do NOT commit redistribution-restricted data):"
Write-Host ""
Write-Host "  ClinVar (public)  pathogenic/benign calls"
Write-Host "    https://www.ncbi.nlm.nih.gov/clinvar/"
Write-Host "    VCF: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
Write-Host "  gnomAD (public)   allele frequencies + per-gene constraint"
Write-Host "    https://gnomad.broadinstitute.org/"
Write-Host "  MaveDB (public)   deep mutational scanning (DMS) atlas"
Write-Host "    https://www.mavedb.org/"
Write-Host "  HGMD (LICENSED -- registration required; NOT auto-downloaded here)"
Write-Host "    http://www.hgmd.cf.ac.uk/"
Write-Host ""
Write-Host "For a larger SYNTHETIC batch (e.g. to see the GPU overtake the CPU), run:"
Write-Host "    python scripts/make_synthetic.py --n 2000000"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
