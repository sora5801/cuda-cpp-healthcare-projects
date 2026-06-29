# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.21 : Polarizable / AMOEBA Force Field MD
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The committed tiny SYNTHETIC sample is
# enough to run the demo offline; this script only points you at the real AMOEBA
# parameter sets and reference data so you can study them yourself.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.21 -- Polarizable / AMOEBA Force Field MD"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a tiny SYNTHETIC ensemble (data/sample/amoeba_ensemble.txt)"
Write-Host "that fully exercises the induced-dipole CG solver offline. No download is"
Write-Host "required to build, run, or verify the demo."
Write-Host ""
Write-Host "To study REAL AMOEBA force-field data and validation targets:"
Write-Host "  * AMOEBA / AMOEBA+ parameter files (Tinker .prm/.key):"
Write-Host "      https://github.com/TinkerTools/tinker        (params/ directory)"
Write-Host "      https://github.com/TinkerTools/poltype2      (AMOEBA+ parameterization)"
Write-Host "  * NIST thermophysical properties (water dielectric / dipole benchmarks):"
Write-Host "      https://webbook.nist.gov"
Write-Host "  * BindingDB experimental affinities (for FEP validation):"
Write-Host "      https://www.bindingdb.org"
Write-Host ""
Write-Host "These are large and/or license-restricted, so we do NOT redistribute them."
Write-Host "Respect each source's license. For a larger SYNTHETIC ensemble instead, run:"
Write-Host "    python scripts/make_synthetic.py --members 256"
