# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs and
# NEVER bypasses credentials/registration. The real sources below need
# registration and/or a feature-extraction pipeline to turn structures and MD
# frames into pharmacophore feature points; that is out of scope for this
# teaching version, so this script only PRINTS guidance and defers to
# scripts/make_synthetic.py for an offline, fully-synthetic stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.33 -- Structure-Based Pharmacophore Modeling from MD Ensembles"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a fully SYNTHETIC sample (data/sample/pharmacophore_sample.txt)."
Write-Host "No real dataset is downloaded. Real sources (require registration and a"
Write-Host "pharmacophore-typing pipeline -- see THEORY.md 'Where this sits in the real world'):"
Write-Host ""
Write-Host "  GPCRmd trajectory archive : https://gpcrmd.org        (GPCR MD ensembles)"
Write-Host "  DUD-E actives/decoys      : https://dude.docking.org  (screening validation)"
Write-Host "  RCSB PDB                  : https://www.rcsb.org      (target-class structures)"
Write-Host "  ZINC drug-like library    : https://zinc20.docking.org (screening library)"
Write-Host ""
Write-Host "  Respect each source's license; none of that data is redistributed here."
Write-Host "  The committed tiny sample is enough to run the demo offline."
Write-Host "  For a larger SYNTHETIC screen, run:"
Write-Host "    python scripts/make_synthetic.py --N 1000000"
