# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.28 : Covalent Docking
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. This project's demo
# runs entirely on the committed SYNTHETIC sample (data/sample/), so no download
# is required to build, run, or learn. This script only points at the REAL
# covalent-docking resources a learner could study next -- it does not (and must
# not) attempt to harvest them automatically.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.28 -- Covalent Docking"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC sample (data/sample/covalent_sample.txt)"
Write-Host "that is sufficient for the demo. No external download is needed."
Write-Host ""
Write-Host "Real covalent-docking resources to study (manual, license-respecting):"
Write-Host "  * PDB covalent complexes ...... https://www.rcsb.org   (search 'covalent ligand')"
Write-Host "      e.g. KRAS G12C + sotorasib (6OIM), BTK + ibrutinib (5P9J)."
Write-Host "  * ChEMBL covalent inhibitors .. https://www.ebi.ac.uk/chembl/"
Write-Host "  * BindingDB covalent entries .. https://www.bindingdb.org"
Write-Host "  * CovDocker benchmark (2025) .. arXiv:2506.21085 (verify the released URL)"
Write-Host ""
Write-Host "To regenerate the synthetic sample (deterministic):"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "NOTE: PDB/ChEMBL/BindingDB each carry their own license terms -- read and"
Write-Host "respect them. Several covalent benchmarks require registration; this"
Write-Host "script prints links ONLY and never bypasses any access control."
