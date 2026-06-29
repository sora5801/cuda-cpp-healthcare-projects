# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.3 -- Molecular Docking Engine   (template skeleton)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. If a
# dataset needs an account, this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.3 -- Molecular Docking Engine"
Write-Host "[download_data] Target data dir: $DataDir"

# This teaching project runs on SYNTHETIC input (scripts/make_synthetic.py). Real
# docking data is not auto-fetched: it requires receptor/ligand preparation and an
# AutoGrid map computation that are outside this didactic scope. We print pointers.
Write-Host ""
Write-Host "  This project's committed sample is SYNTHETIC (data/sample/) and runs the demo offline."
Write-Host "  No real dataset is auto-downloaded -- real docking needs receptor + ligand prep."
Write-Host ""
Write-Host "  Real datasets to study (respect each license):"
Write-Host "    DUD-E    102 targets, actives + decoys      https://dude.docking.org"
Write-Host "    ChEMBL   >2M bioactive compounds            https://www.ebi.ac.uk/chembl/"
Write-Host "    PDBbind  protein-ligand complexes + Kd/Ki   http://www.pdbbind.org.cn"
Write-Host "    CASF     scoring-function benchmark         http://www.pdbbind.org.cn/casf.php"
Write-Host ""
Write-Host "  To dock a real complex you would: prepare receptor+ligand to PDBQT (AutoDockTools/Meeko),"
Write-Host "  precompute energy maps with AutoGrid, then run AutoDock-GPU or Vina (see THEORY.md)."
Write-Host ""
Write-Host "  For a larger SYNTHETIC problem instead, run e.g.:"
Write-Host "    python scripts/make_synthetic.py --n-trans 15 --n-rot 6 --n-grid 32"
