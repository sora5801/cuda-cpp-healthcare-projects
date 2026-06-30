# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.18 : NMR Structure Refinement
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. NMR restraint data is assembled
# per-protein-entry (no single archive), so this script prints the canonical
# sources and defers to scripts/make_synthetic.py for the offline demo input.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.18 -- NMR Structure Refinement"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC committed sample (data/sample/restraints.txt),"
Write-Host "which is all the demo needs. There is no single 'full dataset' to download:"
Write-Host "real NMR restraints are assembled per protein entry from these sources:"
Write-Host ""
Write-Host "  * BMRB  (https://bmrb.io)        - assigned shifts + restraint lists"
Write-Host "  * PDB   (https://www.rcsb.org)   - deposited NMR model ensembles + restraints"
Write-Host "  * RECOORD                        - uniformly recalculated NMR structures"
Write-Host "  * CASD-NMR                       - blind structure-determination benchmarks"
Write-Host ""
Write-Host "  None of the above is fetched automatically; visit the site for the entry"
Write-Host "  you want and respect its terms of use. To experiment at larger scale with"
Write-Host "  no download, regenerate a bigger SYNTHETIC problem:"
Write-Host "    python scripts/make_synthetic.py --n-beads 24 --replicas 4096"
Write-Host ""
Write-Host "  If you later wire a real fetch, follow the idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
