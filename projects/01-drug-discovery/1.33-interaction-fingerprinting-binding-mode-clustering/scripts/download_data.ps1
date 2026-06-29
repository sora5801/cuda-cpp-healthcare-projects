# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 1.33 -- Interaction Fingerprinting & Binding-Mode Clustering
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real interaction fingerprints come from
# docking poses or MD frames of actual complexes; turning those into the bit-
# vectors this project clusters needs a chemistry toolkit (ProLIF/ODDT), which is
# out of scope. So this script PRINTS where to get the structures + how to derive
# IFPs, and defers to make_synthetic.py for an offline, self-contained stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.33 -- Interaction Fingerprinting & Binding-Mode Clustering"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project clusters INTERACTION FINGERPRINTS (bit-vectors). To build them"
Write-Host "from real structures you need (a) protein-ligand poses and (b) an interaction"
Write-Host "detector. Public sources for the structures:"
Write-Host "  * PDBbind   - complexes + affinities : http://www.pdbbind.org.cn"
Write-Host "  * KLIFS     - kinase IFP features    : https://klifs.net"
Write-Host "  * ChEMBL    - bioactivity + structs  : https://www.ebi.ac.uk/chembl/"
Write-Host "  * BindingDB - measured binding data  : https://www.bindingdb.org"
Write-Host ""
Write-Host "Derive IFPs with a toolkit, then emit rows matching data/README.md:"
Write-Host "  * ProLIF : https://github.com/chemosim-lab/ProLIF  (IFPs from MD/poses)"
Write-Host "  * ODDT   : https://github.com/oddt/oddt           (open drug-discovery toolkit)"
Write-Host ""
Write-Host "No credentials are bypassed here. The committed data/sample/ifp_sample.txt is"
Write-Host "enough to run the demo offline. For a larger SYNTHETIC problem:"
Write-Host "    python scripts/make_synthetic.py --per-mode 500"
