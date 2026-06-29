# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.29 : Kinase Selectivity Panel Scoring
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The real
# selectivity datasets (KLIFS, KINOMEscan, ChEMBL, DTC) require registration or
# have redistribution limits, so this script only PRINTS instructions + links and
# defers to scripts/make_synthetic.py for the committed offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.29 -- Kinase Selectivity Panel Scoring"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC committed sample (data/sample/kinase_panel_sample.txt),"
Write-Host "which is all the demo needs. The real selectivity datasets are external:"
Write-Host ""
Write-Host "  * KLIFS (kinase-ligand interaction fingerprints) -> https://klifs.net"
Write-Host "      Use the web API or the 'kissim' package to build per-kinase IFPs."
Write-Host "      Free for academic use; check site terms before any redistribution."
Write-Host "  * KINOMEscan / Kd selectivity (Karaman et al. 2008, DiscoverX/Eurofins)"
Write-Host "      Published supplements or commercial panels; provider-specific license."
Write-Host "  * ChEMBL kinase bioactivity -> https://www.ebi.ac.uk/chembl/  (CC BY-SA 3.0)"
Write-Host "  * Drug-Target Commons (DTC) -> https://drugtargetcommons.fimm.fi  (CC BY 4.0)"
Write-Host ""
Write-Host "These require registration and/or forbid wholesale redistribution, so they"
Write-Host "are NOT downloaded automatically and NOT committed (CLAUDE.md section 8)."
Write-Host ""
Write-Host "For a larger SYNTHETIC panel, edit the PANEL list in scripts/make_synthetic.py"
Write-Host "and re-run:  python scripts/make_synthetic.py"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for credentialed sets, print registration instructions ONLY"
