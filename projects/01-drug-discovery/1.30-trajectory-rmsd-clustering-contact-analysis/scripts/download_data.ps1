# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URLs +
# how to fetch, and NEVER bypasses credentials/registration. The demo runs fully
# offline on the committed synthetic sample, so this script only prints guidance
# and pointers to the real molecular-dynamics trajectory archives.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.30 -- Trajectory RMSD, Clustering & Contact Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/trajectory_sample.txt) is SYNTHETIC and is"
Write-Host "all the demo needs -- no download is required to build, run, or verify."
Write-Host ""
Write-Host "Real molecular-dynamics trajectories to analyze with the same pipeline"
Write-Host "(you will need to adapt the loader to the real file format + atom count):"
Write-Host "  * MDCATH  (curated all-atom trajectories):"
Write-Host "        https://huggingface.co/datasets/compsciencelab/mdcath"
Write-Host "  * GPCRmd  (GPCR molecular-dynamics database):  https://gpcrmd.org"
Write-Host "  * MDDB    (molecular-dynamics database):       https://www.mddbr.eu"
Write-Host "  * PDB trajectory depositions (RCSB / PDB-Dev): https://www.rcsb.org"
Write-Host ""
Write-Host "These archives may require account registration and carry their own licenses."
Write-Host "Respect every license; this script does NOT attempt to bypass any login."
Write-Host ""
Write-Host "For a larger SYNTHETIC trajectory (no download, fully offline), run:"
Write-Host "    python scripts/make_synthetic.py --frames 100000"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print the source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
