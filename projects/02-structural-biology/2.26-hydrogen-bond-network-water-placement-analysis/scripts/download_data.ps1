# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.26 -- Hydrogen Bond Network & Water Placement Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. GIST on
# real structures needs a full MD trajectory + a real GIST tool, which is outside
# this teaching project's scope -- so this script prints the authoritative
# pointers and defers to scripts/make_synthetic.py for the offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
$Sample  = Join-Path $DataDir "sample\water_sample.txt"

Write-Host "[download_data] Project 2.26 -- Hydrogen Bond Network & Water Placement Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""

# Idempotent: the committed synthetic sample is all the demo needs. If it is
# somehow missing, regenerate it deterministically rather than downloading.
if (Test-Path $Sample) {
    Write-Host "[download_data] Synthetic sample already present: $Sample"
} else {
    Write-Host "[download_data] Synthetic sample missing; regenerating it ..."
    python (Join-Path $PSScriptRoot "make_synthetic.py")
}

Write-Host ""
Write-Host "[download_data] This project ships a SYNTHETIC sample (see data/README.md)."
Write-Host "  No real dataset is required to build, run, or study the demo."
Write-Host ""
Write-Host "  To study GIST on REAL structures, use these public sources (respect each license;"
Write-Host "  do NOT commit redistributed data; nothing here is for clinical use):"
Write-Host "    * SAMPL water-placement challenges : https://github.com/samplchallenges/SAMPL"
Write-Host "    * Explicit-solvent PDB structures  : https://www.rcsb.org"
Write-Host "    * GIST reference systems           : T4 lysozyme L99A, FKBP12 (GIST literature)"
Write-Host "    * WaterMap validation sets         : Schrodinger (commercial; verify URL)"
Write-Host ""
Write-Host "  Producing a real GIST input requires an MD trajectory (AMBER/GROMACS/OpenMM) and"
Write-Host "  a GIST tool (cpptraj 'gist' or GISTPP). When wiring such a fetch, follow the"
Write-Host "  idempotent pattern: (1) skip if the file exists with the right SHA256,"
Write-Host "  (2) print source URL + expected size + checksum, (3) for credentialed sets print"
Write-Host "  registration instructions ONLY -- never bypass them."
Write-Host ""
Write-Host "  For a LARGER synthetic problem instead:"
Write-Host "    python scripts/make_synthetic.py --frames 5000"
