# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point at the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This teaching engine runs on a
# SYNTHETIC Lennard-Jones fluid (data/sample/lj_sample.txt) and needs no external
# download, so this script just (a) ensures the synthetic sample exists and (b)
# prints pointers to the real force fields a production engine would consume.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
$Sample  = Join-Path $DataDir "sample\lj_sample.txt"

Write-Host "[download_data] Project 1.1 -- Molecular Dynamics Engine"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""

# (a) Idempotent: regenerate the tiny synthetic sample only if it is missing.
if (Test-Path $Sample) {
    Write-Host "[download_data] Synthetic sample already present: $Sample"
} else {
    Write-Host "[download_data] Generating synthetic sample ..."
    python (Join-Path $PSScriptRoot "make_synthetic.py")
}

# (b) Pointers to the real force fields / trajectory libraries (study material).
Write-Host ""
Write-Host "This engine is a teaching model of the Lennard-Jones force field, so it"
Write-Host "runs entirely on the committed SYNTHETIC sample -- no download required."
Write-Host ""
Write-Host "Production biomolecular MD instead reads these (do NOT commit them here):"
Write-Host "  CHARMM36m force field  : https://mackerell.umaryland.edu/charmm_ff.shtml"
Write-Host "  AMBER ff19SB           : https://ambermd.org"
Write-Host "  GPCRmd trajectories    : https://gpcrmd.org"
Write-Host "  MoDEL protein library  : https://mmb.irbbarcelona.org/MoDEL/"
Write-Host ""
Write-Host "For a larger SYNTHETIC system (e.g. 512 atoms), run:"
Write-Host "    python scripts/make_synthetic.py --side 8"
