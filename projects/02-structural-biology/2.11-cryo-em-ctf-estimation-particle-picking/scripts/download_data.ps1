# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. Real cryo-EM
# micrographs live in EMPIAR and are large (tens of GB per dataset) and in
# binary MRC format; converting them to this project's tiny text format is out of
# scope for a teaching demo. So this script PRINTS guidance and defers to
# scripts/make_synthetic.py for an offline, verifiable stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.11 -- Cryo-EM CTF Estimation & Particle Picking"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real cryo-EM micrograph datasets (free, but large; MRC binary format):"
Write-Host "  * EMPIAR archive home : https://www.ebi.ac.uk/empiar/"
Write-Host "  * EMPIAR-10025 (beta-galactosidase) : classic CTF/processing tutorial set"
Write-Host "  * EMPIAR-10064 (T20S proteasome)    : RELION tutorial micrographs"
Write-Host "  * RELION tutorial data : https://relion.readthedocs.io"
Write-Host ""
Write-Host "These are tens of GB and need an MRC reader (e.g. mrcfile in Python) to"
Write-Host "convert a micrograph into this project's text format:"
Write-Host "    line 1:  n pixel_size lambda cs amp_contrast true_dz"
Write-Host "    body  :  n*n floats (row-major)"
Write-Host "(set true_dz = -1 for real data, where the defocus is unknown.)"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample in data/sample/ is enough to run the"
Write-Host "demo offline. For a larger synthetic problem, run:"
Write-Host "    python scripts/make_synthetic.py --n 256 --defocus 12000"
