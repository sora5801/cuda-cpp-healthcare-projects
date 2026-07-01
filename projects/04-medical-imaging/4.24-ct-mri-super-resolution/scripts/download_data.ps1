# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.24 : CT/MRI Super-Resolution
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real CT/MRI SR datasets require
# accounts and/or forbid redistribution, so this script prints instructions +
# links ONLY and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.24 -- CT/MRI Super-Resolution"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample data/sample/phantom_hr.txt (SYNTHETIC) already lets"
Write-Host "the demo run offline. Real datasets below require registration -- this"
Write-Host "script does NOT bypass it; it only prints where to get them."
Write-Host ""
Write-Host "Real CT/MRI super-resolution datasets (register + accept terms yourself):"
Write-Host "  * HCP 7T/3T paired brain MRI : https://db.humanconnectome.org/"
Write-Host "  * fastMRI                     : https://fastmri.med.nyu.edu/"
Write-Host "  * IXI brain MRI (CC BY-SA 3.0): https://brain-development.org/ixi-dataset/"
Write-Host "  * MSD CT/MRI tasks           : http://medicaldecathlon.com/"
Write-Host ""
Write-Host "To turn a real slice into this project's input format:"
Write-Host "  1) load one axial slice, normalize intensities to [0,1];"
Write-Host "  2) crop/pad both dims to a multiple of SR_SCALE (=2);"
Write-Host "  3) write '<w> <h>' then w*h floats row-major (see data/README.md)."
Write-Host ""
Write-Host "For a larger SYNTHETIC phantom instead, run:"
Write-Host "  python scripts/make_synthetic.py --size 128"
