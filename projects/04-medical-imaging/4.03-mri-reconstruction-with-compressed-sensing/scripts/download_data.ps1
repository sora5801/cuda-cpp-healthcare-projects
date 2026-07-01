# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.3 : MRI Reconstruction with Compressed Sensing
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Every real raw-k-space MRI dataset for
# this project sits behind a data-use agreement, so this script only PRINTS the
# registration instructions and links, and points at make_synthetic.py for an
# offline stand-in. The committed data/sample/ already lets the demo run offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.3 -- MRI Reconstruction with Compressed Sensing"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "All real raw-k-space MRI datasets require a data-use agreement and CANNOT be"
Write-Host "auto-downloaded. Register with the provider, then export one slice's k-space"
Write-Host "into the text layout documented in data/README.md."
Write-Host ""
Write-Host "  fastMRI (NYU/Meta) -- knee + brain raw k-space (data-use agreement required):"
Write-Host "     https://fastmri.med.nyu.edu/"
Write-Host "     https://github.com/facebookresearch/fastMRI"
Write-Host ""
Write-Host "  Calgary-Campinas-359 -- multi-channel brain MRI k-space:"
Write-Host "     https://sites.google.com/view/calgary-campinas-dataset/"
Write-Host ""
Write-Host "  SKM-TEA (Stanford knee MRI):"
Write-Host "     https://github.com/StanfordMIMI/skm-tea"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample in data/sample/ is enough to run the demo."
Write-Host "For a larger synthetic problem, run:"
Write-Host "    python scripts/make_synthetic.py --n 64 --keep 0.30 --iters 80"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY (never bypass)"
