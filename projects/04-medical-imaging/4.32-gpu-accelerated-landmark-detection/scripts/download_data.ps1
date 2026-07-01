# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.32 : GPU-Accelerated Landmark Detection
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The real landmark
# datasets require registration (challenge accounts / data-use agreements), so
# this script only prints where to get them and how they map onto our loader,
# and defers to scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.32 -- GPU-Accelerated Landmark Detection"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project decodes landmark HEATMAPS (a network's output tensors)."
Write-Host "Real annotated landmark datasets require registration / a data-use"
Write-Host "agreement, so we do NOT auto-download them. Sources (register first):"
Write-Host "  * VerSe vertebral challenge  https://github.com/anjany/verse"
Write-Host "        374 CT scans, 26 vertebral landmarks each."
Write-Host "  * RSNA Vertebral Fracture Detection"
Write-Host "        https://rsna-vertebral-labeling-level-detection.grand-challenge.org/"
Write-Host "  * CephaloNet cephalometric landmark dataset (2D)."
Write-Host "  * MICCAI 2015 prostate challenge landmark dataset."
Write-Host ""
Write-Host "To turn a real volume + a network's prediction into our input format,"
Write-Host "export each landmark's heatmap tensor [Z,Y,X] to the layout documented"
Write-Host "in data/README.md (nx ny nz L, then per-landmark: cx cy cz + voxels)."
Write-Host ""
Write-Host "The committed tiny sample in data/sample/ runs the demo offline. For a"
Write-Host "larger SYNTHETIC problem (no registration needed), run:"
Write-Host "    python scripts/make_synthetic.py --nx 64 --ny 64 --nz 64 --landmarks 26"
