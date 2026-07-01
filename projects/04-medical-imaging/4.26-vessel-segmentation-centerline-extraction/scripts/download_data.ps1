# ===========================================================================
# scripts/download_data.ps1  --  Real vessel-imaging dataset pointers (Windows)
# ---------------------------------------------------------------------------
# Project 4.26 : Vessel Segmentation & Centerline Extraction
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs,
# and NEVER bypasses credentials/registration. The real coronary/vessel datasets
# require account registration or a challenge sign-up, so this script only prints
# instructions + links; scripts/make_synthetic.py provides the offline stand-in
# the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.26 -- Vessel Segmentation & Centerline Extraction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "There is nothing to auto-download: the committed synthetic volume in"
Write-Host "data/sample/vessel_volume.txt is enough to run the demo, and the real"
Write-Host "datasets below need registration (do NOT try to bypass it)."
Write-Host ""
Write-Host "REAL 3-D vessel datasets (register on the site, then export to NIfTI):"
Write-Host "  ASOCA (coronary CTA challenge) : https://asoca.grand-challenge.org/"
Write-Host "  ImageCAS (1000 coronary CTAs)  : https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT"
Write-Host "  3D-IRCADb-01 (abdominal/liver) : https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/"
Write-Host ""
Write-Host "To run this teaching filter on real data you must first convert a NIfTI/"
Write-Host "DICOM volume into this project's plain-text format (see data/README.md)."
Write-Host "A tiny converter is left as an exercise (README 'Exercises')."
Write-Host ""
Write-Host "Bigger SYNTHETIC volume (no download, fully offline):"
Write-Host "  python scripts/make_synthetic.py --nx 128 --ny 96 --nz 96 --radius 4"
