# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. fastMRI requires a
# signed data-use agreement, so this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline, synthetic stand-in that the
# demo actually runs on.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.4 : Deep-Learning MRI/CT Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This demo ships a tiny SYNTHETIC acquisition (data/sample/mri_scan_sample.txt);"
Write-Host "no download is required to build or run it."
Write-Host ""
Write-Host "To study REAL learned reconstruction, get raw multi-coil k-space from fastMRI:"
Write-Host "  1) Register + accept the data-use agreement at:  https://fastmri.med.nyu.edu/"
Write-Host "     (NYU Langone; free for research. We do NOT and CANNOT bypass this step.)"
Write-Host "  2) You receive time-limited download links by email. The knee/brain single- and"
Write-Host "     multi-coil sets are large (tens to hundreds of GB) and are .h5 (HDF5) files."
Write-Host "  3) fastMRI+ radiologist annotations: https://github.com/StanfordMIMI/fastMRI_plus"
Write-Host "  4) For learned CT instead, see the 2016 AAPM Low-Dose CT Grand Challenge."
Write-Host ""
Write-Host "Reading .h5 k-space + training an E2E-VarNet is a PyTorch task; this C++ demo"
Write-Host "teaches the unrolled-reconstruction STRUCTURE on a small synthetic phantom instead."
Write-Host ""
Write-Host "Regenerate / resize the synthetic sample with:"
Write-Host "    python scripts/make_synthetic.py --ny 32 --nx 32"
