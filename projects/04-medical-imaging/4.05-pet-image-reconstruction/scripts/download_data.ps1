# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Real PET reconstruction data is large
# and often gated, so this script prints pointers only and defers to
# scripts/make_synthetic.py for an offline stand-in. The committed tiny sample in
# data/sample/ is already enough to run the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.5 -- PET Image Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/sinogram_sample.txt) runs the demo offline."
Write-Host "For real (non-clinical) PET sinograms, the cleanest sources are:"
Write-Host ""
Write-Host "  * PETRIC challenge data (Interfile/STIR sinograms):"
Write-Host "      https://github.com/SyneRBI/PETRIC"
Write-Host "  * SIRF-Exercises (openly usable phantom PET/MR data + notebooks):"
Write-Host "      https://github.com/SyneRBI/SIRF-Exercises"
Write-Host "  * Siemens mMR phantom datasets via STIR/SIRF:"
Write-Host "      https://github.com/SyneRBI/SIRF"
Write-Host "  * TCIA PET collections (mostly reconstructed volumes, license varies):"
Write-Host "      https://www.cancerimagingarchive.net/"
Write-Host "  * OpenNEURO PET datasets:"
Write-Host "      https://openneuro.org/"
Write-Host ""
Write-Host "Notes:"
Write-Host "  - Respect each collection's license and de-identification terms."
Write-Host "  - Reconstruction needs the RAW sinogram or list-mode data, not just the"
Write-Host "    reconstructed image; PETRIC/SIRF are the most direct for that."
Write-Host "  - This project's loader expects the simple text format in data/README.md."
Write-Host "    A converter from Interfile is left as an exercise (see README)."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem right now, run:"
Write-Host "    python scripts/make_synthetic.py --N 64 --K 60 --D 91 --iters 40"
