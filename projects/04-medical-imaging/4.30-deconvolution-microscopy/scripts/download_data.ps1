# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.30 : Deconvolution Microscopy
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Public microscopy benchmark sets are
# distributed as multi-megabyte TIFF stacks under their own licenses; rather than
# silently redistribute them, this script points you at the canonical sources and
# defers to scripts/make_synthetic.py for the offline, runnable stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.30 -- Deconvolution Microscopy"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a TINY SYNTHETIC blurred image in data/sample/ so the"
Write-Host "demo runs fully offline. No download is required to build, run, or learn."
Write-Host ""
Write-Host "To study REAL fluorescence-microscopy deconvolution benchmarks, visit:"
Write-Host "  * EPFL Biomedical Imaging Group deconvolution benchmark + measured PSFs:"
Write-Host "      https://bigwww.epfl.ch/deconvolution/"
Write-Host "  * BioImage Archive fluorescence datasets (raw + restored stacks):"
Write-Host "      https://www.ebi.ac.uk/biostudies/bioimages"
Write-Host "  * ImageJ/Fiji sample images (e.g. the classic confocal stacks):"
Write-Host "      https://imagej.net/"
Write-Host ""
Write-Host "Each is governed by its own license -- respect it; we do not redistribute."
Write-Host "Convert a downloaded 2-D slice to this project's text format:"
Write-Host "  header line '<w> <h>' then h rows of w space-separated intensities,"
Write-Host "  matching load_image() in src/reference_cpu.cpp (see data/README.md)."
Write-Host ""
Write-Host "For a larger SYNTHETIC image (no download), run:"
Write-Host "  python scripts/make_synthetic.py --w 128 --h 128"
