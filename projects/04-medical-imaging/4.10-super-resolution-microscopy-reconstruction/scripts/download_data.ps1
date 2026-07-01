# ===========================================================================
# scripts/download_data.ps1  --  Real SMLM-data pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.10 : Super-Resolution Microscopy Reconstruction  (SMLM). Nothing to
# fetch automatically -- real STORM/PALM movies are large multi-GB TIFF stacks
# and several need registration, so per CLAUDE.md §8 this script only prints the
# sources and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.10 -- Super-Resolution Microscopy Reconstruction"
Write-Host ""
Write-Host "Real SMLM movies are multi-frame TIFF/OME-TIFF stacks. To use one, export"
Write-Host "each frame's pixels into the text format in data/README.md:"
Write-Host "  header:  'F H W background threshold'   then F*H*W floats, row-major."
Write-Host ""
Write-Host "  EPFL SMLM Challenge : https://srm.epfl.ch/srm/dataset/challenge-2016/"
Write-Host "                        (synthetic + real STORM/PALM frames, with ground truth)"
Write-Host "  BioImage Archive    : https://www.ebi.ac.uk/biostudies/bioimages"
Write-Host "                        (public SMLM collections)"
Write-Host "  OME-TIFF standard   : https://www.openmicroscopy.org/ome-files/"
Write-Host ""
Write-Host "Tools to read/convert TIFF stacks: tifffile (Python), Fiji/ImageJ, ThunderSTORM."
Write-Host ""
Write-Host "No download needed for the demo -- the committed data/sample/smlm_stack.txt"
Write-Host "runs offline. For a bigger SYNTHETIC movie:"
Write-Host "    python scripts/make_synthetic.py --frames 200 --width 64 --height 64"
Write-Host ""
Write-Host "Target data dir: $DataDir"
