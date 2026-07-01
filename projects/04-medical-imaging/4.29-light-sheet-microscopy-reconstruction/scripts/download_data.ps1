# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.29 : Light-Sheet Microscopy Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. If a
# dataset needs an account, this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.29 -- Light-Sheet Microscopy Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"

# This teaching project runs on a SYNTHETIC sample (see scripts/make_synthetic.py
# and data/README.md). Real light-sheet datasets are terabyte-scale volumes in
# formats (TIFF/HDF5/N5/Zarr) that this tiny text-based loader does not read, and
# several require a data-use agreement. So this script prints the public sources
# and defers to the synthetic generator rather than downloading TBs.
Write-Host ""
Write-Host "  This project ships a SYNTHETIC sample; no download is required to run the demo."
Write-Host "  Regenerate or enlarge it with:"
Write-Host "    python scripts/make_synthetic.py                 # default 32x32 plane"
Write-Host "    python scripts/make_synthetic.py --h 64 --w 64   # a larger synthetic plane"
Write-Host ""
Write-Host "  Real, publicly-documented LSFM data sources (study these; formats differ"
Write-Host "  from this loader and some need registration -- respect every license):"
Write-Host "    - OpenOrganelle (Janelia):        https://openorganelle.janelia.org/"
Write-Host "    - EMBL LSFM public datasets:      https://www.embl.org/"
Write-Host "    - BioImage Archive (EBI) LSFM:    https://www.ebi.ac.uk/biostudies/bioimages"
Write-Host "    - Zebrafish SPIM atlas data:      from the Nature Methods SPIM papers"
Write-Host ""
Write-Host "  For a credentialed set, register at the source FIRST; this script never"
Write-Host "  bypasses authentication (CLAUDE.md section 8)."
