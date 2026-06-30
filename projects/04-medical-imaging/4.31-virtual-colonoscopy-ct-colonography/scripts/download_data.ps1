# ===========================================================================
# scripts/download_data.ps1  --  Guidance for the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 4.31 : Virtual Colonoscopy & CT Colonography
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# guidance, and NEVER bypasses credentials/registration. Real CT colonography
# volumes require accepting data-use terms, so this script only PRINTS where to
# get them and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.31 -- Virtual Colonoscopy & CT Colonography"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/colon_volume_sample.txt) is SYNTHETIC"
Write-Host "and is all the demo needs -- it runs fully offline."
Write-Host ""
Write-Host "To work with REAL CT colonography volumes (not auto-downloaded here):"
Write-Host ""
Write-Host "  TCIA 'CT Colonography' collection (supine/prone DICOM):"
Write-Host "    https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography"
Write-Host "    - Accept the TCIA Data Usage Policy / per-collection terms."
Write-Host "    - Download with the NBIA Data Retriever (a manifest-based tool)."
Write-Host "    - Then segment the air-filled lumen and resample to a dense grid"
Write-Host "      in the loader's text format (see data/README.md)."
Write-Host ""
Write-Host "  Other sources (may require registration): MICCAI 2018 colon challenge,"
Write-Host "  ACR Lung-RADS CT, NLST CT colonography subsets."
Write-Host ""
Write-Host "For a larger SYNTHETIC volume instead, run:"
Write-Host "    python scripts/make_synthetic.py --nx 96 --ny 96 --nz 128 --width 256 --height 256"
Write-Host ""
Write-Host "This script intentionally downloads nothing: the public CTC sets are"
Write-Host "credentialed/terms-gated and must be fetched by you, per their license."
