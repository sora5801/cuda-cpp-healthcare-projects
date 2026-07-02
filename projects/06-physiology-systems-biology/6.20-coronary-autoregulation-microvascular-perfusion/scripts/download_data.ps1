# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.20 -- Coronary Autoregulation & Microvascular Perfusion
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# guidance, and NEVER bypasses credentials/registration. Every real coronary
# dataset for this project is either credentialed (UK Biobank, some PhysioNet,
# MICCAI) or a geometry repository you convert by hand -- so this script prints
# instructions and links ONLY, and defers to make_synthetic.py for an offline
# stand-in. The committed tiny synthetic sample already runs the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.20 -- Coronary Autoregulation & Microvascular Perfusion"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a TINY SYNTHETIC sample (data/sample/coronary_network.txt)"
Write-Host "that runs the demo offline. There is no automatic bulk download because every"
Write-Host "real coronary dataset is credentialed or must be converted from a geometry model."
Write-Host ""
Write-Host "Real datasets (obtain manually, respecting each license):"
Write-Host "  * UK Biobank coronary CTA (subset)  https://www.ukbiobank.ac.uk"
Write-Host "      -> requires an APPROVED application; redistribution forbidden."
Write-Host "  * PhysioNet coronary pressure/flow   https://physionet.org"
Write-Host "      -> some sets need credentialing + a data use agreement."
Write-Host "  * Vascular Model Repository          http://www.vascularmodel.com"
Write-Host "      -> open cardiovascular geometries; extract centerlines + radii."
Write-Host "  * MICCAI coronary artery tracking     https://grand-challenge.org"
Write-Host "      -> challenge registration required."
Write-Host ""
Write-Host "To build a REAL network for this solver:"
Write-Host "  1) take a centerline model (nodes = branch points, edges = segments),"
Write-Host "  2) write it into data/sample/coronary_network.txt in the documented format"
Write-Host "     (see data/README.md), pinning the inlet and venous outlets,"
Write-Host "  3) run the demo / exe on that file path."
Write-Host ""
Write-Host "For a larger SYNTHETIC network right now, regenerate the sample:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "[download_data] No credentialed data was fetched or bypassed (by design)."
