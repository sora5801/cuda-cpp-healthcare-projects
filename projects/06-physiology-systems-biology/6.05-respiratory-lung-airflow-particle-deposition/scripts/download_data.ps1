# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs
# and NEVER bypasses credentials/registration. Real airway geometries come from
# patient CT archives that require an account and a data-use agreement, so this
# script only PRINTS how to obtain them and points at make_synthetic.py for an
# offline stand-in. The committed tiny sample already runs the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.5 -- Respiratory / Lung Airflow & Particle Deposition"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a self-contained SYNTHETIC parameter file"
Write-Host "(data/sample/lung_params.txt). No download is required to run the demo."
Write-Host ""
Write-Host "To drive the model from a REAL patient airway geometry, obtain a lung CT"
Write-Host "volume, segment the airway tree, and fit per-generation radii/lengths."
Write-Host "Public sources (each needs registration / a data-use agreement -- respect it):"
Write-Host "  * LIDC-IDRI lung CT (1010 cases), TCIA:"
Write-Host "      https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
Write-Host "  * COPDGene lung CT (10000 subjects): https://www.copdgene.org"
Write-Host "  * SPIROMICS bronchial CT:            https://www.spiromics.org"
Write-Host "  * PhysioNet respiratory waveforms:   https://physionet.org"
Write-Host "  Airway segmentation tooling: 3D Slicer + SlicerMorph"
Write-Host "      https://github.com/SlicerMorph/SlicerMorph"
Write-Host ""
Write-Host "For a larger SYNTHETIC experiment (no download), regenerate the sample:"
Write-Host "    python scripts/make_synthetic.py --d_p 1.0 --n 1000000"
Write-Host ""
Write-Host "[download_data] Done (informational only; nothing was downloaded)."
