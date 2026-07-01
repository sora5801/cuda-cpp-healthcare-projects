# ===========================================================================
# scripts/download_data.ps1  --  Real-dataset pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. This project GENERATES its own flow from the
# parameters in data/sample/channel_params.txt, so there is nothing to download
# for the demo. The real, research-grade inputs are credential-gated; this
# script prints instructions and links ONLY and defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.3 -- Hemodynamics / Blood-Flow CFD"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Nothing to download: the solver makes its own channel flow from the"
Write-Host "synthetic parameters in data/sample/channel_params.txt."
Write-Host ""
Write-Host "For REAL patient-specific hemodynamics (image -> mesh -> CFD), the"
Write-Host "catalog datasets are credential-gated or license-restricted; obtain"
Write-Host "them yourself under their terms (this script will not bypass logins):"
Write-Host "  Vascular Model Repository (geometries) : http://www.vascularmodel.com"
Write-Host "  PhysioNet MIMIC-III waveforms          : https://physionet.org/content/mimiciii/1.4/"
Write-Host "  Zenodo Cardiac Mechanics Emulation     : https://zenodo.org/records/7075055"
Write-Host "  UK Biobank aortic 4D-flow MRI          : https://www.ukbiobank.ac.uk"
Write-Host ""
Write-Host "Full image-to-simulation pipeline (out of scope for this teaching project):"
Write-Host "  SimVascular / svFSI : https://github.com/SimVascular/svFSI"
Write-Host "  OpenFOAM            : https://github.com/OpenFOAM/OpenFOAM-dev"
Write-Host ""
Write-Host "Bigger / non-Newtonian SYNTHETIC problem:"
Write-Host "  python scripts/make_synthetic.py --nx 128 --ny 65 --steps 20000"
Write-Host "  python scripts/make_synthetic.py --nu-inf 0.03   # blood shear thinning"
