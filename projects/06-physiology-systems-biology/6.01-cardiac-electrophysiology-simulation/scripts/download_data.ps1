# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.1 -- Cardiac Electrophysiology Simulation   (template skeleton)
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

Write-Host "[download_data] Project 6.1 -- Cardiac Electrophysiology Simulation"
Write-Host "[download_data] Target data dir: $DataDir"

# This project simulates from a parameter file, not a downloaded dataset, so the
# committed synthetic sample is fully self-contained. The catalog datasets below
# are the REAL-WORLD sources you would use to build validated, patient-specific
# cardiac models; none is needed to run this teaching demo.
Write-Host ""
Write-Host "  This project needs NO download: the committed synthetic sample in"
Write-Host "  data/sample/tissue_params.txt is enough to run the demo end-to-end."
Write-Host ""
Write-Host "  Real-world data sources (from the catalog) for validated cardiac EP:"
Write-Host "    * PhysioNet MIT-BIH & MIMIC-III Waveform -- ICU ECG/hemodynamics (https://physionet.org)"
Write-Host "    * CellML Physiome Repository -- curated ionic cell models (https://models.physiomeproject.org)"
Write-Host "    * UK Biobank Cardiac MRI -- cine CMR, access via application (https://www.ukbiobank.ac.uk)"
Write-Host "    * ACDC MICCAI Cardiac Challenge -- CMR with myocardium ground truth"
Write-Host "      (https://www.creatis.insa-lyon.fr/Challenge/acdc/)"
Write-Host ""
Write-Host "  These require registration/credentials; this script never bypasses that."
Write-Host "  For a bigger SYNTHETIC problem, run:"
Write-Host "    python scripts/make_synthetic.py --nx 128 --ny 128 --steps 1200"
Write-Host ""
Write-Host "  If you later wire a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
