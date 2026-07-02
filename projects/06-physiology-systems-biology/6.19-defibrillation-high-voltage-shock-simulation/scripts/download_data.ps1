# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.19 : Defibrillation & High-Voltage Shock Simulation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL(s), and
# NEVER bypasses credentials/registration. This project is a REDUCED-SCOPE
# teaching model (a 1-D FitzHugh-Nagumo cable) that runs entirely on the tiny
# SYNTHETIC sample in data/sample/, so there is no dataset to download for the
# demo. This script exists to point you at the real research data + tools and to
# regenerate a synthetic problem on demand.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.19 -- Defibrillation & High-Voltage Shock Simulation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project needs NO download: the committed synthetic sample"
Write-Host "  data/sample/defib_sweep.txt  is sufficient to build and run the demo."
Write-Host ""
Write-Host "Real defibrillation research data + solvers (respect each license/registration):"
Write-Host "  - PhysioNet fibrillation/defibrillation recordings : https://physionet.org"
Write-Host "  - openCARP defibrillation tutorial cases           : https://opencarp.org"
Write-Host "  - Cardioid (LLNL) bidomain shock examples          : https://github.com/llnl/cardioid"
Write-Host "  - Chaste (bidomain + electrode BCs)                : https://github.com/Chaste/Chaste"
Write-Host "  - MonoAlg3D_C (GPU bidomain-capable)               : https://github.com/rsachetto/MonoAlg3D_C"
Write-Host ""
Write-Host "  PhysioNet requires accepting a data-use agreement; patient-specific ICD"
Write-Host "  datasets require institutional/IRB access. This script does NOT bypass"
Write-Host "  either -- register at the source and download manually."
Write-Host ""
Write-Host "To (re)generate the synthetic sample or a variant, run:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host "    python scripts/make_synthetic.py --biphasic 1 --shock-len 20"
