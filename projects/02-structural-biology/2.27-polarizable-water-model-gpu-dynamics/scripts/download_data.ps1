# ===========================================================================
# scripts/download_data.ps1  --  Reference-data pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.27 : Polarizable Water Model GPU Dynamics
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. The reference data for polarizable
# water is (a) tabular thermophysical properties and (b) force-field parameter
# files / trajectory archives -- none of which this teaching demo needs, because
# its committed synthetic cluster (data/sample/water_cluster.txt) is complete and
# self-contained. So this script PRINTS where the real data lives and defers to
# scripts/make_synthetic.py for larger offline inputs.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.27 -- Polarizable Water Model GPU Dynamics"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This demo needs NO download: data/sample/water_cluster.txt is a complete,"
Write-Host "self-contained SYNTHETIC cluster. The real-world reference data for"
Write-Host "polarizable water models is:"
Write-Host "  * NIST water thermophysical properties (density/dielectric vs T,P):"
Write-Host "      https://webbook.nist.gov/chemistry/fluid/"
Write-Host "  * TIP4P-2005 / SPC/E reference simulation data and force-field params"
Write-Host "    (water density anomaly, dielectric constant convergence benchmarks)."
Write-Host "  * MB-pol / AMOEBA polarizable parameters & code:"
Write-Host "      MBX        https://github.com/paesanilab/MBX"
Write-Host "      OpenMM     https://github.com/openmm/openmm"
Write-Host "      Tinker-HP  https://github.com/TinkerTools/tinker-hp"
Write-Host ""
Write-Host "For a larger SYNTHETIC cluster (e.g. 64 waters), run:"
Write-Host "    python scripts/make_synthetic.py --waters 64"
