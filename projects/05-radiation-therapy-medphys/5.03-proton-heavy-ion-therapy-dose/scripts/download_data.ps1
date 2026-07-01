# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.3 : Proton & Heavy-Ion Therapy Dose
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL(s) and
# NEVER bypasses credentials/registration. The real proton-therapy datasets are
# either patient-derived (need an institutional/DUA agreement) or ship inside a
# Monte-Carlo toolkit (TOPAS/GATE benchmark beams). This project does NOT need
# any of them to run: the committed synthetic plan drives the demo. This script
# therefore prints where to get real data and defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.3 -- Proton & Heavy-Ion Therapy Dose"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project runs entirely on the committed SYNTHETIC plan in"
Write-Host "data/sample/proton_plan_sample.txt -- no download is required."
Write-Host ""
Write-Host "Real / reference proton data you may study (respect every license):"
Write-Host "  * TOPAS / GATE benchmark proton beams -- integral depth-dose and"
Write-Host "    lateral profiles used to COMMISSION analytic engines. Obtain via the"
Write-Host "    TOPAS (https://www.topasmc.org) or GATE (http://www.opengatecollaboration.org)"
Write-Host "    distributions; these are Monte-Carlo outputs, not patient data."
Write-Host "  * POPI-model 4D CT for treatment planning:"
Write-Host "    https://www.creatis.insa-lyon.fr/rio/popi-model"
Write-Host "  * TCIA proton treatment-response collections (registration/DUA required):"
Write-Host "    https://www.cancerimagingarchive.net"
Write-Host ""
Write-Host "Patient-derived / credentialed sets: this script will NOT bypass any"
Write-Host "registration or data-use agreement. Follow the provider's process."
Write-Host ""
Write-Host "To make a LARGER synthetic plan (e.g. a spread-out Bragg peak), run:"
Write-Host "    python scripts/make_synthetic.py --ranges 8 9 10 11 12"
