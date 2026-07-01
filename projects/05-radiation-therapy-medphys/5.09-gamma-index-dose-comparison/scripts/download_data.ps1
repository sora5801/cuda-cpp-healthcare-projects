# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.9 -- Gamma-Index Dose Comparison
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. Real gamma-index inputs are plan+measurement dose
# pairs that are patient-derived and NOT redistributable, so this script only
# prints guidance and defers to scripts/make_synthetic.py for an offline
# stand-in. There is nothing to download automatically.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.9 -- Gamma-Index Dose Comparison"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "No auto-download: gamma-index inputs are patient-derived plan+measurement"
Write-Host "dose pairs and are not redistributable. Obtain them from your own clinic:"
Write-Host ""
Write-Host "  * AAPM TG-218  -- methodology + example criteria for IMRT/VMAT QA"
Write-Host "                    (Med Phys 45(4), 2018)."
Write-Host "  * Plan+measurement DICOM-RTDOSE pairs -- from your TPS + QA system"
Write-Host "                    (film / EPID / diode array). Local access + ethics only."
Write-Host "  * IROC-Houston phantom datasets -- request through IROC."
Write-Host "  * Linac EPID measurement datasets -- from your machine's QA archive."
Write-Host ""
Write-Host "Respect every source license and patient-privacy rule. Do NOT commit"
Write-Host "patient-derived data to this public repo."
Write-Host ""
Write-Host "The committed SYNTHETIC sample in data/sample/dose_pair.txt is enough to"
Write-Host "run the demo. For a larger synthetic problem, run:"
Write-Host "    python scripts/make_synthetic.py --n 128"
