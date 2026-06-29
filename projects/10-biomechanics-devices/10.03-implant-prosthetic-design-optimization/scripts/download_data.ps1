# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 10.3 -- Implant & Prosthetic Design Optimization   (template skeleton)
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

Write-Host "[download_data] Project 10.3 -- Implant & Prosthetic Design Optimization"
Write-Host "[download_data] Target data dir: $DataDir"

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
Write-Host ""
Write-Host "TODO(impl): no full dataset wired up yet for this template skeleton."
Write-Host "  Catalog dataset notes:"
Write-Host "    OrthoLoad Implant Loading Database — in vivo hip/knee/spine implant force telemetry (https://orthoload.com/); MICCAI 2023 VerSe Challenge — vertebral shape dataset for spinal implant design (https://verse-challenge.github.io/); Hip Implant Topology Dataset — validated micro-FE lattice endoprostheses (see https://www.nature.com/articles/s41598-024-56327-4); FDA Orthopaedic Simulator Database — standardized fatigue loading profiles (verify URL via FDA.gov)."
Write-Host ""
Write-Host "  The committed tiny sample in data/sample/ is enough to run the demo."
Write-Host "  For a larger SYNTHETIC problem, run:"
Write-Host "    python scripts/make_synthetic.py --n 1048576"
Write-Host ""
Write-Host "  When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
