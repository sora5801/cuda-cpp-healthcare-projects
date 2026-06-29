# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 7.14 -- Self-Supervised Image Pretraining for Medical Imaging   (template skeleton)
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

Write-Host "[download_data] Project 7.14 -- Self-Supervised Image Pretraining for Medical Imaging"
Write-Host "[download_data] Target data dir: $DataDir"

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
Write-Host ""
Write-Host "TODO(impl): no full dataset wired up yet for this template skeleton."
Write-Host "  Catalog dataset notes:"
Write-Host "    ChestX-ray14 / NIH — 112k chest X-rays with 14 disease labels (https://nihcc.app.box.com/v/ChestXray-NIHCC) MIMIC-CXR — 227k X-rays for self-supervised pretraining (https://physionet.org/content/mimic-cxr/) RadImageNet — 1.35M radiology images across CT/MRI/US for SSL pretraining (verify URL) BraTS + TCIA — unlabelled MRI/CT volumes for 3D MAE pretraining (https://www.cancerimagingarchive.net/)"
Write-Host ""
Write-Host "  The committed tiny sample in data/sample/ is enough to run the demo."
Write-Host "  For a larger SYNTHETIC problem, run:"
Write-Host "    python scripts/make_synthetic.py --n 1048576"
Write-Host ""
Write-Host "  When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
