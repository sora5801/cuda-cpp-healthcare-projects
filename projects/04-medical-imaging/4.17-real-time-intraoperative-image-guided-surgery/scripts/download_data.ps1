# ===========================================================================
# scripts/download_data.ps1  --  Fetch/point-to the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The real IGS datasets below are
# video/volume corpora behind registration or challenge sign-ups, so this
# script only prints instructions + links. The committed synthetic sample in
# data/sample/ is all the demo needs; make_synthetic.py can scale it up.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.17 -- Real-Time Intraoperative / Image-Guided Surgery"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs on a SYNTHETIC point-cloud pair (data/sample/surface_pair.txt)."
Write-Host "No download is required to build, run, or verify the demo."
Write-Host ""
Write-Host "For a LARGER synthetic problem (deterministic), run:"
Write-Host "    python scripts/make_synthetic.py --grid 40 --noise 0.3"
Write-Host ""
Write-Host "Real image-guided-surgery datasets (registration / credentials required --"
Write-Host "this script does NOT bypass any login; it only points you to the source):"
Write-Host "  * Cholec80 laparoscopic videos : https://camma.u-strasbg.fr/datasets"
Write-Host "  * ReMIND2Reg 2025 (brain)      : https://arxiv.org/abs/2508.09649"
Write-Host "  * EndoVis (MICCAI) challenges  : https://endovis.grand-challenge.org/"
Write-Host "  * SurgT tool-tracking benchmark"
Write-Host ""
Write-Host "To use a real surface: sample 3-D points from the two surfaces and write"
Write-Host "them in data/sample/surface_pair.txt's format (see data/README.md)."
