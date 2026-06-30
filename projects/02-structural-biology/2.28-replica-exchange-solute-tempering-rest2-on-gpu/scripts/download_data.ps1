# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This teaching project runs entirely
# on the committed SYNTHETIC sample (data/sample/rest2_config.txt); the datasets
# below are where a *real* REST2 study gets its validation data, so this script
# only prints instructions + links and defers to scripts/make_synthetic.py for
# the offline run.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.28 -- Replica Exchange Solute Tempering (REST2) on GPU"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project's demo needs NO download: it runs on the committed synthetic"
Write-Host "sample data/sample/rest2_config.txt. Regenerate or sweep it with:"
Write-Host "    python scripts/make_synthetic.py --barrier-h 9 --n-replicas 16"
Write-Host ""
Write-Host "Real-world REST2 VALIDATION datasets (open the links; respect each license):"
Write-Host "  * Shaw millisecond folding trajectories -- by request/collaboration; not redistributable."
Write-Host "  * SAMPL challenges      : https://github.com/samplchallenges/SAMPL  (open)"
Write-Host "  * GPCRmd REST2 data     : https://gpcrmd.org                        (web access; site terms)"
Write-Host "  * Chignolin / Trp-cage fast-folder benchmarks -- public sequences; standard REMD test systems."
Write-Host ""
Write-Host "None of these is required for the demo. For credentialed sets, register at the"
Write-Host "source FIRST; this script will never bypass authentication (CLAUDE.md section 8)."
