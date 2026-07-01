# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.13 : Photoacoustic Image Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. If a
# dataset needs an account, this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline stand-in.
#
# NOTE: this project needs NO download to run -- the committed synthetic sample
# (data/sample/pa_sample.txt) is generated locally by make_synthetic.py. The
# pointers below are for learners who want real photoacoustic data.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.13 -- Photoacoustic Image Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo needs NO download: data/sample/pa_sample.txt is synthetic and"
Write-Host "regenerable with:  python scripts/make_synthetic.py"
Write-Host "For a bigger synthetic problem:  python scripts/make_synthetic.py --sensors 256 --samples 1024 --img 256"
Write-Host ""
Write-Host "To study REAL photoacoustic data, see (verify URLs; respect each license):"
Write-Host "  * k-Wave toolbox + example datasets .......... http://www.k-wave.org/"
Write-Host "  * k-Wave CUDA fluid solver ................... https://github.com/klepo/k-Wave-Fluid-CUDA"
Write-Host "  * PyTomography (GPU tomography incl. PA) ..... https://github.com/lukepolson/pytomography"
Write-Host "  * In-vivo PA datasets in open-access Nature Communications papers"
Write-Host "  * PASCAA / IPASC challenge data .............. photoacoustics.org (verify URL)"
Write-Host ""
Write-Host "Credentialed/registration-gated sets: this script will NOT bypass a login."
Write-Host "Register at the source, then place files under data/ yourself."
