# ===========================================================================
# scripts/download_data.ps1  --  Fetch real condensate references (Windows)
# ---------------------------------------------------------------------------
# Project 2.34 : Biophysical Simulation of Biomolecular Condensates
#                (Active Learning Loop)  --  reduced-scope teaching version
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs,
# and NEVER bypasses credentials/registration. This project's DEMO needs no
# download at all -- its input is a synthetic configuration file written by
# scripts/make_synthetic.py. This script instead points at the public datasets a
# learner would use to go from this teaching toy toward a real active-learning
# loop, and explains why none of them is auto-fetched here.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.34 : Biomolecular Condensates (Active Learning Loop)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This demo needs NO download: the input is a synthetic ensemble config"
Write-Host "(data/sample/condensate_ensemble.txt), regenerated any time with:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host "    python scripts/make_synthetic.py --n-members 200   # a larger sweep"
Write-Host ""
Write-Host "Public references for a REAL condensate active-learning loop (study these;"
Write-Host "they are not auto-downloaded -- formats vary and some require agreement to"
Write-Host "terms of use, which this script will not bypass):"
Write-Host "  * PhaSePro  -- curated LLPS proteins/regions : https://phasepro.elte.hu"
Write-Host "  * DisProt   -- intrinsically disordered regions : https://disprot.org"
Write-Host "  * RCSB PDB  -- structures of FUS/TDP-43/hnRNPA1 LC domains : https://www.rcsb.org"
Write-Host "  * CALVADOS  -- residue-level IDP CG model + params : https://github.com/KULL-Centre/CALVADOS"
Write-Host ""
Write-Host "Idempotent pattern to follow when wiring any real fetch:"
Write-Host "  1) skip the download if the file already exists with the right SHA256"
Write-Host "  2) print source URL + expected size + checksum before downloading"
Write-Host "  3) for credentialed/registered sources, print instructions ONLY"
