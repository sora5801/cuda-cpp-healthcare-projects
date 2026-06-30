# ===========================================================================
# scripts/download_data.ps1  --  Guidance for fetching REAL backbones (Windows)
# ---------------------------------------------------------------------------
# Project 2.10 : Protein Design / Inverse Folding Inference
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The committed SYNTHETIC sample in
# data/sample/ is enough to run and verify the demo; this script explains where
# REAL protein backbones come from and how you would wire one in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.10 -- Protein Design / Inverse Folding Inference"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC backbone (data/sample/backbone_sample.txt)."
Write-Host "No download is required to run the demo. To experiment with REAL structures:"
Write-Host ""
Write-Host "  PDB  (https://www.rcsb.org)   -- experimental protein structures."
Write-Host "       Fetch one entry's coordinates, e.g. small protein 1UBQ (ubiquitin):"
Write-Host "         Invoke-WebRequest https://files.rcsb.org/download/1UBQ.pdb -OutFile `"$DataDir\1UBQ.pdb`""
Write-Host "       Then parse its CA atom records into '<x> <y> <z> <native_letter>'"
Write-Host "       lines (one residue per line) -- see data/README.md. PDB data is"
Write-Host "       freely redistributable for most entries (verify per entry)."
Write-Host ""
Write-Host "  CATH (https://www.cathdb.info)              -- 500k+ classified domains."
Write-Host "  ProteinGym (https://github.com/OATML-Markslab/ProteinGym) -- fitness sets."
Write-Host "  CAMEO (https://www.cameo3d.org)             -- fresh validation backbones."
Write-Host ""
Write-Host "  For a larger SYNTHETIC backbone instead, run:"
Write-Host "    python scripts/make_synthetic.py --shells 40 --per 12"
Write-Host ""
Write-Host "  Idempotent-download pattern when wiring a real fetch:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
