# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URL,
# and NEVER bypasses credentials/registration. This project's demo runs on a
# committed SYNTHETIC sample (data/sample/complex_sample.txt), so there is no
# mandatory download -- this script explains how to obtain REAL complexes from
# the PDB and how the offline sample is (re)generated.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.21 -- Protein-Nucleic Acid Docking & Co-Folding"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project needs NO download to run: data/sample/complex_sample.txt"
Write-Host "is a committed SYNTHETIC complex with a known native pose."
Write-Host ""
Write-Host "To regenerate (or resize) the synthetic sample:"
Write-Host "    python scripts/make_synthetic.py --spacing 3500"
Write-Host ""
Write-Host "To work with REAL protein-nucleic-acid complexes:"
Write-Host "  * Protein Data Bank (PDB): https://www.rcsb.org"
Write-Host "      Download a structure, e.g. 1FNT, as mmCIF/PDB:"
Write-Host "      https://files.rcsb.org/download/1FNT.cif"
Write-Host "  * RNA-Puzzles benchmarks:  https://github.com/RNA-Puzzles"
Write-Host "  * Rfam RNA families:       https://rfam.org"
Write-Host ""
Write-Host "  You must convert a downloaded structure into this loader's integer"
Write-Host "  format (extract atoms, assign charge signs in {-1,0,+1}, scale"
Write-Host "  coordinates to milli-Angstrom). See data/README.md for the format."
Write-Host "  No registration or credentials are required for the public PDB; if a"
Write-Host "  benchmark needs an account, follow its site's instructions -- this"
Write-Host "  script will not bypass them."
