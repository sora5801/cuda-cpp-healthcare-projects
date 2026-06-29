# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL structures for SASA (Windows)
# ---------------------------------------------------------------------------
# Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
#
# This project's "real data" is a molecular structure (a PDB file) converted to
# the simple "<element> x y z" format the loader reads. This script prints the
# recipe; it does not require credentials and downloads nothing on its own beyond
# an OPTIONAL public PDB fetch you can uncomment. It defers to make_synthetic.py
# for the fully-offline stand-in (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.31 -- Solvent-Accessible Surface Area (SASA) on GPU"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real SASA runs on an actual structure. To prepare one:"
Write-Host "  1) Pick a PDB id, e.g. 1CRN (crambin, a 46-residue test protein)."
Write-Host "  2) Download the structure (public, no login) from RCSB:"
Write-Host "       https://files.rcsb.org/download/1CRN.pdb"
Write-Host "     (Optional one-liner you can run yourself:)"
Write-Host "       Invoke-WebRequest https://files.rcsb.org/download/1CRN.pdb -OutFile `"$DataDir\1CRN.pdb`""
Write-Host "  3) Convert ATOM/HETATM records to '<element> x y z' (Angstrom). The"
Write-Host "     element is PDB columns 77-78; coords are columns 31-54. Tools that"
Write-Host "     do this cleanly: Biopython (Bio.PDB) or MDTraj."
Write-Host "  4) Validate your SASA against FreeSASA (https://github.com/mittinatten/freesasa)."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible):"
Write-Host "  python scripts/make_synthetic.py    # writes data/sample/molecule_sample.xyz"
Write-Host ""
Write-Host "Idempotency note: when wiring a real fetch, skip the download if the file"
Write-Host "already exists with the expected SHA256, and NEVER bypass any registration."
