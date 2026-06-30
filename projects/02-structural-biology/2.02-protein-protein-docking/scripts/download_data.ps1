# ===========================================================================
# scripts/download_data.ps1  --  Real protein-complex sources (Windows)
# ---------------------------------------------------------------------------
# Project 2.2 : Protein-Protein Docking. Downloads nothing automatically.
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs,
# and NEVER bypasses credentials/registration. The committed synthetic sample is
# enough to run the demo offline; this script points to the real benchmarks.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 2.2 -- Protein-Protein Docking"
Write-Host ""
Write-Host "Real rigid-body docking benchmarks (free for research; please cite):"
Write-Host "  Docking Benchmark 5.5 : https://zlab.umassmed.edu/benchmark/"
Write-Host "                          230 non-redundant complexes (bound + unbound)."
Write-Host "  SAbDab (antibodies)   : https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/"
Write-Host "  PDB (any complex)     : https://www.rcsb.org   (split chains -> receptor/ligand)"
Write-Host ""
Write-Host "To dock a real complex with this code:"
Write-Host "  1) download a PDB/mmCIF complex and split it into two chains."
Write-Host "  2) write each chain's atoms as 'x y z' lines (Angstrom)."
Write-Host "  3) prepend a header 'n_recv n_lig N spacing' (OMIT the known-answer"
Write-Host "     fields -- a real complex has no pre-known rigid translation)."
Write-Host "  See data/README.md for the exact file format."
Write-Host ""
Write-Host "No-download synthetic option (works offline):"
Write-Host "  python scripts/make_synthetic.py --N 48 --spacing 1.5"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
