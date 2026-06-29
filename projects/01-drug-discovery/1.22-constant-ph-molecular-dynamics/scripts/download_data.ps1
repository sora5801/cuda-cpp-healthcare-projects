# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. This project ships a committed
# SYNTHETIC sample (data/sample/cph_system.txt) that is sufficient for the demo,
# so there is nothing to auto-download; this script prints where to obtain the
# real titration benchmarks and defers to make_synthetic.py for offline use.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.22 -- Constant-pH Molecular Dynamics"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs entirely on the committed SYNTHETIC sample:"
Write-Host "    data/sample/cph_system.txt   (regenerate: python scripts/make_synthetic.py)"
Write-Host ""
Write-Host "There is NO automatic download. To extend the project with real"
Write-Host "experimental pKa values, obtain them yourself from:"
Write-Host "  * PKAD  -- experimental protein-residue pKa database:"
Write-Host "      https://compbio.clemson.edu/pkad/"
Write-Host "  * PHMD / benchmark pKa sets for Asp/Glu/His/Cys/Lys residues"
Write-Host "      (see the constant-pH MD literature, e.g. AMBER CpHMD papers)."
Write-Host "  * DrugBank -- ionizable-group compounds (registration required):"
Write-Host "      https://go.drugbank.com"
Write-Host ""
Write-Host "Respect each source's license. For credentialed sets, register on the"
Write-Host "site -- this script will NOT bypass authentication. Map a real residue"
Write-Host "into the toy by editing pKa_intrinsic / charges / positions in the"
Write-Host "sample file (field meanings are in data/README.md)."
