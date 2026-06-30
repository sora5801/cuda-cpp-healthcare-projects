# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# licensing, and NEVER bypasses credentials/registration. The committed tiny
# SYNTHETIC sample (data/sample/complex_sample.txt) already runs the demo
# offline, so this script only points at the real co-folding benchmarks for
# learners who want to go further. It downloads nothing automatically because
# those benchmarks carry their own licenses and are large.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.14 -- Protein-Ligand Co-Folding"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/complex_sample.txt) is SYNTHETIC and"
Write-Host "is all the demo needs. The real co-folding benchmarks below are for"
Write-Host "further study. This project's loader expects its own tiny token format"
Write-Host "(see data/README.md); turning a real PDB complex into that format is"
Write-Host "left as an exercise -- the point here is the diffusion+attention loop,"
Write-Host "not a full structure parser."
Write-Host ""
Write-Host "Real protein-ligand complex benchmarks (study these):"
Write-Host "  * PoseBusters  : 428 recent PDB complexes for pose validation"
Write-Host "                   https://github.com/maabuu/posebusters  (MIT; PDB data CC0-ish, check per entry)"
Write-Host "  * PDBbind v2020: protein-ligand complexes + binding affinities"
Write-Host "                   http://www.pdbbind.org.cn  (registration required; academic license)"
Write-Host "  * Astex Diverse: 85 drug-like ligand complexes"
Write-Host "                   https://www.ccdc.cam.ac.uk (verify current URL / terms)"
Write-Host "  * CASF         : cross-docking scoring benchmarks"
Write-Host "                   http://www.pdbbind.org.cn/casf.php"
Write-Host ""
Write-Host "For a larger SYNTHETIC complex (more tokens / steps), run:"
Write-Host "  python scripts/make_synthetic.py --n-protein 24 --n-ligand 9 --steps 240"
Write-Host ""
Write-Host "NOTE: PDBbind/Astex require accepting a license or registering. This"
Write-Host "script will NOT bypass that -- follow each site's instructions yourself."
