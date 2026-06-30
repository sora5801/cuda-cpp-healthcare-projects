# ===========================================================================
# scripts/download_data.ps1  --  Fetch a FULL Hi-C dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.15 : Hi-C / 3D Genome Contact Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# format notes, and NEVER bypasses credentials/registration. Public Hi-C matrices
# are large and distributed as binary .hic / .mcool files that need a converter
# (cooler / hic2cool / Juicer Tools), so this script PRINTS the exact, vetted
# steps rather than blindly downloading multi-GB files. The committed synthetic
# sample already lets the demo run offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.15 -- Hi-C / 3D Genome Contact Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny sample in data/sample/hic_sample.txt is enough to run"
Write-Host "the demo offline. To work with REAL Hi-C contact maps, use a public source:"
Write-Host ""
Write-Host "  * 4DN Data Portal   https://data.4dnucleome.org/   (.mcool, many cell types)"
Write-Host "  * ENCODE            https://www.encodeproject.org/  (cell-line Hi-C)"
Write-Host "  * GEO GSE63525      https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525"
Write-Host "                      (Rao et al. 2014, the classic high-resolution maps)"
Write-Host ""
Write-Host "Real maps ship as binary .hic (Juicer) or .mcool (cooler). Convert one bin"
Write-Host "resolution to the plain 'i j count' text this project reads, e.g. with cooler:"
Write-Host ""
Write-Host "    pip install cooler"
Write-Host "    # dump one chromosome of a .mcool at 100 kb as COO upper triangle:"
Write-Host "    cooler dump -t pixels --join -r chr1 my_map.mcool::/resolutions/100000 \"
Write-Host "      | awk '{print \$2/100000, \$5/100000, \$7}' > data/full/chr1_100kb.txt"
Write-Host ""
Write-Host "  (Add an 'n nnz' header line matching this project's loader format; see"
Write-Host "   data/README.md. Bin indices must be 0-based and upper-triangular i<=j.)"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead (no download), run:"
Write-Host "    python scripts/make_synthetic.py --out data/full/synthetic_big.txt"
Write-Host ""
Write-Host "[download_data] Nothing downloaded (by design). Follow the steps above for real data."
