# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.9 -- Phylogenetic Likelihood / Tree Inference
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URL +
# what to do, and NEVER bypasses credentials/registration. The committed tiny
# SYNTHETIC sample in data/sample/ already runs the demo offline; this script
# only points at the real curated databases and defers to make_synthetic.py for
# a larger offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.9 -- Phylogenetic Likelihood / Tree Inference"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a tiny SYNTHETIC sample (data/sample/phylo_sample.txt)"
Write-Host "that is sufficient to build, run, and verify the demo offline."
Write-Host ""
Write-Host "Real curated phylogenetic alignments / trees (study these next):"
Write-Host "  * TreeBASE       https://www.treebase.org/        (alignments + trees)"
Write-Host "  * SILVA rRNA     https://www.arb-silva.de/        (large rRNA alignment)"
Write-Host "  * NCBI CDD       https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml"
Write-Host "  * Open Tree      https://opentreeoflife.github.io/ (aggregated phylogenies)"
Write-Host ""
Write-Host "These arrive as FASTA/PHYLIP/NEXUS alignments with Newick trees. Converting"
Write-Host "one to this project's compact text format (encode bases A/C/G/T->0..3, write"
Write-Host "a POST-ORDER node list; see data/README.md) is left as a README exercise."
Write-Host "Respect each source's license; none is redistributed here."
Write-Host ""
Write-Host "For a larger OFFLINE synthetic problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --n-sites 50000 --seed 7"
